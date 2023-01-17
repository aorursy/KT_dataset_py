import glob, os, yaml, csv
import numpy as np
from cv2 import imread , resize
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Input, Flatten, Reshape, Dense,Conv2DTranspose, Add
from keras import Model, callbacks
import matplotlib.pyplot as plt
%matplotlib inline
#We read the images and store them in numpy arrays
X=np.array([imread(each) for each in glob.glob(os.path.join(os.getcwd() , '../input/train/*.png'))[:4]])
y=np.array([imread(each) for each in glob.glob(os.path.join(os.getcwd() , '../input/train_cleaned/*.png'))[:4]])
plt.figure(figsize=(16, 8))
for i in range(4):
    plt.subplot(241+i)
    fig=plt.imshow(X[i])
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.subplot(245+i)
    fig=plt.imshow(y[i])
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
plt.show()
shapes=np.unique([imread(each).shape for each in glob.glob(os.path.join(os.getcwd() , '../input/train/*.png'))],axis=0)
print(shapes)
class Autoencoder:
    def __init__(self,
                dimensions_factor=1,
                layers=4,
                k=3,
                filter_size=None,
                pooling_factor=None,
                only_decoder=False,
                only_encoder=False,
                loss='mean_squared_error',
                channels=3,
                start_filters=32,
                skip_connection=True):
        
        #Initialization of non-specified variables
        self.channels=channels
        self.layers=layers
        self.k=k
        self.filter_size=[(k,k)]*(layers*2) if (filter_size is None) else filter_size
        self.pooling_factor=[2]*(layers*2) if (pooling_factor is None) else pooling_factor
        self.dimensions_factor=dimensions_factor
        self.model_type='conv_AE'
        self.start_filters=start_filters
        self.encoder_layers=[]
        self.skip_connection=skip_connection
        
        #Build encoder
        self.input_img = Input(shape=(None, None, self.channels))
        x=self.input_img
        self.encoder_layers.append(self.input_img)
        for i in range(self.layers):
            filters=int(self.pooling_factor[i]*int(x.shape[-1])*self.dimensions_factor) if i != 0 else self.start_filters
            x = Conv2D(filters, self.filter_size[i], activation='relu', padding='same',name='encoding_conv_'+str(i))(x)
            self.encoder_layers.append(x)
            x = MaxPooling2D((self.pooling_factor[i], self.pooling_factor[i]), padding='valid',name='encoding_pool_'+str(i))(x)

        x = Conv2D(int(x.shape[-1]), (1,1), activation='relu', padding='same',name='code')(x)
        self.coded=x
        for i in range(layers):
            filters=int(self.encoder_layers[-(i+1)].shape[-1])#int(int(x.shape[-1])/(self.pooling_factor[i])*self.dimensions_factor)
            x = Conv2DTranspose(filters, self.filter_size[i], activation='relu', strides=self.pooling_factor[i], padding='same',name='decoding_conv_'+str(i))(x)
            if self.skip_connection:
                x = Add()([x,self.encoder_layers[-(i+1)]])
        x = Conv2D(self.channels, (3), activation='sigmoid', padding='same',name='decoded')(x)
        self.decoded=x
        autoencoder = Model(self.input_img, self.decoded)
        autoencoder.compile(optimizer='adam', loss=loss, metrics=['mse'])

        self.model=autoencoder

    def get_encoder(self):
        if self.model is None:
            raise('Model is not created')
        else:
            encoder=Model(self.input_img, self.coded)

            return encoder

    def get_decoder(self):
        if self.model is None:
            raise('Model is not created')
        else:
            decoder=Model(self.coded, self.decoded)

            return decoder
class Image_generator():
    def __init__(self,path,val_percentage=0.115,batch_size=4, reduction_factor=16):
        self.base_path=path
        self.batch_size=batch_size
        self.reduction_factor=reduction_factor
        
        self.y=np.array([imread(each).astype('float32') / 255 for each in sorted(glob.glob(os.path.join(path, '../input/train_cleaned/*.png')))])
        self.X=np.array([imread(each).astype('float32') / 255 for each in sorted(glob.glob(os.path.join(path , '../input/train/*.png')))])

        self.idx_train,self.idx_val = self.split_batches(val_percentage)
        
        self.train_steps = len(self.idx_train)
        self.val_steps = len(self.idx_val)
    
    def split_batches(self,val_percentage):
        idx=[]
        self.shapes=np.unique([each.shape for each in self.X],axis=0)
        for shape in self.shapes:
            shape_idx=np.argwhere([all(a.shape==shape) for a in self.X]).flatten()
            np.random.shuffle(shape_idx)
            idx.extend(np.array_split(shape_idx, len(shape_idx)/self.batch_size))
        idx=np.array([x for x in idx if x.size == self.batch_size])
        np.random.shuffle(idx)
        samples=len(idx)
        val_samples=int(samples*val_percentage)
        
        return idx[:-val_samples],idx[-val_samples:]

    def check_size(self,batch_x,batch_y,axis):
        for each_axis in axis:
            while (batch_x.shape[each_axis]%self.reduction_factor)!=0:
                batch_x=np.insert(batch_x,batch_x.shape[each_axis],1,axis=each_axis)
                batch_y=np.insert(batch_y,batch_y.shape[each_axis],1,axis=each_axis)
        return batch_x,batch_y

    def get_train_batch(self):
        while True:
            batch_x = np.stack(self.X[self.idx_train[0]])
            batch_y = np.stack(self.y[self.idx_train[0]])
            batch_x, batch_y = self.check_size(batch_x, batch_y,[-2,-3])
            self.idx_train = np.roll(self.idx_train,1,axis=0)
            yield batch_x, batch_y

    def get_val_batch(self):
        while True:
            batch_x = np.stack(self.X[self.idx_val[0]])
            batch_y = np.stack(self.y[self.idx_val[0]])
            batch_x, batch_y = self.check_size(batch_x, batch_y,[-2,-3])
            self.idx_val = np.roll(self.idx_val,1,axis=0)
            yield batch_x, batch_y
np.random.seed(42)
data_generator=Image_generator(os.getcwd())
autoencoder=Autoencoder(loss='binary_crossentropy',skip_connection=True)
log=autoencoder.model.fit_generator(generator = data_generator.get_train_batch(),
                                  steps_per_epoch=data_generator.train_steps,
                                  epochs=50,
                                  shuffle=False,
                                  validation_data = data_generator.get_val_batch(),
                                  validation_steps = data_generator.val_steps,
                                  use_multiprocessing = False)

plt.figure()
[plt.plot(v,label=str(k)) for k,v in log.history.items()]
plt.legend()
plt.show()
def to_csv(npdata,ids):
    with open('submission.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(('id','value'))
        for i,each in enumerate(npdata):
            rows,cols,_=each.shape
            for row in range(rows):
                for col in range(cols):
                    id_pixel=str(ids[i])+'_'+str(row+1)+'_'+str(col+1)
                    value_pixel=str(np.mean(each[row,col,:]))
                    csvwriter.writerow([id_pixel,value_pixel])

X_test=np.array([imread(each) for each in sorted(glob.glob(os.path.join(os.getcwd() , '../input/test/*.png')))])
ids=[each.split('/')[-1][:-4] for each in sorted(glob.glob(os.path.join(os.getcwd() , '../input/test/*.png')))]
predictions=[]
for x in X_test:
    original_shape=x.shape
    while (x.shape[0]%16)!=0:
        x=np.insert(x,x.shape[0],1,axis=0)
    while (x.shape[1]%16)!=0:
        x=np.insert(x,x.shape[1],1,axis=1)
    x=x.reshape((1,)+x.shape)
    
    prediction=autoencoder.model.predict(x)
    prediction=prediction.reshape(prediction.shape[1:])
    prediction=prediction[:original_shape[0],:original_shape[1],:]
    
    predictions.append(prediction)
    print('%d of %d predictions calculated'%(len(predictions),len(X_test)), end='\r') 
print('\nSaving...')
to_csv(np.array(predictions),ids)
print('Saved')

indexes=[4,5,6,7]
fig=plt.figure(figsize=(10, 15))
for i,idx in enumerate(indexes):
    fig.add_subplot(len(indexes), 2, (i)*2+1)
    plt.imshow(X_test[i])
    fig.add_subplot(len(indexes), 2, (i)*2+2)
    plt.imshow(predictions[i])
plt.show()
encoder= autoencoder.get_encoder()

example=X_test[10].reshape((1,)+X_test[10].shape)
filters=encoder.predict(example)

plt.figure(1)
plt.imshow(example.reshape(example.shape[1:4]))

%%capture
import matplotlib.animation as animation
from IPython.display import HTML

fig=plt.figure(2)
ims=[]
for i in range(filters.shape[-1]):
    im = plt.imshow(filters[:,:,:,i].reshape(filters.shape[1:3]), animated=True)
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,
                                repeat_delay=0)
HTML(ani.to_jshtml())

