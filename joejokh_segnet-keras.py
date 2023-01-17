import os

from keras.layers import Activation,BatchNormalization,Conv2D

from keras.engine.topology import Layer

from keras.models import Model

from keras.layers import Activation,Input,ZeroPadding2D,Cropping2D

from keras import backend as K

import tensorflow as tf

from keras.models import load_model

import numpy as np

from keras.layers import UpSampling2D,MaxPooling2D

import cv2

from sklearn.utils import shuffle

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

import matplotlib.pyplot as plt

from keras.utils import plot_model
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
class MaxPoolingWithIndices(Layer):

    def __init__(self, pool_size,strides,padding='SAME',**kwargs):

        super(MaxPoolingWithIndices, self).__init__(**kwargs)

        self.pool_size=pool_size

        self.strides=strides

        self.padding=padding

        return

    def call(self,x):

        pool_size=self.pool_size

        strides=self.strides

        if isinstance(pool_size,int):

            ps=[1,pool_size,pool_size,1]

        else:

            ps=[1,pool_size[0],pool_size[1],1]

        if isinstance(strides,int):

            st=[1,strides,strides,1]

        else:

            st=[1,strides[0],strides[1],1]

        output1,output2=tf.nn.max_pool_with_argmax(x,ps,st,self.padding)

        return [output1,output2]

    def compute_output_shape(self, input_shape):

        if isinstance(self.pool_size,int):

            output_shape=(input_shape[0],input_shape[1]//self.pool_size,input_shape[2]//self.pool_size,input_shape[3])

        else:

            output_shape=(input_shape[0],input_shape[1]//self.pool_size[0],input_shape[2]//self.pool_size[1],input_shape[3])

        return [output_shape,output_shape]





class UpSamplingWithIndices(Layer):

    def __init__(self, **kwargs):

        super(UpSamplingWithIndices, self).__init__(**kwargs)

        return

    def call(self,x):

        argmax=K.cast(K.flatten(x[1]),'int32')

        max_value=K.flatten(x[0])

        print(x[0],x[1])

        with tf.compat.v1.variable_scope(self.name):

            input_shape=K.shape(x[0])

            batch_size=input_shape[0]

            image_size=input_shape[1]*input_shape[2]*input_shape[3]

            output_shape=[input_shape[0],input_shape[1]*2,input_shape[2]*2,input_shape[3]]

            indices_0=K.flatten(tf.compat.v1.matmul(K.reshape(tf.compat.v1.range(batch_size),(batch_size,1)),K.ones((1,image_size),dtype='int32')))

            indices_1=argmax%(image_size*4)//(output_shape[2]*output_shape[3])

            indices_2=argmax%(output_shape[2]*output_shape[3])//output_shape[3]

            indices_3=argmax%output_shape[3]

            indices=tf.compat.v1.stack([indices_0,indices_1,indices_2,indices_3])

            output=tf.compat.v1.scatter_nd(K.transpose(indices),max_value,output_shape)

            

            return output

        

    def compute_output_shape(self, input_shape):

        return input_shape[0][0],input_shape[0][1]*2,input_shape[0][2]*2,input_shape[0][3]



def CompositeConv(inputs,num_layers,num_features):

    output=inputs

    if isinstance(num_features,int):

        for i in range(num_layers):

            output=Conv2D(num_features,(3,3),activation='relu', padding='same',)(output)

        return output
VGG_Weights_path = "/kaggle/input/seg-data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

def create_model():

    inputs=Input(shape=(224,224,3))





    x=CompositeConv(inputs,2,64)

    #x,argmax1=MaxPoolingWithIndices(pool_size=2,strides=2)(x)

    x=MaxPooling2D(padding='SAME')(x)

    

    

    x=CompositeConv(x,2,128)

    #x,argmax2=MaxPoolingWithIndices(pool_size=2,strides=2)(x)

    x=MaxPooling2D(padding='SAME')(x)

    

    x=CompositeConv(x,3,256)

    #x,argmax3=MaxPoolingWithIndices(pool_size=2,strides=2)(x)

    x=MaxPooling2D(padding='SAME')(x)

    

    x=CompositeConv(x,3,512)

    #x,argmax4=MaxPoolingWithIndices(pool_size=2,strides=2)(x)

    x=MaxPooling2D(padding='SAME')(x)

    

    x=CompositeConv(x,3,512)

    #x,argmax5=MaxPoolingWithIndices(pool_size=2,strides=2)(x)

    x=MaxPooling2D(padding='SAME')(x)

    

    vgg  = Model(  inputs , x  )

    vgg.load_weights(VGG_Weights_path)

    

    #print(argmax5.shape, x.shape)

    #x=UpSamplingWithIndices()([x,argmax5])

    x=UpSampling2D()(x)

    x=CompositeConv(x,3,512)

    

    #print(argmax4.shape, x.shape)

    #x=UpSamplingWithIndices()([x,argmax4])

    x=UpSampling2D()(x)

    x=CompositeConv(x,3,512)

    

    #print(argmax3.shape, x.shape)

    #x=UpSamplingWithIndices()([x,argmax3])

    x=UpSampling2D()(x)

    x=CompositeConv(x,3,256)

    

    #print(argmax2.shape, x.shape)

    #x=UpSamplingWithIndices()([x,argmax2])

    x=UpSampling2D()(x)

    x=CompositeConv(x,2,128)

    

    #print(argmax1.shape, x.shape)

    #x=UpSamplingWithIndices()([x,argmax1])

    x=UpSampling2D()(x)

    x=CompositeConv(x,2,64)

    

    o=Conv2D(3,(3,3),padding='same',activation='softmax')(x)

    #o=Activation('softmax')(x)



    my_model=Model(inputs=inputs,outputs=o)

    

    return my_model

my_model=create_model()

my_model.summary()

#plot_model(my_model, show_shapes=True, to_file='SegNet.png')


dir_img = "/kaggle/input/db-ter-m2/DB_TER_M2/potato"

dir_seg = "/kaggle/input/db-ter-m2/DB_TER_M2/potato_seg/"



ldimg = np.array(os.listdir(dir_img))



X = []

Y = []

#pour chaque image

for path in ldimg:

    #lire une image, redimentiner l'image

    img_path = os.path.join(dir_img, path)

    img = cv2.imread(img_path)

    img = np.float32(cv2.resize(img, ( 224 , 224 )))/255.

    X.append(img)



    ##lire l'image ségmenter corespondante, redimentiner l'image

    seg_path = os.path.join(dir_seg, path.replace(".JPG","_final_masked.jpg"))

    seg_img = cv2.imread(seg_path)

    seg_img = np.float32(cv2.resize(seg_img, ( 224 , 224 )))

    

    #modifier l'image vers catégorical selon la ségmentation fait 

    #blanc => zone infecté

    #noir => background

    #autrement => zone saine 

    

    seg_labels = np.zeros(shape=(  224 , 224  , 3 ))

    seg_img = seg_img[:, : , 0]

    

    #blanc == 255 , zone infecté, chanel 0

    seg_labels[: , : , 0 ] = (seg_img == 255.).astype(int)

    

    #noir == 0 , background, chanel 2

    seg_labels[: , : , 2 ] = (seg_img == 0.).astype(int)

    

    #autrement zone saine, chanel 1

    seg_labels[: , : , 1 ] = (seg_img != 255.).astype(int) * (seg_img != 0.).astype(int)

    

    Y.append(seg_labels)

    

X, Y = np.array(X) , np.array(Y)

print(X.shape,Y.shape)





train_rate = 0.75

index_train = np.random.choice(X.shape[0],int(X.shape[0]*train_rate),replace=False)

index_test  = list(set(range(X.shape[0])) - set(index_train))

                            

X, Y = shuffle(X,Y)

X_train, y_train = X[index_train],Y[index_train]

X_test, y_test = X[index_test],Y[index_test]

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)



#Augmentation des données





# we create two instances with the same arguments

data_gen_args = dict(horizontal_flip=True,

                     vertical_flip=True)

seed = 1

#tarin

image_datagen = ImageDataGenerator(**data_gen_args)

mask_datagen = ImageDataGenerator(**data_gen_args)



image_generator_train = image_datagen.flow(X_train, batch_size=16, seed=seed)

mask_generator_train = mask_datagen.flow(y_train, batch_size=16, seed=seed)

train_generator = zip(image_generator_train, mask_generator_train)



#test

image_datagen_test = ImageDataGenerator(**data_gen_args)

mask_datagen_test = ImageDataGenerator(**data_gen_args)



image_generator_test = image_datagen.flow(X_test, batch_size=32, seed=seed)

mask_generator_test = mask_datagen.flow(y_test, batch_size=32, seed=seed)

test_generator = zip(image_generator_test, mask_generator_test)

num_classes=3

batch_size=32

epochs=200

loss_function='categorical_crossentropy'

metrics=['accuracy']



my_model=create_model()    



sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)

my_model.compile(sgd,loss=loss_function,metrics=metrics)

hist1 = my_model.fit_generator(train_generator,

                           steps_per_epoch=64,

                           epochs=200,validation_data=test_generator,

                           validation_steps=64)



my_model.save('SEGNET-200_aug.h5')

for key in ['loss', 'val_loss','accuracy','val_accuracy']:

    plt.plot(hist1.history[key],label=key)

plt.legend()

plt.savefig('history_test_aug.png')

plt.show()
def get_proportion_disease(seg_img):

    return (np.sum(seg_img[: , : , 0 ])/(np.sum(seg_img[: , : , 0 ])+np.sum(seg_img[: , : , 1 ])))*100
y_pred = my_model.predict_generator(test_generator,steps=16)



for i in range(10):

    img_is  = (X_test[i])

    seg = y_pred[i]

    segtest = y_test[i]



    fig = plt.figure(figsize=(10,30))    

    ax = fig.add_subplot(1,3,1)

    ax.imshow(img_is)

    ax.set_title("original")

    

    ax = fig.add_subplot(1,3,2)

    ax.imshow(seg)

    ax.set_title("predicted class")

    ax.text(180, 0, "PoD = {0:.1f}%".format(get_proportion_disease(seg)),fontsize=10)

    

    ax = fig.add_subplot(1,3,3)

    ax.imshow(segtest)

    ax.set_title("true class")

    ax.text(180, 0, "PoD = {0:.1f}%".format(get_proportion_disease(segtest)),fontsize=10)

    plt.savefig('test_augm'+str(i)+'.png')

    plt.show()
print(my_model.evaluate_generator(test_generator,steps=16))