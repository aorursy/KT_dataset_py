import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from ast import literal_eval
import time
import os
import keras as k
from keras import backend as K
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
def space_replace(s):
    return s.replace(' ', '_')

folder_name = '../input/train_simplified/'
files= os.listdir(folder_name)
files= [space_replace(os.path.splitext(item)[0]) for item in files]

files_to_id = {j:i for i,j in enumerate(files)}
get_file_name = {j:i for i,j in zip(files_to_id.keys(), files_to_id.values())}


def replace_name(item):
    item= space_replace(item)
    return files_to_id[item]
files= os.listdir(folder_name)
num_samp= 1000     #Number of lines to read in each csv every iteration
num_lines= 10000     #Number of lines from each csv to include in the final dataframe max on average 100k
num_blocks = int(num_lines / num_samp)
dataframe_length= 0
for block in range(num_blocks):
    dataframe = pd.DataFrame()
    block_idx = list(np.arange(1,num_samp*block))
    for item in files:
        file_path = os.path.join(folder_name, item)
        df= pd.read_csv(file_path, 
                        usecols=['drawing', 'recognized', 'word'],
                        dtype={'drawing':str, 'recognized':str, 'word':str},
                        nrows= num_samp, skiprows= block_idx, low_memory=False)
        df= df.loc[df.recognized=='True']
        dataframe = dataframe.append(df, ignore_index=True)
    dataframe= dataframe.sample(frac=1).reset_index(drop=True)
    dataframe_length += len(dataframe)
    dataframe.to_csv('training_data_10k.csv', columns=['drawing', 'word'], index=False, header=False, mode='a')
    print ('Done processing block: '+ str(block), end='\r')
def get_path_signatures(seq):
    # An implementation of path signatures based 
    # on https://arxiv.org/pdf/1603.03788.pdf
    # Here 1 dimensional paths were chosen

    
    seq = literal_eval(seq)
    reshaped_seq = []
    for i in range(len(seq)):
        reshaped_seq.append(np.array(list(zip(*seq[i]))))
    
    all_arrays= np.concatenate(reshaped_seq)
       
    path01= all_arrays[:,0]
    path02= all_arrays[:,1]
    
    path1, path2, = [],[]
    path11, path12, path21, path22 = [],[],[],[]
    path111, path222 = [], []
    path1111, path2222 = [], []
    
    for i in range(len(all_arrays)-1):
        path1.append(all_arrays[i+1,0] - all_arrays[i,0])
        path2.append(all_arrays[i+1,1] - all_arrays[i,1])
        path11.append(0.5*((all_arrays[i+1,0] - all_arrays[i,0])**2))
        path111.append(((all_arrays[i+1,0] - all_arrays[i,0])**3)/6)
        path1111.append(((all_arrays[i+1,0] - all_arrays[i,0])**4)/24)
        path22.append(0.5*((all_arrays[i+1,1] - all_arrays[i,1])**2))
        path222.append(((all_arrays[i+1,1] - all_arrays[i,1])**3)/6)
        path2222.append(((all_arrays[i+1,1] - all_arrays[i,1])**4)/24)
    
    path1 = np.array(path1)
    path1 = np.append(path1,[0])
    path2 = np.array(path2)
    path2 = np.append(path2,[0])
    path11 = np.array(path11)
    path11 = np.append(path11,[0])
    path111 = np.array(path111)
    path111 = np.append(path111,[0])
    path1111 = np.array(path1111)
    path1111 = np.append(path1111,[0])
    path22 = np.array(path22)
    path22 = np.append(path22,[0])
    path222 = np.array(path222)
    path222 = np.append(path222,[0])
    path2222 = np.array(path2222)
    path2222 = np.append(path2222,[0])
    
    stacked =  np.transpose(
        np.stack((path01, path02, 
                  path1, path2, path11,
                  path22, path111, path222,
                  path1111, path2222
                 )))
    return stacked
dataframe_length = 31206300

class training_batch_generator(k.utils.Sequence):
    # A generator to preprocess the input and 
    # feed the data in batches.
    def __init__(self,
                 tr_iterator,
                 batch_size=50,  
                 data_length=1000, 
                 n_classes=340):
        
        self.iterator = tr_iterator
        self.batch_size = batch_size
        self.data_length = data_length
        self.n_classes = n_classes
        self.chunk= next(self.iterator).sample(frac=1).reset_index(drop=True)
        
    def __len__(self):
        return int(np.floor(self.data_length / self.batch_size))

    def __getitem__(self,index):
        
        labels_in = self.chunk.loc[index*self.batch_size:
                                  (index+1)*self.batch_size-1,1].apply(
                                   replace_name).values
    
        xystrokes = self.chunk.loc[index*self.batch_size:
                                  (index+1)*self.batch_size-1,0].apply(
                                   get_path_signatures).values
        xystrokes = pad_sequences(xystrokes, maxlen= 128, dtype=np.int32)
        
        labels_in = k.utils.to_categorical(labels_in, num_classes= self.n_classes)
        
        return xystrokes, labels_in
    
    def on_epoch_end(self):
        self.chunk= next(self.iterator).sample(frac=1).reset_index(drop=True)

dataframe_length = 31206300

class validation_batch_generator(k.utils.Sequence):
    # A generator to preprocess the input and 
    # feed the data in batches.
    def __init__(self,
                 data_chunk,
                 batch_size=50,  
                 data_length=1000, 
                 n_classes=340):
        
        self.chunk = data_chunk
        self.batch_size = batch_size
        self.data_length = data_length
        self.n_classes = n_classes
        
        
    def __len__(self):
        return int(np.floor(self.data_length / self.batch_size))

    def __getitem__(self,index):
        
        labels_in = self.chunk.loc[index*self.batch_size:
                                  (index+1)*self.batch_size-1,1].apply(
                                   replace_name).values
    
        xystrokes = self.chunk.loc[index*self.batch_size:
                                  (index+1)*self.batch_size-1,0].apply(
                                   get_path_signatures).values
        xystrokes = pad_sequences(xystrokes, maxlen= 128, dtype=np.int32)
        
        labels_in = k.utils.to_categorical(labels_in, num_classes= self.n_classes)
        
        return xystrokes, labels_in

def top3accuracy(y_ture, y_pred):
    return k.metrics.top_k_categorical_accuracy(y_ture, y_pred, k=3)

earlystop = k.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=4, verbose=1)
reduce_lr = k.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.9, patience=1, verbose=1)
def gated_activation(layer_in):
    x1,xg = tf.split(layer_in,2,2)
    xg = Activation('sigmoid')(xg)
    xg = Multiply()([x1 , xg])
    return xg
    

base = 680
def ConvBlock(x, 
              filters=base,
              size=3,
              d_rate=1
              ):
    
    x1 = Lambda(gated_activation)(x)
    x2 = Conv1D(filters, size, padding='same', dilation_rate=2**(d_rate-1))(x1)
    x2a = Lambda(gated_activation)(x2)
    x3 = Conv1D(filters, size, padding='same', dilation_rate=2**(d_rate))(x2a)
    x3a = Add()([x3, x])
    
    return x3a

input1 = Input(shape=(128,10))
norm = BatchNormalization()(input1)
norm_s = Conv1D(base,1,padding='same', use_bias=False)(norm)
res = Conv1D(base,3,padding='same')(norm)
res = Lambda(gated_activation)(res)
res = Conv1D(base,3,padding='same')(res)
res = Add()([norm_s, res])

for i in range(2):
    rate = 2*i + 2
#     rate = [2,4], so we get dilation rate = [2,4,8,16]
    res = ConvBlock(res, base, d_rate=rate)

res = Lambda(gated_activation)(res)
res = Conv1D(base,3,padding='same',dilation_rate=32)(res)
res = Lambda(gated_activation)(res)

pool = GlobalAveragePooling1D()(res)
output = Dense(340, activation='softmax')(pool)

model = Model(inputs=input1, outputs=output)
opt = k.optimizers.Adam(lr=0.001, decay=0.0)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['categorical_accuracy', top3accuracy])
# model.summary()
data = pd.read_csv('training_data_10k.csv', 
                   usecols=[0,1], 
                   header=None,
                   nrows=3000000,
                   dtype={0:str, 1:str}, 
                   iterator=True,
                   chunksize=100000)

val_data = pd.read_csv('training_data_10k.csv', 
                       usecols=[0,1], 
                       header=None,
                       skiprows=3000000,
                       nrows= 10000,
                       dtype={0:str, 1:str}, 
                       )

val_generator = validation_batch_generator(data_chunk= val_data, 
                                batch_size=100, data_length=10000)

t= time.time()

train_generator= training_batch_generator(tr_iterator=data, 
                                batch_size=100, data_length=100000)

r = model.fit_generator(train_generator,
                        validation_data= val_generator,
                        verbose=2,
                        epochs=29,
                        use_multiprocessing=False,
                        callbacks=[earlystop, reduce_lr] 
                        )
tt= time.time()
print('Training time:', (tt-t)/60)
plt.plot(r.history['loss'])
plt.plot(r.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Iterations (x1000)')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
plt.plot(r.history['categorical_accuracy'])
plt.plot(r.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Iterations (x1000)')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
class predict_batch_generator(k.utils.Sequence):
    # A generator to preprocess the input and 
    # feed the data in batches.
    def __init__(self,
                 data_chunk,
                 batch_size=200,  
                 data_length=1000
                ):
        
        self.chunk = data_chunk
        self.batch_size = batch_size
        self.data_length = data_length
        
        
    def __len__(self):
        return int(np.floor(self.data_length / self.batch_size))

    def __getitem__(self,index):
    
        xystrokes = self.chunk.loc[index*self.batch_size:
                                  (index+1)*self.batch_size-1,0].apply(
                                   get_path_signatures).values
        xystrokes = pad_sequences(xystrokes, maxlen= 128, dtype=np.int32)
        
        return xystrokes
test_df = pd.read_csv('training_data_10k.csv', 
                       usecols=[0,1], 
                       header=None,
                       skiprows=3000000,
                       nrows= 10000,
                       dtype={0:str, 1:str}, 
                       )

pred_g = predict_batch_generator(data_chunk=test_df, batch_size=100, data_length=len(test_df))
preds = model.predict_generator(pred_g,verbose=1)
def get_top_3(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=-1)[:, :3], columns=[1,2,3])

top3preds = get_top_3(preds)
get_names = test_df.loc[:,1].apply(replace_name).values
get_names = [[item] for item in get_names]
def apk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
mapk(get_names, top3preds.values, k=3)