import numpy as np

import matplotlib.image as mpimg 

import matplotlib.pyplot as plt 

import os,sys

import tensorflow as tf

from tensorflow.keras.utils import Sequence

from tensorflow.python.tpu import datasets

from tensorflow.python.data.ops import dataset_ops
from PIL import Image, ImageFilter, ImageEnhance
np.random.seed(16)

tf.random.set_seed(16)
import time 

# code found in https://www.kaggle.com/danmoller/make-best-use-of-a-kernel-s-limited-uptime-keras

#let's also import the abstract base class for our callback

from keras.callbacks import Callback



#defining the callback

class TimerCallback(Callback):

    

    def __init__(self, maxExecutionTime, byBatch = False, on_interrupt=None):

        

# Arguments:

#     maxExecutionTime (number): Time in minutes. The model will keep training 

#                                until shortly before this limit

#                                (If you need safety, provide a time with a certain tolerance)



#     byBatch (boolean)     : If True, will try to interrupt training at the end of each batch

#                             If False, will try to interrupt the model at the end of each epoch    

#                            (use `byBatch = True` only if each epoch is going to take hours)          



#     on_interrupt (method)          : called when training is interrupted

#         signature: func(model,elapsedTime), where...

#               model: the model being trained

#               elapsedTime: the time passed since the beginning until interruption   



        

        self.maxExecutionTime = maxExecutionTime * 60

        self.on_interrupt = on_interrupt

        

        #the same handler is used for checking each batch or each epoch

        if byBatch == True:

            #on_batch_end is called by keras every time a batch finishes

            self.on_batch_end = self.on_end_handler

        else:

            #on_epoch_end is called by keras every time an epoch finishes

            self.on_epoch_end = self.on_end_handler

    

    

    #Keras will call this when training begins

    def on_train_begin(self, logs):

        self.startTime = time.time()

        self.longestTime = 0            #time taken by the longest epoch or batch

        self.lastTime = self.startTime  #time when the last trained epoch or batch was finished

    

    

    #this is our custom handler that will be used in place of the keras methods:

        #`on_batch_end(batch,logs)` or `on_epoch_end(epoch,logs)`

    def on_end_handler(self, index, logs):

        

        currentTime      = time.time()                           

        self.elapsedTime = currentTime - self.startTime    #total time taken until now

        thisTime         = currentTime - self.lastTime     #time taken for the current epoch

                                                               #or batch to finish

        

        self.lastTime = currentTime

        

        #verifications will be made based on the longest epoch or batch

        if thisTime > self.longestTime:

            self.longestTime = thisTime

        

        

        #if the (assumed) time taken by the next epoch or batch is greater than the

            #remaining time, stop training

        remainingTime = self.maxExecutionTime - self.elapsedTime

        if remainingTime < self.longestTime:

            

            self.model.stop_training = True  #this tells Keras to not continue training

            print("\n\nTimerCallback: Finishing model training before it takes too much time. (Elapsed time: " + str(self.elapsedTime/60.) + " minutes )\n\n")

            

            #if we have passed the `on_interrupt` callback, call it here

            if self.on_interrupt is not None:

                self.on_interrupt(self.model, self.elapsedTime)

normal = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/')

diseased = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/')

normal = ['/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/'+i for i in normal]

diseased = ['/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/'+i for i in diseased]

file_list_train = np.concatenate([normal,diseased])
normal = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/')

diseased = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/')

normal = ['/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/'+i for i in normal]

diseased = ['/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/'+i for i in diseased]

file_list_test = np.concatenate([normal,diseased])
from sklearn.preprocessing import scale

from multiprocessing import Pool

from functools import partial

import gc



class Data_generator(Sequence):



    def __init__(self,file_list,batch_size,shuffle=False):

        self.files = file_list

        if shuffle:

            np.random.shuffle(file_list)

        self.batch_size = batch_size

        self.files_split = np.array_split(self.files,np.ceil(len(self.files)/self.batch_size))

        self.shuffle = shuffle

        self.on_epoch_end()



    def __len__(self):

        length = len(self.files_split)

        return length

    

    def __getitem__(self,index):

        list_IDs_temp = self.files_split[index]



        X, Y = self.__data_generation(list_IDs_temp)

        

        return X,Y





    def on_epoch_end(self):

        self.indexes = np.arange(len(self.files_split))



    def __data_generation(self,self_IDs_temp):

        image1 = []

        image2 = []

        image3 = []

        image4 = []

        labels = []

            

        for i in self_IDs_temp:

            im = Image.open(i)

            enh = ImageEnhance.Contrast(im)

            enh = enh.enhance(1.8)

            enh = enh.resize((1000,800))

            enh = enh.convert(mode='L')

            

            for k in range(2):

                for j in range(2):

                    left = k*1000/2

                    top = j*800/2

                    right = (k+1) * 1000/2

                    bottom = (j+1)*800/2

                    new_image = enh.crop((left,top,right,bottom))

                    if (k==0)&(j==0):

                        image1.append(np.vstack(np.array_split(scale(list(new_image.getdata())),400)).reshape(500,400))

                    elif (k==0)&(j==1):

                        image2.append(np.vstack(np.array_split(scale(list(new_image.getdata())),400)).reshape(500,400))

                    elif (k==1)&(j==0):

                        image3.append(np.vstack(np.array_split(scale(list(new_image.getdata())),400)).reshape(500,400))

                    else:

                        image4.append(np.vstack(np.array_split(scale(list(new_image.getdata())),400)).reshape(500,400))

            

            if ('virus' in i) or ('bacteria' in i):

                labels.append(0)

            else:

                labels.append(1)

            

                

        image1 = np.array(image1)

        image2 = np.array(image2)

        image3 = np.array(image3)

        image4 = np.array(image4)

        labels = np.array(labels)

        gc.collect()

        

        return [image1,image2,image3,image4],labels

import tensorflow.keras

from tensorflow.keras import initializers

from tensorflow.keras.models import Sequential,load_model, Model

from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, concatenate, Concatenate, Lambda, ELU,Activation, ZeroPadding2D

from tensorflow.keras.layers import SeparableConv2D, MaxPooling2D, GaussianNoise, Conv2D, GlobalAveragePooling2D

from tensorflow.keras.layers import LeakyReLU, SimpleRNN, GRU, LSTM, Reshape

from tensorflow.keras import regularizers

input1 = Input(shape=(500,400))

input2 = Input(shape=(500,400))

input3 = Input(shape=(500,400))

input4 = Input(shape=(500,400))



filters = 64





CNN1 = LSTM(filters)(input1)

CNN2 = LSTM(filters)(input2)

CNN3 = LSTM(filters)(input3)

CNN4 = LSTM(filters)(input4)



model = Concatenate(axis=-1)([CNN1,CNN2,CNN3,CNN4])



model = Dense(32)(model)

model = LeakyReLU()(model)



predictions = Dense(1, activation='sigmoid',kernel_regularizer = regularizers.l1(0.1))(model)



model = Model(inputs = [input1,input2,input3,input4], outputs=[predictions])

        

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',tensorflow.keras.metrics.AUC()])
best_model = tensorflow.keras.callbacks.ModelCheckpoint('output_model.h5',

                                             monitor='val_accuracy',

                                             save_best_only=True,

                                             save_weights_only=False,

                                             mode='max')



history = model.fit_generator(Data_generator(file_list=file_list_train,batch_size = 5,shuffle=True),

                              epochs = 5,

                              validation_data=Data_generator(file_list_test,4),

                              callbacks = [best_model,TimerCallback(330)],

                              use_multiprocessing = True,

                              workers = 5)
from keras.models import save_model

save_model(model,filepath='model.h5')
normal = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL/')

diseased = os.listdir('/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/')

normal = ['/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL/'+i for i in normal]

diseased = ['/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA/'+i for i in diseased]

file_list_val = np.concatenate([normal,diseased])
model1 = load_model('/kaggle/working/output_model.h5',compile=False)

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',tensorflow.keras.metrics.AUC()])
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import StratifiedKFold
recall0 = []

recall1 = []

cutoff = np.arange(0.1,0.51,0.01)



values = []

for i in file_list_test:

    if ('bacteria' in i) or ('virus' in i):

        values.append(0)

    else:

        values.append(1)

        

skf = StratifiedKFold(n_splits=5)



for cut in cutoff:

    print(cut)

    r1 = 0

    r2 = 0

    for train_index,test_index in skf.split(file_list_test,values):

        pred = model1.predict_generator(Data_generator(file_list_test[test_index],5))

        values_pred = []

   

        for i in pred:

            if i < cut:

                values_pred.append(0)

            else:

                values_pred.append(1)

        cm = confusion_matrix(np.array(values)[test_index],values_pred)

        tp,fn,fp,tn = cm.flatten()

        r1 += tp/(tp+fn)

        r2 += tn/(tn+fp)

    recall0.append(r1/5)

    recall1.append(r2/5)
difference = [abs(i-j) for i,j in zip(recall0,recall1)]

cutoff_value = cutoff[np.argmin(difference)]
plt.figure()

plt.plot(cutoff,recall0,label='Pneumonia')

plt.plot(cutoff,recall1,label='Normal')

plt.xlabel('Cutoff')

plt.ylabel('Recall')

plt.legend()

plt.savefig('recall_plot')
pred = model1.predict_generator(Data_generator(file_list_test,5))



values = []

for i in file_list_test:

    if ('bacteria' in i) or ('virus' in i):

        values.append(0)

    else:

        values.append(1)

values_pred = []

for i in pred:

    if i < cutoff_value:

        values_pred.append(0)

    else:

        values_pred.append(1)

        
tp,fn,fp,tn = confusion_matrix(y_true=values,y_pred=values_pred).flatten()

sensitivity = tp/(tp+fn)

specificity = tn/(tn+fp)

print('tp:',tp)

print('tn:',tn)

print('fp:',fp)

print('fn:',fn)

print('Sensitivity: ', sensitivity)

print('Specificity: ', specificity)

print('Accuracy: ', (tp+tn)/len(values))