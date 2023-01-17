import numpy as np

import pandas as pd

import glob

from random import shuffle

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import tensorflow as tf

import math

import os

from PIL import Image

from tensorflow.keras import regularizers
train_dataset=pd.read_csv('../input/fndfeatures6/trainDataset5.csv',sep='\t')

#train_dataset=train_dataset.drop('Image_path',axis=1)

train_dataset.head()
str1=train_dataset.iloc[0]['properties'][1:-1]

str_list=str1.split(',')

prop=list(map(float,str_list))

prop
test_Dataset=pd.read_csv('../input/fndtestdataset/TestDataset.csv',sep=',')

test_Dataset.head()

test_Features=np.empty((len(test_Dataset),21))

for i in range(len(test_Dataset)):

    str1=test_Dataset.iloc[i]['Features'][1:-1]

    str_list=str1.split(',')

    test_Features[i,:]=(list(map(float,str_list)))

    

test_Features[0]
class DataGenerator1(tf.keras.utils.Sequence):

    def __init__(self,train_data,batch_size=64,shuffle=True):

        self.train_data=train_data

        self.batch_size=batch_size

        self.shuffle=shuffle

        self.on_epoch_end()

        

    def __len__(self):

        return (int)(np.floor(len(self.train_data)/self.batch_size))

    

    def __getitem__(self,index):

        batch_ids=self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        X,y=self.__data_generation(batch_ids)

        return X,y

        

    def on_epoch_end(self):

        self.indexes=np.arange(len(self.train_data))

        if(self.shuffle):

            np.random.shuffle(self.indexes)

            

    def __data_generation(self,batch_ids):

        X=np.empty((len(batch_ids),21))

        y=np.empty((len(batch_ids),1))

        for idx,batch_id in enumerate(batch_ids):

            str1=self.train_data.iloc[batch_id]['properties'][1:-1]

            str_list=str1.split(',')

            prop=list(map(float,str_list))

            X[idx,]=prop

            y[idx,]=self.train_data.iloc[batch_id]['Class']

        return X,y
def DNN_Model():

    inputs=tf.keras.layers.Input((21))

    X=tf.keras.layers.BatchNormalization()(inputs)

    X=tf.keras.layers.Dense(512,activation='relu',kernel_regularizer=regularizers.l2(0.001))(X)

    X=tf.keras.layers.Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.001))(X)

    X=tf.keras.layers.Dropout(0.2)(X)

    X=tf.keras.layers.BatchNormalization()(X)

    X=tf.keras.layers.Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.001))(X)

    X=tf.keras.layers.Dropout(0.2)(X)

    X=tf.keras.layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.001))(X)

    #X=tf.keras.layers.Dropout(0.2)(X)

    X=tf.keras.layers.Dense(32,activation='relu',kernel_regularizer=regularizers.l2(0.001))(X)

   # X=tf.keras.layers.Dropout(0.2)(X)

    X=tf.keras.layers.BatchNormalization()(X)

    #X=tf.keras.layers.Dense(24,activation='relu')(X)

    #X=tf.keras.layers.Dropout(0.2)(X)

    X=tf.keras.layers.Dense(16,activation='relu',kernel_regularizer=regularizers.l2(0.001))(X)

    #X=tf.keras.layers.Dropout(0.2)(X)

    X=tf.keras.layers.Dense(8,activation='relu',kernel_regularizer=regularizers.l2(0.001))(X)   

    #X=tf.keras.layers.Dropout(0.2)(X)

    X=tf.keras.layers.BatchNormalization()(X)

    X=tf.keras.layers.Dense(4,activation='relu',kernel_regularizer=regularizers.l2(0.001))(X)

    #X=tf.keras.layers.Dropout(0.2)(X)

    X=tf.keras.layers.Dense(2,activation='relu',kernel_regularizer=regularizers.l2(0.001))(X)

    #X=tf.keras.layers.Dropout(0.2)(X)

    outputs=tf.keras.layers.Dense(1,activation='sigmoid')(X)

    

    return tf.keras.Model(inputs=inputs,outputs=outputs)
test_data=train_dataset.iloc[0:3000]

train_dataset=train_dataset.iloc[3000:]
test_data.head()

len(train_dataset)
train_data, val_data=train_test_split(train_dataset,test_size=0.2,random_state=42)



train_generator=DataGenerator1(train_data)

val_generator=DataGenerator1(val_data)

#X,y=train_generator.__getitem__(0)



model=DNN_Model()

model.compile(optimizer='SGD',loss=tf.keras.losses.BinaryCrossentropy(),\

              metrics=[tf.keras.metrics.BinaryAccuracy()])



history=model.fit_generator(generator=train_generator,validation_data=val_generator\

                            ,epochs=300,verbose=1)
tf.keras.utils.plot_model(model, show_shapes=True)
epochs=300

acc = history.history['binary_accuracy']

val_acc = history.history['val_binary_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(epochs)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
test_X=[]

test_label=[]

for i in range(3000):

    str1=test_data.iloc[i]['properties'][1:-1]

    str_list=str1.split(',')

    prop=list(map(float,str_list))

    test_X.append(prop)

    test_label.append(test_data.iloc[i]['Class'])

    

test_X=np.array(test_X)

test_label=np.array(test_label)
y_pred=model.predict(test_X)

pred_y=[]

for i in range(len(y_pred)):

    if(y_pred[i]>=0.5):

        pred_y.append(1)

    else:

        pred_y.append(0)

        

pred_y=np.array(pred_y)
from sklearn import metrics

metrics.accuracy_score(test_label,pred_y)
metrics.f1_score(test_label,pred_y)
test_y=model.predict(test_Features)

y_test=[]

for i in range(len(test_y)):

    if(test_y[i]>=0.5):

        y_test.append(1)

    else:

        y_test.append(0)
submission=pd.DataFrame(

    {'ImageId':test_Dataset['Iamge_Id'],

    'label':y_test})

submission.head()
submission.to_csv('./submission.csv',index=False)

model.save('./MLP10.h5')