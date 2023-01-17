import pandas as pd
import numpy as np
import os
import tensorflow as tf
import datetime
from sklearn.model_selection import train_test_split
import h5py
import matplotlib.pyplot as plt
def guiyihua(data):
    import numpy as np
    data=((data-np.min(data))/(np.max(data)-np.min(data)))*2-1
    return data
xi_1=np.load('../input/data-class-7/l_1.npy')
xi_2=np.load('../input/data-class-7/l_2.npy')
xi_3=np.load('../input/data-class-7/l_3.npy')
xi_4=np.load('../input/data-class-7/l_4.npy')
xi_5=np.load('../input/data-class-7/l_5.npy')
xi_6=np.load('../input/data-class-7/l_6.npy')
xi_7=np.load('../input/data-class-7/l_7.npy')
from sklearn import preprocessing
label=[]
for i in range(7):
    for t in range(500):
        label.append(i)

xi_data=np.concatenate((xi_1,xi_2,xi_3,xi_4,xi_5,xi_6,xi_7),axis=0)
xi_data.shape
train_label_onehot=tf.keras.utils.to_categorical(label)
train_data=guiyihua(xi_data)
train_data=train_data.reshape(3500,4800,1)
x_train,x_test,y_train,y_test=train_test_split(train_data,train_label_onehot,test_size=0.15)
x_train.shape,x_test.shape
zip_train=tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x_train),tf.data.Dataset.from_tensor_slices(y_train)))
zip_test=tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(x_test),tf.data.Dataset.from_tensor_slices(y_test)))
zip_train
zip_test
BATCH_SIZE=128
batch_zip_train=zip_train.shuffle(2975).repeat().batch(BATCH_SIZE)
batch_zip_val=zip_test.batch(BATCH_SIZE)
model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv1D(64,3,padding='same',activation='relu', input_shape=(4800,1)))
model.add(tf.keras.layers.MaxPool1D())
model.add(tf.keras.layers.Conv1D(64,3,padding='same',activation='relu'))
model.add(tf.keras.layers.MaxPool1D())
model.add(tf.keras.layers.Conv1D(32,3,padding='same',activation='relu'))
model.add(tf.keras.layers.MaxPool1D())
model.add(tf.keras.layers.Conv1D(16,3,padding='same',activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100,activation='relu'))
model.add(tf.keras.layers.Dense(7,activation='softmax'))
print(model.summary())
tf.keras.utils.plot_model(model)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'] )
log_dir=os.path.join('logs','清洗数据'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir,histogram_freq=2)
history=model.fit(batch_zip_train,
             epochs=1000,
              steps_per_epoch=2975//128,
              validation_data=batch_zip_val,
             callbacks=[tensorboard_callback]) 
model.save('the_save_model.h5')
