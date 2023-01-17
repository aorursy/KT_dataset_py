
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import math
print(tf.__version__)
import os,datetime

%load_ext tensorboard
%tensorboard --logdir {logs_base_dir}

dataset = pd.read_csv("/kaggle/input/nasa-asteroids-classification/nasa.csv")
data_x  = dataset.iloc[:,:-2]
data_y = dataset.iloc[:,-1].values
data_x.drop(data_x.columns[[0,1,11,12,20,21,22]],axis=1,inplace=True)
data_x = data_x.apply(pd.to_numeric)
data_x = data_x.values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size = 0.19,random_state=0)
print("x_train"+str(x_train.shape))
print("x_test"+str(x_test.shape))
print("y_train"+str(y_train.shape))
print("y_test"+str(y_test.shape))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
y_train = np.reshape(y_train,(y_train.shape[0],1))
y_test  = np.reshape(y_test,(y_test.shape[0],1))
y_train = y_train*1
y_test  = y_test*1
x_train=np.float32(x_train)
x_test = np.float32(x_test)
print("x_train"+str(x_train.shape))
print("x_test"+str(x_test.shape))
print("y_train"+str(y_train.shape))
print("y_test"+str(y_test.shape))
n_x = x_train.shape[1]
print("x_train"+str(x_train.shape))
print("x_test"+str(x_test.shape))
print("y_train"+str(y_train.shape))
print("y_test"+str(y_test.shape))
print(n_x)
model  = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10,activation = 'relu', kernel_initializer = 'glorot_normal' , bias_initializer = 'glorot_normal',input_shape = (n_x,)))
model.add(tf.keras.layers.Dense(5, activation='relu', kernel_initializer='glorot_normal',bias_initializer = 'glorot_normal'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid',kernel_initializer = 'glorot_normal',bias_initializer = 'glorot_normal'))


model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
training_history = model.fit(x_train,y_train,epochs=35,batch_size = 16,verbose=1,use_multiprocessing=True,callbacks=[tensorboard_callback])

losstrain,acctrain = model.evaluate(x_train,y_train,verbose=0)
loss, acc = model.evaluate(x_test, y_test, verbose=0)



print('Train Accuracy: %.3f' % acctrain)
print('Test Accuracy: %.3f' % acc)
print("Average test loss: ", np.average(training_history.history['loss']))

%tensorboard --logdir logs

