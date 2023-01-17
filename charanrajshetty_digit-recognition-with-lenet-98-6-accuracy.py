import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

TrainSet = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
X_train = TrainSet.iloc[:,1:].values
Y_train = TrainSet.iloc[:,0].values
TestSet = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
X_test = TestSet.iloc[:,:].values
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
X_train
X_train.shape
X_test.shape
rows,columns = 28,28
X_train = X_train.reshape(X_train.shape[0],rows,columns,1)
X_test  = X_test.reshape(X_test.shape[0],rows,columns,1)
from sklearn.model_selection import train_test_split
X_train1,X_check1,Y_train1,Y_check1 = train_test_split(X_train,Y_train,test_size=0.2,random_state=1)
X_train1
print(X_train1.shape)
print(X_check1.shape)
input_shape = [rows,columns,1]
Y_train1  = tf.keras.utils.to_categorical(Y_train1,10)
Y_check1  = tf.keras.utils.to_categorical(Y_check1,10)
def construct_lenet(input_shape):
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(
                                    filters= 6,
                                    kernel_size = (5,5),
                                    strides=(1,1),
                                    activation='tanh',
                                    input_shape=input_shape))
    
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2),
                                            strides=(2,2)))
    
    model.add(tf.keras.layers.Conv2D(
                                    filters=16,
                                    kernel_size = (5,5),
                                    strides=(1,1),
                                    activation='tanh'))
    
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2),
                                            strides=(2,2)))
    
    model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dense(units=120,activation='tanh'))
    
    model.add(tf.keras.layers.Dense(units=84,activation='tanh'))
    
    model.add(tf.keras.layers.Dense(units=10,activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.SGD(lr=0.1,momentum=0.0,decay=0.0),metrics=['accuracy'])
    
    return model
lenet = construct_lenet(input_shape)

epochs=50

history = lenet.fit(X_train1,Y_train1,
                    epochs=epochs,
                   batch_size=128)


loss , acc = lenet.evaluate(X_check1,Y_check1)
print('Accuracy :',acc)

results = lenet.predict(X_test)
results = np.argmax(results,axis = 1)
print(results)
results = pd.Series(results,name="Label")
print(results)
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("Lenet_results.csv",index=False)