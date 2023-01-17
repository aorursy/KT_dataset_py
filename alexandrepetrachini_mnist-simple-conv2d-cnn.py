# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout,Activation,Conv2D,Flatten,MaxPool2D, GlobalAveragePooling2D

from keras.utils import np_utils

from sklearn.metrics import confusion_matrix

from keras.datasets import mnist

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score, StratifiedKFold

from keras.applications.resnet50 import ResNet50







        

data_train = pd.read_csv('../input/digit-recognizer/train.csv')

data_test = pd.read_csv('../input/digit-recognizer/test.csv')



data_train_w = data_train.drop(['label'],axis = 1)

data = data_train_w.append(data_test,ignore_index = True)







y_train = data_train.iloc[:,0]

print(y_train.shape)

X_train = (data_train.iloc[:,1:].values).astype('float32')

print(X_train.shape)







X_test = (data_test.iloc[:,0:].values).astype('float32')









# pre-processing: divide by max and substract mean

scale = np.max(X_train)

scale

X_train /= scale

X_test /= scale



mean = np.std(X_train)

X_train -= mean

X_test -= mean



y_train = np_utils.to_categorical(y_train,10)







y_train =(np.array(y_train))







X_train2,X_test2,y_train2,y_test2 = train_test_split(X_train,y_train)





val = round((784+10) / 2)



X_train2 = X_train2.reshape(-1,28,28,1)







modelo = Sequential()



modelo.add(Conv2D(32,(3,3),input_shape = (28,28,1)))

modelo.add(Activation('relu'))

modelo.add(MaxPool2D(pool_size =(2,2)))

modelo.add(Conv2D(32,(3,3),input_shape = (28,28,1)))

modelo.add(Activation('relu'))

modelo.add(MaxPool2D(pool_size =(2,2)))



modelo.add(Conv2D(32,(3,3),input_shape = (28,28,1)))

modelo.add(Activation('relu'))

modelo.add(MaxPool2D(pool_size =(2,2)))



modelo.add(Flatten())

modelo.add(Dense(10))

modelo.add(Activation('softmax'))





modelo.summary()



modelo.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',metrics=['accuracy'])



#neural_network = KerasClassifier(build_fn = modelo, epochs = 20, batch_size = 100, verbose = 0)

#scores = cross_val_score(modelo,data,y,cv= 5,scoring = 'accuracy')



modelo.fit(X_train2, y_train2, epochs=20, batch_size=16, validation_split=0.1, verbose=2)





preds = modelo.predict_classes(X_test.reshape(-1,28,28,1), verbose=0)



def write_preds(preds, fname):

    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)



write_preds(preds, "keras-mlp.csv")


