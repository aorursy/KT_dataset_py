# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Dropout,Activation,Conv2D,Flatten,MaxPool2D, GlobalAveragePooling2D, GlobalMaxPooling2D

from keras.utils import np_utils

from sklearn.metrics import confusion_matrix

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score, StratifiedKFold

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.resnet_v2 import ResNet50V2

from tensorflow.keras import optimizers

from tensorflow.keras.callbacks import EarlyStopping





import tensorflow as tf

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



import os

#run plaidml on Keras

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"



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



X_train2 = X_train2.reshape(-1,28,28)



X_train2 = np.repeat(X_train2[..., np.newaxis], 3, -1)



X_test = X_test.reshape(-1,28,28)



X_test = np.repeat(X_test[..., np.newaxis], 3, -1)



y_train2 = y_train2.astype('int32')

y_t = y_train2



y_train3 = np_utils.to_categorical(y_train2)



y_train2 = np.repeat(y_train3[..., np.newaxis], 3, -1)

y_train2 = y_train2.reshape(-1,2,3,10)



fatia = X_train2[2]

fatia = fatia.reshape(1,28,28,3)



y_fatia = y_train2[2]

y_fatia = y_fatia.reshape(1,2,3,10)



modelo = ResNet50V2(include_top = False,weights = None)

x = modelo.output

x = GlobalMaxPooling2D()(x)

x = Dense(256, activation='relu', name='Dense_Intermediate')(x)

x = Dropout(0.3)(x) #add new layer

x = Dense(256, activation='relu', name='Dense_Intermediate2')(x)





predictions = Dense(units = 10, activation= 'softmax')(x)

modelo = Model(inputs = modelo.input, outputs = predictions)



opt = optimizers.Adam()









modelo.summary()



modelo.compile(optimizer = opt, loss = 'categorical_crossentropy',metrics=['accuracy'])



es = EarlyStopping(patience = 10,restore_best_weights = True)





#neural_network = KerasClassifier(build_fn = modelo, epochs = 20, batch_size = 100, verbose = 0)

#scores = cross_val_score(modelo,data,y,cv= 5,scoring = 'accuracy')



history = modelo.fit(X_train2,y_t, epochs=100,batch_size = 100, validation_split = 0.2,callbacks = [es])





preds = modelo.predict(X_test, verbose=0)

predict_class = np.argmax(preds, axis=1)





def write_preds(preds, fname):

    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)



write_preds(predict_class, "keras-mlp.csv")



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()




