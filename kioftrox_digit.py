# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers import Input, Dropout, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
NB_EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 2
NB_CLASSES = 10 #numero de salida = numero de digitos
OPTIMIZER = Adam() #Optimizador Adam
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2 #cuanto entrenamiento esta reservado para validación
np.random.seed(1671)
def get_data(path_train='../input/train.csv',path_test='../input/test.csv'):
    file = open(path_train)

    Data_train = pd.read_csv(file)
    y_train = np.array(Data_train.iloc[:,0])
    x_train = np.array(Data_train.iloc[:,1:])

    file.close()

    file = open(path_test)

    Data_test = pd.read_csv(file)
    y_test = np.array(Data_test.iloc[:,0])
    x_test = np.array(Data_test.iloc[:,0:])

    file.close()

    return(y_train,x_train,y_test,x_test)
Y_train,X_train,Y_test,X_test = get_data()
X_train = X_train.reshape(X_train.shape[0],28,28,1)
X_test = X_test.reshape(X_test.shape[0],28,28,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train,NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test,NB_CLASSES)
n1 = Input(shape=(28,28,1))

conv_lay_1 = Conv2D(40, kernel_size=6, padding='same', input_shape=(28,28,1),activation='relu')(n1)
drop_lay_1 = Dropout(0.3)(conv_lay_1)
conv_lay_2 = Conv2D(50,kernel_size=5,padding='valid',activation='relu')(drop_lay_1)
drop_lay_2 = Dropout(0.3)(conv_lay_2)
pool_lay_1 = MaxPooling2D(pool_size=(3,3), strides =(2,2))(drop_lay_2)
drop_lay_3 = Dropout(0.3)(pool_lay_1)
conv_lay_3 = Conv2D(70, kernel_size=4, padding="same", activation = 'relu')(drop_lay_3)
pool_lay_2 = MaxPooling2D(pool_size=(3,3), strides =(2,2))(drop_lay_3)
conv_lay_4 = Conv2D(100, kernel_size=3, padding="valid", activation = 'relu')(pool_lay_2)
pool_lay_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv_lay_4)
flate_lay  = Flatten()(pool_lay_3)
na = Dense(100, activation='relu')(flate_lay)
nb = Dense(50, activation='relu')(na)
nc = Dense(25,activation='relu')(nb)
print(NB_CLASSES)
out = Dense(NB_CLASSES,activation='softmax')(nc)


model = Model(inputs=[n1], outputs=out)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
y_proba = model.predict(X_test)
Y = []
for element in y_proba:
    Y.append(np.argmax(element))
data_predict = {"ImageId":range(1, X_test.shape[0]+1), "Label":Y}
data_predict = pd.DataFrame(data_predict)
data_predict.to_csv("output.csv", index = False)
print(os.listdir('../working/'))
