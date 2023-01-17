import matplotlib.pyplot as plt

import pandas as pd

from keras.utils import np_utils

from keras.initializers import he_normal

import seaborn as sns

from keras.models import Sequential

from keras.layers import Dense , Activation

from keras import optimizers

from keras import losses

import numpy as np

import pandas as pd

import time

from keras.layers.normalization import BatchNormalization

from keras.layers import Dropout

from keras.layers.merge import concatenate

from keras.utils import plot_model

from keras.layers import Input

from keras.models import Model



nepoch = 30

outlayer = 2

batch_size = 1000

X_test = pd.read_csv(r'/kaggle/input/xor-puf64128/XOR_Arbiter_PUFs/5xor_128bit/test_5xor_128dim.csv',header = None)





X_test.head()

Y_test = X_test[[128]]

X_test.drop(128,axis = 1,inplace = True)

y_test = np_utils.to_categorical(Y_test, 2)

#distrb = X_train.iloc[:,64].value_counts()



input_layer = Input(shape = (128,))

model = Sequential()

model.add(Dense(100,input_dim=128,activation = 'sigmoid'))

model.add(Dense(64,input_dim=128,activation = 'sigmoid'))

model.add(Dense(20,input_dim=128,activation = 'sigmoid'))



model.add(Dense(2,activation = 'softmax'))



# summarize layers

print(model.summary())



# plot graph

plot_model(model, to_file='MODEL.png')



adam = optimizers.Adam(lr = 0.001)

model.compile(loss=losses.categorical_crossentropy, optimizer = adam, metrics=['accuracy'])

start = time.clock()

chunksize = 1000000

i=0

for X_train in pd.read_csv(r'/kaggle/input/xor-puf64128/XOR_Arbiter_PUFs/5xor_128bit/train_5xor_128dim.csv',header = None, chunksize=chunksize):

    print("done")

    print(i)

    i=i+1





    Y_train = X_train[[128]]

    X_train.drop([128],axis = 1,inplace = True)





    y_train = np_utils.to_categorical(Y_train, 2) 



    print(X_train.describe())



    print(X_train.shape)

















    hist = model.fit(X_train, y_train, epochs=nepoch, batch_size=batch_size,validation_data = (X_test,y_test))





# Final evaluation of the model

scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))

print(time.clock() - start)







        


