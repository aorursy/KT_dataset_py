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
import pandas as pd



df = pd.read_csv('../input/train-full/Train_full.csv', sep=',')

df.head()
df.info()
# Data preparation

import numpy as np

from keras.utils import to_categorical



data = df.iloc[:,1:-1].values   # remove the first and last column, then convert data frame to numpy array

y_train = to_categorical(df['up_down']) # change the last column to one-hot-encoding for training

x_train = data

#y_train = df['output'][:-1]



from sklearn import preprocessing  # feature scaling



x_train_scaled = preprocessing.scale(x_train)   # scaled data now has zero mean and unit variance

#y_train_scaled = preprocessing.scale(y_train) 
# Try out the combination of different value of hyper-paramameters learning rate and mini-batch size

# Do not use the grid search here, instead I sample the two hyper-parameters at random in an appropriate scale

from keras.optimizers import Adam



r = -4*np.random.rand()   

lr = 10**r                                      # Choose learning rate in the log scale in [0.0001, 1]

list_size = [32, 64, 128, 256, 512]             # Choose batch_size in list

mini_batch_size = np.random.choice(list_size)

print(lr)

print(mini_batch_size)
# Building the model

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras import metrics

from keras.callbacks import EarlyStopping

from keras.regularizers import l2

from keras.layers.normalization import BatchNormalization 



np.random.seed(9)

n_cols = x_train.shape[1] # no of features

model = Sequential()

model.add(Dense(100, input_shape=(n_cols,))) # there's n_cols nodes in the input layer, 100 nodes in the 

                                             # first hidden layer

model.add(BatchNormalization())                    # Batch Normalization for the first hidden layer

model.add(Activation('relu'))                      # use the activation function 'relu' for the nodes

model.add(Dense(80))            # 80 nodes for the second hidden layer

model.add(BatchNormalization())                    # Batch Normalization for the second hidden layer

model.add(Activation('relu'))

model.add(Dense(2, activation='softmax'))



                                  # The output layer has two nodes which are (1 0) or (0 1), 



# activation function here is 'softmax'

#activity_regularizer=l2(0.0001)



my_optimizer = Adam(lr=lr)

early_stopping_monitor = EarlyStopping(patience=3)

# compile the model

model.compile(optimizer=my_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# train the model

history = model.fit(x_train_scaled, y_train, validation_split=0.2, batch_size=mini_batch_size, epochs=20, callbacks=[early_stopping_monitor])
import matplotlib.pyplot as plt



plt.plot(history.history['loss'], label='train')     # plot the accuracy of train and validation data

plt.plot(history.history['val_loss'], label='val')

plt.show()
# Read the test data

df1 = pd.read_csv('../input/test-small/Test_small_features.csv', sep=',')

print(df1.shape)

df1.head()
#df1_re = df1[(df1['Open']==df1['High']) & (df1['Open']==df1['Low']) & (df1['Open']==df1['Close']) & (df1['Volume']==0)]

#print(df1_re.shape)

#df1 = df1[(df1['Open']!=df1['High']) | (df1['Open']!=df1['Low']) | (df1['Open']!=df1['Close']) | (df1['Volume']!=0)]

#print(df1.shape)
# data preparation for test set

data1 = df1.iloc[:,1:-1].values   

y_test = df1['up_down'].values

x_test = data1



x_test_scaled = preprocessing.scale(x_test) 
# Prediction

pred = model.predict(x_test_scaled) # pred has two columns. The first column is the prob of down, and the second one 

                                    # is the prob of up

pred1 = pred[:,1]

list1 = []

count = 0

for element in pred1:

    if element >= 0.5:

        temp = 1

    else:

        temp = 0

    list1.append(temp)

for i,j in zip(list1, y_test):

    if i == j:

        count += 1

count/len(y_test)
pred2 = model.predict_classes(x_test_scaled)
pd.DataFrame(pred2).to_csv('result.csv', sep=',', header=['label'], index_label='num')
#count1 = 0

#for i in df1_re['up_down']:

 #   if i == 0:

  #      count1 += 1

#print(count1)

#(count+count1)/(len(y_test)+count1)