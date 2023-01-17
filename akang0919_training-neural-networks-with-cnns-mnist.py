# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import random

import pandas as pd

import numpy as np

import datetime

from datetime import datetime

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.utils.np_utils import to_categorical

from keras.utils import np_utils





from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV

from sklearn.metrics import confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
RANDOM_SEED = 1

SET_FIT_INTERCEPT = True



os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

random.seed(RANDOM_SEED)



np.random.seed(RANDOM_SEED)
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train_data = np.array(train)

test_data = np.array(test)



# drop the label (target) of train dataset

X_train_data = (train.values[:,1:]).astype(np.float32)



# get the label (target) of train dataset 

y_train_data = (train.values[:,0]).astype(np.int32)



# test dataset does not contain a label

X_test_data = (test.values).astype(np.float32) 



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_data, y_train_data, test_size = .2, random_state=RANDOM_SEED)
X_train = X_train.reshape(-1, 28, 28, 1)

X_test = X_test.reshape(-1, 28, 28, 1)

print(X_train.shape, y_train.shape)
X_train = X_train.astype('float32')/255

X_test = X_test.astype('float32')/255 
print('Integer-valued labels:')

print(y_train[:10])



# one-hot encode the labels

y_train = np_utils.to_categorical(y_train, 10)

y_test = np_utils.to_categorical(y_test, 10)



# print first ten (one-hot) training labels

print('One-hot labels:')

print(y_train[:10])
# examine the shape of the loaded data

print('train_data shape:', X_train.shape)

print('test_data shape:', X_test.shape, '\n')
# X_train = tf.keras.utils.normalize(X_train, axis = 1)

# X_test = tf.keras.utils.normalize(X_test, axis = 1)
test_data = test_data.reshape(-1,28,28,1)
print('response count\n', train.label.value_counts())
train_plot = train.drop('label',axis=1)
plt.figure(figsize=(14,12))

for digit_num in range(0,70):

    plt.subplot(7,10,digit_num+1)

    grid_data = train_plot.iloc[digit_num].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array

    plt.imshow(grid_data, interpolation = "none", cmap = "afmhot")

    plt.xticks([])

    plt.yticks([])

plt.tight_layout()
optimizers = ['Adamax']

num_nodes = [256]
from sklearn.model_selection import ParameterGrid

parameters = {

'optimizer': optimizers,

'num_nodes' : num_nodes

}
parameterList = list(ParameterGrid(parameters))
num_nodes_list = []

optimizer_list = []

train_acc_list = []

test_acc_list = []

proc_list = []
from keras.callbacks import ModelCheckpoint   

from keras.callbacks import ReduceLROnPlateau



# train the model

#checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', 

#                               verbose=1, save_best_only=True)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, verbose=1,

                              patience=2, min_lr=0.00000001)
optim = parameterList[0]['optimizer']

numNodes = parameterList[0]['num_nodes']

print("Optimizer: ", optim)

print("Number of Nodes: ", numNodes)



model1 = Sequential([

    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu',input_shape=(28,28,1)),

    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),

    Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),

    MaxPooling2D(pool_size=2),

    Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),

    Conv2D(filters=192, kernel_size=3, padding='same', activation='relu'),

    MaxPooling2D(pool_size=2),

    Conv2D(filters=192, kernel_size=5, padding='same', activation='relu'),

    MaxPooling2D(pool_size=2, padding='same'),

    Flatten(),

    Dense(numNodes, activation='relu'),

    Dense(10, activation='softmax'),

    ])



model1.compile(

    optimizer=optim,

    loss='categorical_crossentropy',

    metrics=['accuracy'],

    )



start = datetime.now()

model1.fit(X_train, y_train, batch_size=100, epochs=50,validation_data=(X_test, y_test), callbacks=[reduce_lr],verbose=1, shuffle=True)

end = datetime.now()

processing_time = end-start



model1.summary()



score_train,acc_train = model1.evaluate(X_train,y_train)

score_test,acc_test = model1.evaluate(X_test,y_test)



print("Training Accuracy: ", acc_train)

print("Test Accuracy: ", acc_test)

print("Processing Time: ", processing_time)

num_nodes_list.append(numNodes)

optimizer_list.append(optim)

train_acc_list.append(acc_train)

test_acc_list.append(acc_test)

proc_list.append(processing_time)
performance_df = pd.DataFrame(columns = ["Combination", 

                                         "Optimizer",

                                         "Number of Nodes",

                                         "Processing Time", 

                                         "Train Accuracy", 

                                         "Test Accuracy"])
performance_df['Combination'] = ParameterGrid(parameters)

performance_df['Optimizer'] = optimizer_list

performance_df['Number of Nodes'] = num_nodes_list

performance_df['Processing Time'] = proc_list

performance_df['Train Accuracy'] = train_acc_list

performance_df['Test Accuracy'] =  test_acc_list
performance_df = performance_df.sort_values(by='Test Accuracy',ascending=False)
performance_df
results = model1.predict_classes(test_data,verbose=2)
results
results = pd.Series(results,name="Label")
results.head(5)
np.savetxt('results.csv',

           np.c_[range(1,len(test)+1),results],

           delimiter=',',

           header='ImageId,Label',

           comments= '',

           fmt = '%d'

          )