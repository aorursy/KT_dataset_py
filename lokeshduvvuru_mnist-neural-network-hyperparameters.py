# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
print(train.shape)

print(test.shape)
train_data = np.array(train,dtype='float32')

test_data = np.array(test,dtype='float32')
X_train = train_data[:,1:]/255

y_train = train_data[:,0]

test = test_data[:,0:]/255
print(X_train.shape)

print(y_train.shape)
print(X_train.max())

print(X_train.min())
import matplotlib.pyplot as plt

plt.imshow(X_train[10].reshape(28,28))
import tensorflow
print(y_train[10])

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=10)

print(y_train[10])
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras import regularizers, optimizers
def train_and_test_loop(iterations, lr, Lambda, verb=True):



    ## hyperparameters

    iterations = iterations

    learning_rate = lr

    hidden_nodes = 256

    output_nodes = 10

        

    model = Sequential()

    model.add(Dense(hidden_nodes, input_shape=(784,), activation='relu'))

    model.add(Dense(hidden_nodes, activation='relu'))

    model.add(Dense(output_nodes, activation='softmax', kernel_regularizer=regularizers.l2(Lambda)))

    

    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9)

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    

    # Fit the model

    model.fit(X_train, y_train, epochs=iterations,batch_size=500, verbose= 1)
def train_and_test_loop1(iterations, lr, Lambda, verb=True):



    ## hyperparameters

    iterations = iterations

    learning_rate = lr

    hidden_nodes = 256

    output_nodes = 10

        

    model = Sequential()

    model.add(Dense(hidden_nodes, input_shape=(784,), activation='relu'))

    model.add(Dense(hidden_nodes, activation='relu'))

    model.add(Dense(output_nodes, activation='softmax', kernel_regularizer=regularizers.l2(Lambda)))

    

    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9)

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    

    # Fit the model

    model.fit(X_train, y_train, epochs=iterations,batch_size=500, verbose= 1)

    score = model.evaluate(X_train, y_train, verbose=0)

    

    return score
lr = 0.00001

Lambda = 0

train_and_test_loop(1, lr, Lambda)
lr = 0.00001

Lambda = 1e3

train_and_test_loop(1, lr, Lambda)
lr = 1e-7

Lambda = 1e-7

train_and_test_loop(20, lr, Lambda)
lr = 1e8

Lambda = 1e-7

train_and_test_loop(20, lr, Lambda)
lr = 1e4

Lambda = 1e-7

train_and_test_loop(20, lr, Lambda)
import math

for k in range(1,10):

    lr = math.pow(10, np.random.uniform(-7.0, 3.0))

    Lambda = math.pow(10, np.random.uniform(-7,-2))

    best_acc = train_and_test_loop1(100, lr, Lambda, False)

    print("Try {0}/{1}: Best_val_acc: {2}, lr: {3}, Lambda: {4}\n".format(k, 100, best_acc, lr, Lambda))
for k in range(1,5):

    lr = math.pow(10, np.random.uniform(-4.0, -1.0))

    Lambda = math.pow(10, np.random.uniform(-4,-2))

    best_acc = train_and_test_loop1(100, lr, Lambda, False)

    print("Try {0}/{1}: Best_val_acc: {2}, lr: {3}, Lambda: {4}\n".format(k, 100, best_acc, lr, Lambda))
lr = 2e-2

Lambda = 1e-4

train_and_test_loop1(100, lr, Lambda)
lr = 2e-2

Lambda = 1e-4

hidden_nodes = 256

output_nodes = 10

model = Sequential()

model.add(Dense(hidden_nodes, input_shape=(784,), activation='relu'))

model.add(Dense(hidden_nodes, activation='relu'))

model.add(Dense(output_nodes, activation='softmax', kernel_regularizer=regularizers.l2(Lambda)))

    

sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9)

# Compile model

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    

 # Fit the model

model.fit(X_train, y_train, epochs=100,batch_size=500, verbose= 1)
predections = model.predict_classes(test,verbose=1)
sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

submission = sub['ImageId']
pred = pd.DataFrame(data=predections ,columns=["Label"])

DT = pd.merge(submission , pred, on=None, left_index= True,

    right_index=True)

DT.head()
DT.to_csv('submission.csv',index = False)