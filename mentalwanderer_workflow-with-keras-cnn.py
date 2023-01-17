import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks")

# Artificial neural network packages
from keras.models import *
from keras.layers import *
from keras.wrappers.scikit_learn import *
from keras.optimizers import *
from keras.metrics import *
from keras.callbacks import *
from keras.utils.np_utils import * 
import keras.backend as K

from sklearn.model_selection import train_test_split

import os
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
numTrain,numTest = len(train),len(test)
print("Length of training data is: " + str(numTrain) + " entries")
print("Length of test data is: " + str(numTest) + " entries")
train.info()
train.describe()
test.info()
test.describe()
_ = plt.hist(train["label"],bins = 20)
train.isnull().any().any()
test.isnull().any().any()
fullData = pd.concat([train,test]).reset_index(drop = True)
fullData.drop(['label'],axis = 1,inplace = True)
y_train = train["label"]
y_train = to_categorical(y_train)
fullData = fullData / 255.0
fullData = fullData.values.reshape(-1,28,28,1)
X_train = fullData[:numTrain]
X_test = fullData[numTrain:]
np.random.seed(1)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = 0.2,random_state = 1)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dense(10, activation = "softmax"))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
model.fit(X_train,y_train,epochs = 30,batch_size = 128,validation_data = (X_val,y_val),callbacks = [learning_rate_reduction])
model.evaluate(X_val,y_val,batch_size = 128)
# Predictions
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob,axis = 1)
results = pd.read_csv("../input/sample_submission.csv")
results['Label'] = y_pred
results.to_csv("submission.csv",index = False)