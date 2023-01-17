# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import keras
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#checking the dimesions of data
print(train.shape)
print(test.shape)
#separating labels and images from data
X = train.iloc[:,1:]
y = train.iloc[:,0]
X.shape
#checking for null values
X.isnull().any().describe()
#checking for null values
test.isnull().any().describe()
#normalizing data
X =X / 255.0
test = test / 255.0
#reshaping the data
X = X.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
X.shape
test.shape
#label encoding
from keras.utils.np_utils import to_categorical
y = to_categorical(y, num_classes = 10)
# Split the train and the validation set for the fitting
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=2)
#defining layers
model = Sequential()

model.add(Conv2D(filters = 128, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1024, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
#defining optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#compiling model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
#training model
mnist_train =model.fit(X_train, y_train, validation_data=(X_test,y_test),  batch_size=50, epochs=10, verbose=1)
#predictions
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)
y_true = np.argmax(y_test ,axis=1)
#confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_true , y_pred)
#overall accuracy on validation set
accuracy_score(y_true , y_pred)
#visualizing first five predicted values
pd.DataFrame({"y_true":y_true[:5],"y_predicted":y_pred[:5]})
#predicting result for submission
submission = model.predict(test)
submission = np.argmax(submission,axis = 1)
submission = pd.DataFrame({"ImageId":range(1,28001),"Label":submission})
submission.to_csv("submission.csv",index=False)
