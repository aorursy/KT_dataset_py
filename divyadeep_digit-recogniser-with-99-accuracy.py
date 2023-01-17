# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/sample_submission.csv")
Xtrain=train.iloc[:,1:]
Ytrain=train.iloc[:,0]
Xtrain = Xtrain/255
seed = 1
x_train , x_test ,y_train ,y_test = train_test_split(Xtrain,Ytrain,test_size=0.1,random_state=seed)
x_train.shape , y_train.shape , x_test.shape
x_train = x_train.values.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.values.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
x_train.shape , y_train.shape , x_test.shape , y_train
plt.imshow(x_train[1000].reshape(28,28))
num_classes = y_test.shape[1]


# create model
model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(100, activation = "relu"))
model.add(Dense(50, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = "softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=100, verbose=1)

# Final evaluation of the model
scores = model.evaluate(x_test,y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
test = test.values.reshape(28000,28,28,1)/255
prediction = model.predict_classes(test)
np.sum()
plt.imshow(test[5].reshape(28,28))
for i in range(28000):
    print(str(i+1)+","+str(prediction[i]))
model.save_weights('my_model_weights.h5')
model.save('weights.model')
