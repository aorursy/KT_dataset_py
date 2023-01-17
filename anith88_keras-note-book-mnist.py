# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

# Any results you write to the current directory are saved as output.
digits = pd.read_csv('../input/train.csv')

digits.head()
digits.shape
index = 11

plt.imshow(((digits.iloc[index,1:]).values).reshape(28,28),cmap='Greys')

print('The lable is : ' + str(digits.iloc[index, 0]))
X_train = (digits.iloc[5000:,1:].values).reshape(digits.shape[0]-5000,28,28,1)/255.0

Y_train = to_categorical((digits.iloc[5000:,0].values).reshape(digits.shape[0]-5000,1))

X_test = (digits.iloc[:5000,1:].values).reshape(5000,28,28,1)/255.0

Y_test = to_categorical((digits.iloc[:5000:,0].values).reshape(5000,1))

print("X_train shape : "+str(X_train.shape))

print("Y_train shape : "+str(Y_train.shape))

print("X_test shape : "+str(X_test.shape))

print("Y_test shape : "+str(Y_test.shape))
index = 51

plt.imshow(X_train[index,:].reshape(28,28),cmap='Greys')

print("Image lable : "+str(Y_train[index]))
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
model.fit(X_train, Y_train, epochs=6, validation_data=(X_test, Y_test))
test_df = pd.read_csv('../input/test.csv')

test_df.shape
predictions = model.predict(((test_df.values).reshape(28000,28,28,1))/255.0)
predictions_array =  np.argmax(predictions, axis = 1)

print('pred shape : '+ str(predictions.shape))

print('predictions_array shape : '+ str(predictions_array.size))
index = 53

plt.imshow(test_df.iloc[index,:].values.reshape(28,28),cmap='Greys')

print("Predicted lable : "+str(predictions[index]))

print("predictions_array lable : "+str(predictions_array[index]))
imageId = np.array(range(1,predictions_array.size + 1))
my_submission = pd.DataFrame({'ImageId': imageId, 'Label': predictions_array})

my_submission.to_csv('submission.csv', index=False)
my_submission.head()