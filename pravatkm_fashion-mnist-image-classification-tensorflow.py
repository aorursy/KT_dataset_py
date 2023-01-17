# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

from tensorflow import keras

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
mnist = keras.datasets.fashion_mnist
type(mnist)
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()  ## using as tuple to store train and test data
print("X_train:",X_train.shape)  ### Displaying no of rows and column

print("Y_train:",Y_train.shape)

print("X_test:", X_test.shape)

print("Y_test:", Y_test.shape)
print("X_train max:",np.max(X_train))  ## Displaying maximum of array elements

print("X_test max:", np.max(X_test))

print("Y_train max:", np.max(Y_train))

print("Y_test max:", np.max(Y_test))
Y_train  ## It contains 10 items which are assigned from 0 to 9
Y_test  ## It contains 10 items
class_names = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'] \

## Labeling the items
plt.figure()    ## display ankle boot

plt.imshow(X_train[0])

plt.colorbar()
plt.figure()   ## display Top

plt.imshow(X_train[3])

plt.colorbar()
X_train = X_train/255.0

X_test = X_test/255.0
plt.figure()   ## Now the pixel is from 0 to 1.

plt.imshow(X_train[9])

plt.colorbar()
## Building the model with TF2.0

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten, Dense 
model = Sequential()

model.add(Flatten(input_shape = (28, 28)))##Flatten converts the multidimensional pixel into1D array to be fitted to denselayer

model.add(Dense(128, activation = 'relu'))  ## 128 neurons 

model.add(Dense(10, activation = 'softmax')) ## 10 output layer
model.summary()
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train,Y_train, epochs = 10)
from sklearn.metrics import accuracy_score
y_pred = model.predict_classes(X_test)
accuracy_score(Y_test, y_pred)
y_pred
pred = model.predict(X_test)
pred.shape  ## It has 10000 rows and 9 columns
pred[9]  ## displaying the 10th element
np.argmax(pred[9])  ## determining the highest prediction score