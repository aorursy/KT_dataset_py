import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import random

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report

import keras

from keras.optimizers import Adam

from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense

from keras.callbacks import TensorBoard

from keras.models import Sequential

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Importing dataset

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#Viewing the data

train.head()
test.head()
train.shape
test.shape
#Lets try to visualize any random number

training = np.array(train, dtype = 'float32')

testing = np.array(test, dtype = 'float32')
i = random.randint(0, len(training))

plt.imshow(training[i, 1:].reshape(28, 28))

label = training[i, 0]

label
train['label'].value_counts()
x = []

w_grid = 5

l_grid = 2

fig, axes = plt.subplots(l_grid, w_grid, figsize = (20, 8))

axes = axes.ravel()

i = 0

while len(x) < 10:

    index = np.random.randint(0, len(training))

    if training[index, 0] not in x:

        axes[i].imshow(training[index, 1:].reshape((28, 28))) 

        x.append(training[index, 0])

        i += 1

X_train = training[:, 1:]

y_train = training[:, 0]
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)
#Testing is the testdata set that we will use to test at the end

#Reshaping the datasets

X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))

X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))
X_train.shape
X_validate.shape
#Initializing cnn

cnn = Sequential()
#Adding the convolutional layer

cnn.add(Conv2D(128, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))
#Adding the max pool layer

cnn.add(MaxPooling2D(pool_size = (2, 2)))
#Adding the flattening layer

cnn.add(Flatten())
#Adding the ann layer 

cnn.add(Dense(output_dim = 32, activation = 'relu'))
#Adding the output layer

cnn.add(Dense(output_dim = 10, activation = 'sigmoid'))
cnn.compile(optimizer= Adam(lr = 0.001), loss = 'sparse_categorical_crossentropy', metrics= ['accuracy'])
epochs = 50

cnn.fit(X_train, y_train, batch_size = 512, epochs = epochs, validation_data = (X_validate, y_validate), verbose = 1,)
cnn.summary()
prediction = cnn.predict_classes(X_validate)
cm = confusion_matrix(y_validate, prediction)
plt.figure(figsize = (15,10))

sns.heatmap(cm, annot = True)
num_classes = 10

target_names = ['Class {}'.format(i) for i in range(num_classes)]



print(classification_report(y_validate, prediction, target_names = target_names))
test = np.array(test, 'int')

Test = test.reshape(test.shape[0], *(28, 28, 1))

result = cnn.predict(Test)
class_prediction = cnn.predict_classes(Test)
class_prediction
result = np.argmax(result,axis = 1)

result = pd.Series(result,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],axis = 1)



submission.to_csv("Prediction_test_output.csv",index=False)


