# loading the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# loading the data

seed = 6
np.random.seed(seed)

training_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')
# Here, training set is used for training and evaluating the performance of the neural network. Final predictions are made on the Test set.
# checking the characteristics of data
training_set.head()
# storing the class labels and image values into separate variables.

X = training_set.iloc[:, 1:].values
y = training_set.iloc[:, 0].values
# checking the data characteristics

print ('Input vectors : {}'.format(X.shape))
print ('Class Labels : {}'.format(y.shape))
# checking the first 9 images

images = X.reshape(-1,28,28)

for i in range(0,9):
    plt.subplot(330 + 1 + i)
    img = images[i]
    plt.imshow(img, cmap = 'gray')

plt.show()
# Splitting the dataset into the Training and Validation set

from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.1, random_state = 0)
# normalizing the inputs

X_train = X_train.astype('float32')/255.0
X_validation = X_validation.astype('float32')/255.0
# checking the class labels

print (y_train.shape)
print (y_train[0])    # gives the value stored at 1 position
from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation)

print (y_train.shape)
print (y_train[0])
print (y_validation.shape)
print (y_validation[0])
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# building the model
model = Sequential()
# adding first hidden layer
model.add(Dense(397, kernel_initializer = 'uniform', input_dim = 784, activation = 'relu'))
model.add(Dropout(0.2))

# adding output layer
model.add(Dense(10, kernel_initializer = 'uniform', activation = 'softmax'))
# compiling the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# fitting the model to training data
model.fit(X_train, y_train, batch_size = 100, epochs = 10)
# evaluating the model

loss_accuracy = model.evaluate(X_validation, y_validation, batch_size = 100)

print ('Loss = {}'.format(loss_accuracy[0]))
print ('Accuracy = {}'.format(loss_accuracy[1]))
X_test = test_set.values

# normalizing the inputs
X_test = X_test.astype('float32')/255.0

# predicting the values
y_pred = model.predict(X_test)

# selecting the class with highest probability
y_pred = np.argmax(y_pred, axis=1)
#Create a  DataFrame with the results

results = pd.DataFrame({'ImageID':pd.Series(range(1,28001)),
                        'Label':y_pred})

filename = 'Digit Recognition Predictions.csv'

results.to_csv(filename, index=False)

print('Saved file: ' + filename)
