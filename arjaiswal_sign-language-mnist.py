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
#Importing required libraries
from keras import layers, models
import matplotlib.pyplot as plt
%matplotlib inline
from scipy import ndimage
#Loading dataset as DataFrames using pandas
sign_train = pd.read_csv("../input/sign_mnist_train.csv")
sign_test = pd.read_csv("../input/sign_mnist_test.csv")
#Looking at the Dataset, it has 1st column has label and rest columns are pixels. 
#Let's check how many pixel does each row has.
sign_train.head()
#Looking at out of this command it shows, there are 785 columns, out which 1 is label.
#So, there are 784 pixels in each row
sign_train.columns
#The "shape" shows us that there are 27455 rows, so we have 27455 records for training.
sign_train.shape
#We will need the actual values to work with, so lets get them in numpy arrays
x_train = sign_train.iloc[:, 1:].values
y_train = sign_train.iloc[:, 0].values

x_test = sign_test.iloc[:, 1:].values
y_test = sign_test.iloc[:, 0].values
#Lets check how many labels we can classify our data into
#Maximum value in the label
max([i for i in y_test])
#Minimum value in the label
min([i for i in y_test])
#Let's see how the data looks
#There are 784 pixels as 28*28 pixel image
img = np.array(x_train[0,:].reshape(28, 28))
plt.imshow(img)
#To just verify, lets check labels as well 
y_train[0]
#One hot encoding with 25 dimensions
#The following method helps to enable the label as one hot encoded values
def vectorize_sequence(labels, dimension=25):
    results = np.zeros([len(labels), dimension])
    for i, sequence in enumerate(labels):
        results[i, sequence] = 1
    return results
#Getting max value from the image pixels
max(max([i for i in seq] for seq in x_train))
#We need to scale the pixel data to values between 0 to 1
#As well as vectorize labels
#Scaling train data
x_train = x_train.astype('float32') / 255
y_train = vectorize_sequence(y_train)
#scaling test data
x_test = x_test.astype('float32') / 255
y_test = vectorize_sequence(y_test)
#It's always better to keep some data to validate the trained model before we actually test with our test data
#Lets keep some data to validate
x_data = x_train[:20000,]
y_data = y_train[:20000]
x_data = x_data.reshape(x_data.shape[0], 28, 28, 1)
x_val = x_train[20000:, ]
y_val = y_train[20000:]
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_val.shape
#Lets desgin model
model = models.Sequential()
model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (5,5), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(25, activation='softmax'))
model.summary()
#Let's compile the model now
model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=["acc"])
#Generally we can run the model for 20 epochs, but we will get stable results after 9-10 epochs
results = model.fit(x_data, y_data, epochs=10, batch_size=512, 
                    validation_data=(x_val, y_val))
#The "results" has history attribute which has historical values for loss and accuracy for all epochs
results.history.keys()
#Let's plot the Training vs Validation Loss and Accuracy
#Training vs Validation Loss
epochs = len(results.history['acc'])+1
plt.plot(range(1, epochs), results.history['acc'], 'b', label="Training Accuracy")
plt.plot(range(1, epochs), results.history['val_acc'], 'r', label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()
#Training vs Validation Accuracy
plt.plot(range(1, epochs), results.history['loss'], 'b', label="Training Loss")
plt.plot(range(1, epochs), results.history['val_loss'], 'r', label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()
#Lets train with 9 epochs now
#Redifining the model
model = models.Sequential()
model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (5,5), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(25, activation='softmax'))
model.summary()
#Recompiling the model now
model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=["acc"])
results = model.fit(x_data, y_data, epochs=9, batch_size=512, 
                    validation_data=(x_val, y_val))
#Let's test now with our test data
test_loss, test_acc = model.evaluate(x_test, y_test)
#The above will give error, as our model expects data of the shape (a, 28, 28, b)
#Let's reshape the test data as we have already reshaped training data
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test.shape
#Now finally lets test the model with test data
test_loss, test_acc = model.evaluate(x_test, y_test)
#To see the test loss
test_loss
#To see the test accuracy
test_acc
