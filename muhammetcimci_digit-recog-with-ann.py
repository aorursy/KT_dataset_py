# import necessary libraries
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# # load dataset
# from keras.datasets import mnist


# # split dataset into training and test set
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
y_train = train.iloc[:,0:1]
x_train = train.iloc[:,1:]

x_test =test
x_train.shape, y_train.shape 
# # Display images

import matplotlib.pyplot as plt

# w=100
# h=10
# fig=plt.figure(figsize=(20, 20))
# columns = 20
# rows = 1
# for i in range(1, columns*rows +1):
#     img = np.random.randint(10, size=(h,w))
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(x_train[i-1], cmap=plt.cm.binary)
# plt.show()

# # for i in range (0,5):
# #     plt.imshow(x_train[i], cmap=plt.cm.binary)
# #     plt.show()
#y_train[0:20]
# View number of dimensions of tensor

print(x_train.ndim)
# View the dimension of tensor

print(x_train.shape)
x_train.info()

# View the data type of tensor

#print(x_train.dtype)
# scale the input values to type float32

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')
# scale the input values within the interval [0, 1]

x_train /= 255
x_test /= 255
x_train.shape, x_test.shape
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
print(x_train.shape)
print(x_test.shape)
from keras.utils import to_categorical
#print(y_test[0])
#print(y_train)
print(y_train.shape)
#print(x_test.shape)
y_train = to_categorical(y_train, num_classes=10)

#y_test = to_categorical(y_test, num_classes=10)
#print(y_test[0])
print(y_train[0])
print(y_train.shape)
#print(y_test.shape)

from keras.models import Sequential
from keras.layers.core import Dense, Activation


model = Sequential()
model.add(Dense(70, activation='sigmoid', input_shape=(784,)))
model.add(Dense(60, activation='tanh'))
model.add(Dense(50, activation='softmax'))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(30, activation='tanh'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(loss="categorical_crossentropy",
              optimizer= "Adam",
              metrics = ['accuracy'])

X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.20, random_state=42)
model.fit(X_train, Y_train, batch_size=512, epochs=200)
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', round(test_acc,4))
# # Plot confusion matrix 
# # Note: This code snippet for confusion-matrix is taken directly from the SKLEARN website.
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=30)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('Actual class')
#     plt.xlabel('Predicted class')
# from collections import Counter
# from sklearn.metrics import confusion_matrix
# import itertools

# # Predict the values from the validation dataset
# Y_pred = model.predict(Y_test)
# # Convert predictions classes to one hot vectors 
# Y_pred_classes = np.argmax(Y_pred, axis = 1) 
# # Convert validation observations to one hot vectors
# Y_true = np.argmax(Y_test, axis = 1) 
# # compute the confusion matrix
# confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# # plot the confusion matrix
# plot_confusion_matrix(confusion_mtx, classes = range(10))

predictions = model.predict(X_test)
predictions[0:2]
np.argmax(predictions[11])
preds = model.predict_proba(x_test)[:,1]
# predict results
results = model.predict(x_test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("ann_submit.csv",index=False)
np.sum(predictions[11])
### Get the basics

# import Keras library
#import keras


# load dataset
# from keras.datasets import mnist


# split dataset into training and test set
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

#############################################################################

### Display images

# import matplotlib.pyplot as plt

# plt.imshow(x_train[7], cmap=plt.cm.binary)

# View the labels

# print(y_train[7])

#############################################################################

### Data representation in Keras


# View number of dimensions of tensor

# print(x_train.ndim)



# View the dimension of tensor

# print(x_train.shape)



# View the data type of tensor

# print(x_train.dtype)


##############################################################################

### Data normalization in Keras


# Scale the input values to type float32

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')


# Scale the input values within the interval [0, 1]
# x_train /= 255
# x_test /= 255


# Reshape the input values
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)

# print(x_train.shape)
# print(x_test.shape)


# from keras.utils import to_categorical

# print(y_test[0])
# print(y_train[0])

# print(y_train.shape)
# print(x_test.shape)

# y_train = to_categorical(y_train, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)

# print(y_test[0])
# print(y_train[0])

# print(y_train.shape)
# print(y_test.shape)

###############################################################################

### Define the model

# from keras.models import Sequential
# from keras.layers.core import Dense, Activation

# model = Sequential()
# model.add(Dense(10, activation='sigmoid', input_shape=(784,)))
# model.add(Dense(10, activation='softmax'))

###############################################################################

### Model summary

# model.summary()

###############################################################################

### Implementation of Neural Network in Keras


# Compiling the model with compile() method

# model.compile(loss="categorical_crossentropy",
# optimizer="sgd", metrics = ['accuracy'])	


# Training the model with fit() method

# model.fit(x_train, y_train, batch_size=100, epochs=10)


# Evaluate model with evaluate() method

# test_loss, test_acc = model.evaluate(x_test, y_test)


################################################################################

### Accuracy of the model

# print('Test accuracy:', round(test_acc,4))
