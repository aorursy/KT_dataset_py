# import necessary libraries
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
#import digit dataset as train and test set
train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
# to see first 5 rows 
train.head()
# seperate target feature as y_train and define x_train set without target feature.
y_train = train.iloc[:,0:1]
x_train = train.iloc[:,1:]

# to see shape of the datasets, 1st values in brackets are sample counts and 2nd values are pixels counts
x_train.shape, y_train.shape, test.shape
# View number of dimensions of tensor

print(x_train.ndim)
# View the dimension of tensor

print(x_train.shape)
x_train.info(), test.info()
# scale the input values to type float32

x_train = x_train.astype('float32')
test = test.astype('float32')
# scale the input values within the interval [0, 1]

x_train /= 255
test /= 255
from keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=10)
y_train[0]
# seperate dataset with train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
from keras.models import Sequential
from keras.layers.core import Dense, Activation


model = Sequential()
model.add(Dense(100, activation='sigmoid', input_shape=(784,)))
model.add(Dense(90, activation='relu'))
model.add(Dense(80, activation='tanh'))
model.add(Dense(70, activation='sigmoid'))
model.add(Dense(60, activation='relu'))
model.add(Dense(50, activation='tanh'))
model.add(Dense(10, activation='softmax'))

model.summary()
#sgd = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss="categorical_crossentropy",
              optimizer= "RMSprop",
              metrics = ['accuracy'], run_eagerly=False)
model.fit(x_train, y_train, batch_size=256, epochs=200)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', round(test_acc,4))
# Plot confusion matrix 
# Note: This code snippet for confusion-matrix is taken directly from the SKLEARN website.
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')
from collections import Counter
from sklearn.metrics import confusion_matrix
import itertools

# Predict the values from the validation dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test, axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10))

test_result = model.predict(test)


results = np.argmax(test_result,axis = 1) 

results = pd.Series(results,name="Label")


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission_best.csv",index=False)
