#To Predict A Happy Face Using CNN
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

import keras 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation, Dropout
from keras.optimizers import Adam, RMSprop ,Adadelta
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
def load_dataset():
    train_dataset = h5py.File("../input/train_happy.h5","r")
    train_x = np.array(train_dataset['train_set_x'][:])
    train_y = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File("../input/test_happy.h5","r")
    test_x = np.array(test_dataset['test_set_x'][:])
    test_y = np.array(test_dataset['test_set_y'][:])
    # y reshaped

    train_y = train_y.reshape((1, train_y.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))

    return train_x,train_y,test_x,test_y
X_train,Y_train,X_test,Y_test=load_dataset()
X_train=X_train/255
X_test=X_test/255
Y_train=Y_train.T
Y_test=Y_test.T
print("X_train Shape-{}".format(X_train.shape))
print("Number of training examples-{}".format(X_train.shape[0]))
print("X_test Shape-{}".format(X_test.shape))
print("Number of testing examples-{}".format(X_test.shape[0]))
print("Y_train Shape-{}".format(Y_train.shape))
print("Y_test Shape-{}".format(Y_test.shape))
img=X_train[10]
plt.imshow(img)
model=Sequential()
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(BatchNormalization())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1,activation='sigmoid',kernel_regularizer=regularizers.l1(0.01)))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, Y_train, batch_size=20, epochs=35)
# Predict the test set results
Y_pred = model.predict_classes(X_test)

print ("test accuracy: %s" %accuracy_score(Y_test, Y_pred))
print ("precision: %s"  %precision_score(Y_test, Y_pred))
print ("recall: %s" %recall_score(Y_test, Y_pred))
print ("f1 score: %s"  %f1_score(Y_test, Y_pred))


cm = confusion_matrix(Y_test,Y_pred)
sns.heatmap(cm,annot=True)
