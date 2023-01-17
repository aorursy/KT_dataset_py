import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, AveragePooling2D, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
# Input data files are available in the "../input/" directory.
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
train.shape, test.shape
X = train.drop(labels=["label"], axis=1)
y = train["label"]

# check the shape
X.shape, y.shape
# checking manually
y.value_counts()
# Checking by plotting the same
plt.subplots(figsize = (10,8))
plt.title('Counts in numbers to their labels ')
sns.countplot(x=y, data=train)
plt.show()
X_train , X_test , y_train , y_test = train_test_split(X,y, test_size = 0.1 , random_state = 99)
# check the shape now
X_train.shape,X_test.shape,y_train.shape,y_test.shape,test.shape
X_train=X_train.values.astype('float32')
X_test=X_test.values.astype('float32')
test=test.values.astype('float32')
# changing the shape of X_train and y_train and test also
X_train=X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test=X_test.reshape(X_test.shape[0], 28, 28, 1)
test=test.reshape(test.shape[0] , 28 , 28 , 1)
X_train.shape,X_test.shape,test.shape
# check the maximum values in the dataset
X_train.max(),X_train.min()
X_train=X_train/255
X_test=X_test/255
test=test/255
# check the maximum values in the dataset
X_train.max(),X_train.min()
input_shape=X_train[0].shape
input_shape
# Build the Model
model= Sequential()
model.add(Conv2D(filters = 6, kernel_size = (5,5), activation='relu', input_shape = (28, 28, 1), padding = 'same'))
model.add(MaxPooling2D(pool_size = (2,2), strides=2))
model.add(Conv2D(filters=16, kernel_size = 5, activation='relu'))
model.add(MaxPooling2D(pool_size = 2, strides=2))
model.add(Conv2D(filters=120, kernel_size = 5, activation='relu'))
model.add(Flatten())
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))
# model.compile(optimizer= Adam(learning_rate =0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer= 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, batch_size=512, epochs=10, validation_split=0.2)
#Evaluating the Model
model.evaluate(X_test, y_test, verbose=0)
# plot confusion matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
y_pred = model.predict_classes(X_test)
class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
class_names
mat=confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=mat, class_names= class_names,show_normed=True, figsize=(7,7))
