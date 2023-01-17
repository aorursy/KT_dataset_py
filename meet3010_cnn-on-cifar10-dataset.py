import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense
# import the dataset
from tensorflow.keras.datasets import cifar10
# dividing the data into trainind and testing data
(X_train,y_train),(X_test,y_test) = cifar10.load_data()

# check the shape of the data
X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train,y_train
shape =X_train[1].shape
shape
target_class=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
target_class
plt.imshow(X_train[1])
plt.colorbar()
plt.show()
# check the range
X_train.max(), X_train.min()
# scaling the data
X_train = X_train/255
X_test = X_test/255

# check the range
X_train.max(), X_train.min()
# import the model
model=Sequential()
# import the layers
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', input_shape=shape))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),padding='same'))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10,activation='softmax'))
# check the summary
model.summary()
# compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
%%time
# fitting the model
history=model.fit(X_train,y_train,epochs=10,batch_size=32,validation_data=(X_test,y_test))
loss,accuracy=model.evaluate(X_test,y_test)
loss, accuracy
y_pred=model.predict_classes(X_test)
y_pred
# plot confusion matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
mat=confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=mat, class_names= target_class,show_normed=True, figsize=(7,7))
