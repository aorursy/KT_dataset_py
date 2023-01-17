# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

#import plotly.plotly as py
#import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot # plotly offline mode
init_notebook_mode(connected=True) 

import seaborn as sns
import cv2

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import random
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/digit-recognizer"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/digit-recognizer/train.csv")
df.head()
df.shape
X=df.drop('label' , axis=1)
X.head()
Y = df['label']
print(X.describe())
print(Y.describe())
X=X/255.0
X = np.array(X).reshape(X.shape[0],28,28)
Y = keras.utils.to_categorical(Y,num_classes = 10)
plt.figure(0, figsize=(12,12))
for i in range(1,3):
    plt.subplot(1,3,i)
    img = np.squeeze(X[np.random.randint(0, X.shape[0])])
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
plt.imshow(np.squeeze(X[99]), cmap="gray");
X = X.reshape(-1,28,28,1)
X_train, X_val, Y_train, Y_val = train_test_split(X,Y,test_size = 0.20)
X_train.shape
model = keras.Sequential()

model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'same',
                              activation = 'relu',input_shape = (28,28,1)))
model.add(keras.layers.Conv2D(filters = 64,kernel_size = (5,5),
                              activation = 'relu'))

model.add(keras.layers.MaxPooling2D((2,2),strides = (1,1),padding = 'same'))

model.add(keras.layers.Conv2D(filters = 32,kernel_size = (3,3),
                             activation = 'relu',padding = 'same'))
model.add(keras.layers.Conv2D(filters = 32,kernel_size = (3,3),
                             activation = 'relu'))

model.add(keras.layers.MaxPooling2D((2,2),padding = 'same'))

model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(256,activation = 'relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(10,activation = 'softmax'))
import tensorflow as tf
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=5,
    decay_rate=0.96,
    staircase=True)

optimizer = Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train,Y_train,epochs= 25,batch_size=64,validation_data=(X_val,Y_val)) 
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(train_acc,label = "Training")
plt.plot(val_acc,label = 'Validation/Test')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
train_loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(train_loss,label = 'Training')
plt.plot(val_loss,label = 'Validation/Test')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
X_test = pd.read_csv('../input/digit-recognizer/test.csv')
X_test = X_test/255.0
X_test = np.array(X_test).reshape(X_test.shape[0],28,28)
X_test = X_test.reshape(-1,28,28,1)
X_test.shape
# confusion matrix
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
test_predictions = model.predict_classes(X_test)
test_predictions = test_predictions.reshape(28000,1)
idx = 825
img = X_test[idx].reshape(28,28)
plt.imshow(img)
print(test_predictions[idx])
image_id = np.arange(1,28001).reshape(28000,1)
test_results = np.concatenate((image_id,test_predictions),axis=1)
np.savetxt('test_predictions.csv',test_results,fmt="%d",header="ImageId,Label",delimiter=",")