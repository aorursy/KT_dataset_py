# Libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os

# Reading the input images and putting them into a numpy array
data=[]
labels=[]

#Dimensions of Resized Image
dim = (32, 32)
classes = 43

for i in range(classes) :
    path = "../input/Train/{0}/".format(i)
    print(path)
    Class=os.listdir(path)
    for a in Class:
        try:
            image=cv2.imread(path+a)
            size_image = cv2.resize(image, dim)
            image_sum =size_image.sum(axis=2)
            data.append(np.array(image_sum))
            labels.append(i)
        except AttributeError:
            print(" ")
            
Cells=np.array(data)
labels=np.array(labels)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split( Cells, labels, test_size=0.20, random_state=7777)

#Standardising X_train Values between 0 and 1

X_train = X_train.astype('float32')/765
X_val = X_val.astype('float32')/765
#Encoding for the train and validation labels
from keras.utils import to_categorical
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)
X_train = X_train.reshape(-1, 32, 32, 1)
X_val = X_val.reshape(-1, 32, 32, 1)
print("X_train Shape: ", X_train.shape)
print("X_test Shape: ", X_val.shape)
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(32,32,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))

model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

model.summary()
#using ten epochs for the training and saving the accuracy for each epoch
epochs = 10
history = model.fit(X_train, y_train, batch_size=50, epochs=epochs,validation_data=(X_val, y_val))
print(history.history.keys())
#Display of the accuracy and the loss values
import matplotlib.pyplot as plt

plt.figure(0)
plt.plot(history.history['acc'], label='training accuracy')
plt.plot(history.history['val_acc'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

y_test=pd.read_csv("../input/Test.csv")
labels=y_test['Path'].to_numpy()
y_test=y_test['ClassId'].values

data=[]

for f in labels:
    image=cv2.imread('../input/Test/'+f.replace('Test/', ''))
    size_image = cv2.resize(image, dim)
    image_sum =size_image.sum(axis=2)

    data.append(np.array(image_sum))
   
X_test=np.array(data).reshape(-1, 32, 32, 1)
X_test = X_test.astype('float32')/765 

pred = model.predict_classes(X_test)
#Accuracy with the test data
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)
from sklearn.metrics import confusion_matrix
cnf_matrix=confusion_matrix(y_test, pred)
print(cnf_matrix)

import matplotlib.pyplot as plt
import numpy as np
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.GnBu):
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
import seaborn as sns
%matplotlib inline
sns.set_style("darkgrid")
plt.figure(figsize=(20,20))
plt.title("Confusion Matrix ")
plt.grid(False)
# call pre defined function
plot_confusion_matrix(cnf_matrix, classes=range(43))