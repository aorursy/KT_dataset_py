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
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense
x_train = train.drop(['label'],axis=1)

y_train = train['label']
x_train = x_train/255.0

test = test/255.0



x_train = x_train.values.reshape(-1,28,28,1)

test = np.array(test).reshape(-1,28,28,1)
plt.imshow(test[0][:,:,0])
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,random_state=42,test_size=0.2)



y_train = pd.get_dummies(y_train)

y_test = pd.get_dummies(y_test)


model = Sequential()

model.add(Conv2D(filters=40,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))

model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Dropout(0.3))

model.add(Conv2D(filters=80,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))

model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(units=128,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(units=10,activation='softmax'))

model.compile(optimizer= 'adam',loss = 'categorical_crossentropy',metrics= ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(x_train)





from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=84),

                    steps_per_epoch=len(x_train) /84, epochs=15,validation_data=(x_test,y_test),callbacks=[learning_rate_reduction])

plt.figure(figsize=(15,5))



plt.plot(history.history['val_accuracy'])

plt.plot(history.history['accuracy'])



plt.title('Model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.grid()

plt.legend(['Test Accuracy','Training Accuracy'], loc='upper left')

axes = plt.gca()

axes.set_ylim([0.9,1])

plt.show()
import itertools



from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize=(10,12))



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

Y_pred = model.predict(x_test)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.array(y_test.idxmax(axis=1))

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
a = np.nonzero(Y_true-Y_pred_classes)

rows=6

columns=10

n=0

fig,ax = plt.subplots(rows,columns,sharex=True,sharey=True,figsize=(70,60))

#plt.figure(figsize=(70,60))



for i in range(0,rows):

    for j in range(0,columns):

        ax[i,j].imshow(x_test[a[0][n]][:,:,0])

        ax[i,j].set_title("Predicted label :{}\nTrue label :{}".format(Y_pred_classes[a[0][n]],Y_true[a[0][n]]),{'fontsize':40})

        n+=1                  

    
results = model.predict(test)

results = np.argmax(results,axis=1)

results = pd.Series(results,name='Label')
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission.csv",index=False)