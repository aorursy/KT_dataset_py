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
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
x_train=train.drop(['label'],axis=1)

y_train=train['label']
x_train=x_train/255

test=test/255
x_train=x_train.values.reshape(-1,28,28,1)

test=test.values.reshape(-1,28,28,1)
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

y_train = to_categorical(y_train, num_classes = 10)
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.1,random_state=42)
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')
g = plt.imshow(x_train[0][:,:,0])
from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
classifier=Sequential()
classifier.add(Conv2D(32,(5,5),input_shape=(28,28,1),activation='relu'))
classifier.add(Conv2D(32,(5,5),activation='relu'))
classifier.add(MaxPooling2D(2,2))

classifier.add(Dropout(0.25))
classifier.add(Conv2D(64,(3,3),activation='relu'))

classifier.add(Conv2D(64,(3,3),activation='relu'))

classifier.add(MaxPooling2D(2,2))

classifier.add(Dropout(0.25))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))

classifier.add(Dense(units=10, activation = 'softmax'))
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
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
batchsize=50

epochs=25
history = classifier.fit_generator(datagen.flow(x_train,y_train, batch_size=batchsize),

                              epochs = epochs, validation_data = (x_val,y_val),verbose=2,

                              steps_per_epoch=x_train.shape[0] // batchsize

                              )
y_result=classifier.predict(test)
y_result

# Look at confusion matrix 

from sklearn.metrics import confusion_matrix

import itertools

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

    plt.xticks(tick_marks, classes, rotation=45)

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

y_pred = classifier.predict(x_val)

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
results = np.argmax(y_result,axis = 1)



results = pd.Series(results,name="Label")
results

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)





# import the modules we'll need

from IPython.display import HTML



import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)





# create a link to download the dataframe

create_download_link(results)
submission.to_csv('submission.csv',index=False)