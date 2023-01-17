# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import plotly.plotly as py

import plotly.graph_objs as go

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

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# plotting a random image

img = plt.imread("../input/flowers/flowers/daisy/100080576_f52e8ee070_n.jpg")

img = cv2.resize(img,(124,124))

plt.imshow(img)

plt.axis("off")

plt.show()
x_ = list()

y = list()

IMG_SIZE = 128

for i in os.listdir("../input/flowers/flowers/daisy"):

    try:

        path = "../input/flowers/flowers/daisy/"+i

        img = plt.imread(path)

        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

        x_.append(img)

        y.append(0)

    except:

        None

for i in os.listdir("../input/flowers/flowers/dandelion"):

    try:

        path = "../input/flowers/flowers/dandelion/"+i

        img = plt.imread(path)

        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

        x_.append(img)

        y.append(1)

    except:

        None

for i in os.listdir("../input/flowers/flowers/rose"):

    try:

        path = "../input/flowers/flowers/rose/"+i

        img = plt.imread(path)

        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

        x_.append(img)

        y.append(2)

    except:

        None

for i in os.listdir("../input/flowers/flowers/sunflower"):

    try:

        path = "../input/flowers/flowers/sunflower/"+i

        img = plt.imread(path)

        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

        x_.append(img)

        y.append(3)

    except:

        None

for i in os.listdir("../input/flowers/flowers/tulip"):

    try:

        path = "../input/flowers/flowers/tulip/"+i

        img = plt.imread(path)

        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

        x_.append(img)

        y.append(4)

    except:

        None

x_ = np.array(x_)
#plottin one of all flower types in data

plt.figure(figsize = (20,20))

for i in range(5):

    img = x_[950*i]

    plt.subplot(1,5,i+1)

    plt.imshow(img)

    plt.axis("off")

    plt.title(y[950*i])
# for replacement process i'll use keras.to_categorical 

from keras.utils.np_utils import to_categorical

y = to_categorical(y,num_classes = 5)
# test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_,y,test_size = 0.15,random_state = 42)
# validation and trains split

x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size = 0.15,random_state = 42)
plt.figure(figsize = (20,20))

for i in range(5):

    img = x_train[600*i]

    plt.subplot(1,5,i+1)

    plt.imshow(img)

    plt.axis("off")

    plt.title(y_train[600*i])

plt.show()
x_train.shape # look traing shape
model = Sequential()

# 1st Convolutional Layer

model.add(Conv2D(filters=64, kernel_size=(3,3),padding="Same",activation="relu" , input_shape = (IMG_SIZE,IMG_SIZE,3)))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.2))

# 2nd Convolutional Layer

model.add(Conv2D(filters=128, kernel_size=(3,3),padding="Same",activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.3))

# 3rd Convolutional Layer

model.add(Conv2D(filters=128, kernel_size=(3,3),padding="Same",activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.3))

# 4th Convolutional Layer

model.add(Conv2D(filters=256,kernel_size = (3,3),padding="Same",activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.2))

# 5th Convolutional Layer

model.add(Conv2D(filters=512,kernel_size = (3,3),padding="Same",activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Flatten())

# 1st Fully Connected Layer

model.add(Dense(1024,activation="relu"))

model.add(Dropout(0.5))

model.add(BatchNormalization())

# Add output layer

model.add(Dense(5,activation="softmax"))



model.summary() # print summary my model

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) #compile model
model.compile(loss='categorical_crossentropy',

              optimizer=Adam(lr=0.001),

              metrics=['accuracy'])
epoch = 50 

batch_size = 64
datagen = ImageDataGenerator(

    featurewise_center=False,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization=False,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    rotation_range=60,  # randomly rotate images in the range (60, 0 to 180)

    zoom_range = 0.1, # Randomly zoom image 

    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

    height_shift_range=0.1,

    shear_range=0.1,

    fill_mode = "reflect"

    ) 

datagen.fit(x_train)
history = model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),

                              epochs= epoch,validation_data=(x_val,y_val),

                              steps_per_epoch=x_train.shape[0] // batch_size

                              )
print("Test Accuracy: {0:.2f}%".format(model.evaluate(x_test,y_test)[1]*100)) #get score acording to test datas
x_ = np.array(range(len(history.history['loss']))) # get loss values from the history

trace1 = go.Scatter(

        x = x_,

        y = history.history['loss'], # get loss values from the history

        mode = "lines",

        marker = dict(color = "rgba(0,255,0,0.9)"),

        text = "Loss"

)

trace2 = go.Scatter(

        x = x_,

        y = history.history['acc'],# get accuracy values from the history

        mode = "lines",

        marker = dict(color = "rgba(0,0,255,0.9)"),

        text = "Accuracy"

)

data = [trace1,trace2]

layout = dict(title = "Training Accuracy and Loss")

fig = dict(data = data,layout=layout)

iplot(fig)
x_ = np.array(range(len(history.history['val_loss'])))# get validation loss values from the history

trace1 = go.Scatter(

        x = x_,

        y = history.history['val_loss'], # get validation loss values from the history

        mode = "lines",

        marker = dict(color = "rgba(0,0,0,0.9)"),

        text = "Validation Loss"

)

trace2 = go.Scatter(

        x = x_,

        y = history.history['val_acc'],# get validation accuracy values from the history

        mode = "lines",

        marker = dict(color = "rgba(255,0,0,0.9)"),

        text = "Validation Accuracy"

)

data = [trace1,trace2]

layout = dict(title = "Validation Accuracy and Loss")

fig = dict(data = data,layout=layout)

iplot(fig)
from sklearn.metrics import confusion_matrix

Y_pred = model.predict(x_val)

Y_pred_classes = np.argmax(Y_pred,axis = 1)

Y_true = np.argmax(y_val,axis = 1)

confusion_mtx = confusion_matrix(Y_true,Y_pred_classes)

f,ax = plt.subplots(figsize = (8,8))

sns.heatmap(confusion_mtx,annot=True,linewidths = 0.01,cmap="Reds",

            linecolor = "gray",fmt = ".2f",ax=ax

            )

plt.xlabel("predicted label")

plt.ylabel("True Label")

plt.title("confusion matrix")

plt.show()