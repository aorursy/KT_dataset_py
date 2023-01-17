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
import os

print(os.listdir('/kaggle/input'))
import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt



from plotly.offline import init_notebook_mode, iplot 

init_notebook_mode(connected=True) 



import seaborn as sns

import cv2



import keras

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten

from keras.layers import Conv2D

from keras.layers import MaxPooling2D,MaxPool2D

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, Input, Dropout,Flatten, Conv2D

from tensorflow.keras.layers import BatchNormalization, Activation, MaxPooling2D



from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.utils import plot_model

from sklearn.metrics import confusion_matrix



import os

print(os.listdir("../input/flowers-recognition"))
img = plt.imread("../input/flowers-recognition/flowers/daisy/10172379554_b296050f82_n.jpg")

img = cv2.resize(img,(124,124))

plt.imshow(img)

plt.axis("off")

plt.show()
x_ = list()

y = list()

IMG_SIZE = 128

for i in os.listdir("../input/flowers-recognition/flowers/daisy"):

    try:

        path = "../input/flowers-recognition/flowers/daisy/"+i

        img = plt.imread(path)

        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

        x_.append(img)

        y.append(0)

    except:

        None

for i in os.listdir("../input/flowers-recognition/flowers/dandelion"):

    try:

        path = "../input/flowers-recognition/flowers/dandelion/"+i

        img = plt.imread(path)

        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

        x_.append(img)

        y.append(1)

    except:

        None

for i in os.listdir("../input/flowers-recognition/flowers/rose"):

    try:

        path = "../input/flowers-recognition/flowers/rose/"+i

        img = plt.imread(path)

        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

        x_.append(img)

        y.append(2)

    except:

        None

for i in os.listdir("../input/flowers-recognition/flowers/sunflower"):

    try:

        path = "../input/flowers-recognition/flowers/sunflower/"+i

        img = plt.imread(path)

        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

        x_.append(img)

        y.append(3)

    except:

        None

for i in os.listdir("../input/flowers-recognition/flowers/tulip"):

    try:

        path = "../input/flowers-recognition/flowers/tulip/"+i

        img = plt.imread(path)

        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))

        x_.append(img)

        y.append(4)

    except:

        None

x_ = np.array(x_)
plt.figure(figsize = (20,20))

for i in range(5):

    img = x_[745*i]

    plt.subplot(1,5,i+1)

    plt.imshow(img)

    plt.axis("off")

    plt.title(y[745*i])
y = to_categorical(y,num_classes = 5)
x_
y
x_train,x_test,y_train,y_test = train_test_split(x_,y,test_size = 0.15,random_state = 42)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size = 0.15,random_state = 42)
plt.figure(figsize = (40,40))

for i in range(5):

    img = x_train[600*i]

    plt.subplot(1,5,i+1)

    plt.imshow(img)

    plt.axis("off")

    plt.title(y_train[600*i])

plt.show()
x_train.shape
y_test.shape
datagen = ImageDataGenerator(

    featurewise_center=False,  

    samplewise_center=False,  

    featurewise_std_normalization=False,  

    samplewise_std_normalization=False,  

    rotation_range=60, 

    zoom_range = 0.1,  

    width_shift_range=0.1,  

    height_shift_range=0.1,

    shear_range=0.1,

    fill_mode = "reflect"

    ) 

datagen.fit(x_train)
x_train.shape
model = Sequential()



model.add(Conv2D(filters=64, kernel_size=(3,3),padding="Same",activation="relu" , input_shape = (IMG_SIZE,IMG_SIZE,3)))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Conv2D(filters=128, kernel_size=(3,3),padding="Same",activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Conv2D(filters=128, kernel_size=(3,3),padding="Same",activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Conv2D(filters=256,kernel_size = (3,3),padding="Same",activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.2))



model.add(Conv2D(filters=512,kernel_size = (3,3),padding="Same",activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Flatten())



model.add(Dense(1024,activation="relu"))

model.add(Dropout(0.5))

model.add(BatchNormalization())



model.add(Dense(5,activation="softmax"))



model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) #compile model
model.compile(loss='categorical_crossentropy',

              optimizer=Adam(lr=0.001),

              metrics=['accuracy'])
epoch = 50

batch_size = 32
history = model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),

                              epochs= epoch,validation_data=(x_val,y_val),

                              steps_per_epoch=x_train.shape[0] // batch_size

                              )
print("Test Accuracy: {0:.2f}%".format(model.evaluate(x_test,y_test)[1]*100)) #get score acording to test datas
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
Y_pred = model.predict(x_val)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(y_val,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
predictions = model.predict_classes(x_test)

predictions
Y_pred = model.predict(x_test)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(y_test,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()