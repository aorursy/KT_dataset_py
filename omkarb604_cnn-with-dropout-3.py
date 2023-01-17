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
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout

from skimage import io, measure, exposure

from sklearn.model_selection import train_test_split, KFold

from sklearn.preprocessing import MinMaxScaler

import random

import numpy as np

import matplotlib.pyplot as plt

import os

import keras

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras.callbacks import TensorBoard

from sklearn.metrics import roc_curve, auc

from tensorflow.keras.optimizers import Adam
A_dir = '../input/a/A'

V_dir = '../input/v/V'

images_A = [img for img in os.listdir(A_dir) if img.endswith(".tif")]

images_V = [img for img in os.listdir(V_dir) if img.endswith(".tif")]

imgs = images_A[:900] + images_V[:900]  

random.shuffle(imgs)
X = [] # images

Y = [] # labels    

for image in imgs:

    if '112A' in image:

        img = io.imread(os.path.join(A_dir, image),as_gray=True) #Read the image

        scaler = MinMaxScaler(copy=True)

        scaler.fit(img)

        scaled_img = scaler.transform(img) # normalizing the image

        equalized_hist = exposure.equalize_hist(scaled_img)

        X.append(equalized_hist)   

    else:

        img = io.imread(os.path.join(V_dir, image),as_gray=True)

        scaler = MinMaxScaler(copy=True)

        scaler.fit(img)

        scaled_img = scaler.transform(img) # normalizing the image

        equalized_hist = exposure.equalize_hist(scaled_img) #Histogram equalisation

        X.append(equalized_hist)

        #get the labels

    if '112A' in image:

        Y.append(1)

    elif '112V' in image:

        Y.append(0)

        
X=np.array(X)
IMG_SIZE = 50

n_box=[]

for img in X:

#     slice_1 = img[:50,:50]

#     slice_2 = img[50:100,50:100]

#     slice_3 = img[100:150,100:150]

#     slice_4 = img[150:200,150:200]

#     n_box.append(slice_1)

#     n_box.append(slice_2)

#     n_box.append(slice_3)

#     n_box.append(slice_4)

    x = np.random.randint(len(img[0])-IMG_SIZE+1,size=9)

    y = np.random.randint(len(img[1])-IMG_SIZE+1,size=9)

    for x,y in zip(x,y):

        box = img[x:x+IMG_SIZE,y:y+IMG_SIZE]

        n_box.append(box)

#     n_box.append(slice_1)

#     n_box.append(slice_2)

#     n_box.append(slice_3)

#     n_box.append(slice_4)
n_box=np.array(n_box)
n_box.shape
n_box=np.array(n_box).reshape(n_box.shape[0],50,50,1)


Y_new=[]

for i in Y:

    print(i)

    for index in range(0,9):

        Y_new.append(i)   

#        print(i)
Y_new=np.array(Y_new)

cv=KFold(n_splits=3, random_state=10, shuffle=True)



for train,test in cv.split(n_box):

    X_train,X_test = n_box[train],n_box[test]

    Y_train,Y_test = Y_new[train],Y_new[test] 



    s=Adam(lr=0.001)



    model=Sequential()



    model.add(Conv2D(90,(3,3),strides=(1,1), input_shape =(50,50,1)))

    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.3))



    # model.add(Conv2D(64,(3,3),strides=(1,1)))

    # model.add(Activation("relu"))

    # model.add(MaxPooling2D(pool_size=(2,2)))

    # model.add(Dropout(0.4))



    model.add(Conv2D(180,(3,3)))

    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2,2)))





    # model.add(Conv2D(128,(3,3)))

    # model.add(Activation("relu"))



    model.add(Conv2D(90,(3,3)))

    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.3))



    model.add(Flatten())



    model.add(Dense(1))

    model.add(Activation('sigmoid'))



    tensorboard = TensorBoard("logs")



    model.compile(loss="binary_crossentropy",optimizer= s ,metrics=['accuracy'])



    Reduce=ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=2,verbose=1,min_lr=0)



    history=model.fit(n_box, Y_new, batch_size=200,epochs=50,validation_split=0.1,callbacks=[Reduce,tensorboard])

        

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()
# %load_ext tensorboard.notebook

# %tensorboard --logdir logs

model.summary()
imgs_test = images_A[900:] + images_V[900:]  

random.shuffle(imgs_test)

X_test = [] # images

Y_test = [] # labels    

for image in imgs_test:

    if '112A' in image:

        img = io.imread(os.path.join(A_dir, image),as_gray=True) #Read the image

        scaler = MinMaxScaler(copy=True)

        scaler.fit(img)

        scaled_img = scaler.transform(img) # normalizing the image

        equalized_hist = exposure.equalize_hist(scaled_img)

        X_test.append(equalized_hist)   

    else:

        img = io.imread(os.path.join(V_dir, image),as_gray=True)

        scaler = MinMaxScaler(copy=True)

        scaler.fit(img)

        scaled_img = scaler.transform(img) # normalizing the image

        equalized_hist = exposure.equalize_hist(scaled_img) #Histogram equalisation

        X_test.append(equalized_hist)

        #get the labels

    if '112A' in image:

        Y_test.append(1)

    elif '112V' in image:

        Y_test.append(0)
X_test=np.array(X_test)

print(X_test.shape)
IMG_SIZE = 50

n_test=[]

for img in X_test:

    x = np.random.randint(len(img[0])-IMG_SIZE+1)

    y = np.random.randint(len(img[1])-IMG_SIZE+1)

    box=img[x:x+50,y:y+50]

    n_test.append(box)
n_test=np.array(n_test).reshape(202,50,50,1)
y_score=model.predict(n_test)

fpr, tpr, thresholds = roc_curve(Y_test,y_score)

plt.plot(fpr, tpr,alpha=1)

plt.plot([0,1],[0,1])

plt.xlabel("False Positive  Response ")

plt.ylabel("True Positive Response")

plt.title("ROC Plot")

plt.grid()
model.summary()