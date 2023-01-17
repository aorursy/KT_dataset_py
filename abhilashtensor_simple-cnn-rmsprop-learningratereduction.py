# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from IPython.display import display, Image

import os

import PIL.Image

import cv2

import matplotlib.pyplot as plt
dig_mnist = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")

print(dig_mnist.shape)

dig_mnist.head()
test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

print(test.shape)

test.head()
train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

print(train.shape)

train.head()
ss = pd.read_csv("/kaggle/input/Kannada-MNIST/sample_submission.csv")

print(ss.shape)

ss.head()
lbl = train['label'].tolist()

print(lbl[:15])
train.drop(['label'],axis=1,inplace=True)

tr = train.as_matrix(columns=None)

print(tr.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(tr, lbl, test_size=0.20, random_state=42)
from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

prediction = clf.predict(X_test)

accscore = accuracy_score(y_test,prediction)

print(accscore)
tr = tr.reshape((tr.shape[0],28,28))
%matplotlib inline

n = np.random.randint(0,tr.shape[0])

image = tr[n]

imgplot = plt.imshow(image,cmap=plt.get_cmap('gray'))
from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import np_utils

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
nb_classes=10

y = np_utils.to_categorical(lbl, nb_classes).astype(np.float32)

class_totals = y.sum(axis=0)

class_weight = class_totals.max() / class_totals

y.shape
X=tr.reshape(tr.shape[0],28,28,1)

X.shape
# An observation code for our dataset

datagen_try = ImageDataGenerator(rotation_range=15,

                             width_shift_range = 0.25,

                             height_shift_range = 0.25,

                             shear_range = 25,

                             zoom_range = 0.4,)

# fit parameters from data

datagen_try.fit(X)

# configure batch size and retrieve one batch of images

for x_batch, y_batch in datagen_try.flow(X, y, batch_size=9):

    # create a grid of 3x3 images

    for i in range(0, 9):

        plt.subplot(330 + 1 + i)

        plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))

    # show the plot

    plt.show()

    break
# construct the actual Python generator

print("[INFO] generating images...")

total = X.shape[0]

print(total,"is total")

finalX,finalY = [],[]

i = 0

for x_batch, y_batch in datagen_try.flow(X, y, batch_size=1):

    finalX.append(x_batch)

    finalY.append(y_batch)

    i+=1

    if i==total:

        break

    if i%10000==0 and i>999:

        print(i,"/60000 done")
finalX = np.array(finalX)

finalY = np.array(finalY)
finalX = finalX.reshape((60000, 28, 28, 1))
finalY = finalY.reshape((60000, 10))
# from __future__ import print_function

# from keras.utils import np_utils

# import keras

# from keras.models import Sequential

# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

# from keras.optimizers import RMSprop

# from keras.preprocessing.image import ImageDataGenerator

# from keras.callbacks import ReduceLROnPlateau
# nb_classes=10

# y = np_utils.to_categorical(lbl, nb_classes).astype(np.float32)

# class_totals = y.sum(axis=0)

# class_weight = class_totals.max() / class_totals

# y.shape
# X=tr.reshape(tr.shape[0],28,28,1)

# X.shape
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = X.shape[1:]))

model.add(Conv2D(filters = 32, kernel_size = (4,4),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.3))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.3))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.3))



model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

model.summary()
# validation_split = 0.01

# model.fit(X, y, batch_size=1000, class_weight=class_weight, epochs=30, verbose=1, validation_split=validation_split)
validation_split = 0.01

model.fit(finalX, finalY, batch_size=1000, class_weight=class_weight, epochs=80, verbose=1, validation_split=validation_split)
open('model.json', 'w').write(model.to_json())

model.save_weights('weights.h5')

print("Model Saved")
%matplotlib inline

from matplotlib import pyplot as plt

plt.plot(model.model.history.history['loss'],'red')

plt.plot(model.model.history.history['accuracy'],'blue')

plt.plot(model.model.history.history['val_loss'],'yellow')

plt.plot(model.model.history.history['val_accuracy'],'black')

plt.show()
print(model.model.history.history)
test.drop(['id'],axis=1,inplace=True)

Y = test.as_matrix(columns=None)

Y.shape
Y=Y.reshape(test.shape[0],28,28,1)

Y.shape
predict = model.predict(Y)
predict.shape
results = np.argmax(predict,axis = 1)

results
Y=Y.reshape(test.shape[0],28,28)
fig, axs = plt.subplots(4, 4,figsize=(15,15))



num_ims = 4

a,b = 4,4

for j in range(a):

    for k in range(b):

        ids  = np.random.randint(0,5000)

        prob = results[ids]

        image = Y[ids] 

        axs[j,k].imshow(image)

        axs[j,k].set_title("Predicted label :"+str(prob))



    

plt.axis('off')





# Hide x labels and tick labels for top plots and y ticks for right plots.

for ax in axs.flat:

    ax.label_outer()
ss.head()
n = ss.columns[1]
ss.drop(n,axis=1,inplace=True)
ss.head()
ss[n] = results
ss.head()
ss.to_csv('submission.csv',index=False)