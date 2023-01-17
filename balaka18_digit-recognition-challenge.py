# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as ml

import matplotlib.pyplot as plt

%matplotlib inline

ml.style.use('ggplot')



from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from keras.utils.np_utils import to_categorical

from keras.layers import Dropout, Flatten ,BatchNormalization , MaxPool2D

from keras.layers.convolutional import Conv2D



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

train.head(10)
X = train.drop(['label'],axis=1)

y = train['label']
X = X / 255.0

test = test / 255.0



# Reshaping to 28 x 28 x 1, where 1 represents the color channel

X = X.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
fig,axes = plt.subplots(1,20,figsize=(30,10))

fig.tight_layout()



for i in range(20):

    axes[i].imshow(X[i].reshape(28,28))          # 28 x 28 matrix

    axes[i].axis('off')

    axes[i].set_title(y[i])           # Setting the title of all images to the label to see if they match

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

print("X : ", X_train.shape, X_test.shape)

print("y : ", y_train.shape, y_test.shape)
aug = ImageDataGenerator(width_shift_range=0.1,   

                            height_shift_range=0.1,

                            zoom_range=0.2,  

                            shear_range=0.1, 

                            rotation_range=10)  

aug.fit(X_train)

itr = aug.flow(X_train,y_train,batch_size=20)       # Creating an iterator from the dataset. Returns one batch of augmented images for each iteration.

X_batch,y_batch = next(itr)



# Visualizing to check

fig,axes = plt.subplots(1,10,figsize=(30,10))

fig.tight_layout()



for i in range(10):

    axes[i].imshow(X_batch[i].reshape(28,28))          # 28 x 28 matrix

    axes[i].axis('off')

    axes[i].set_title(y_batch[i])           # Setting the title of all images to the label to see if they match

plt.show()
y_train = to_categorical(y_train,10)

y_test = to_categorical(y_test,10)
cnn = Sequential()



# First

cnn.add(Conv2D(filters = 64, kernel_size = (3,3) ,activation ='relu', input_shape = (28,28,1)))

cnn.add(Conv2D(filters = 56, kernel_size = (3,3),activation ='relu'))

cnn.add(BatchNormalization())

cnn.add(MaxPool2D(pool_size=(2,2)))

cnn.add(Dropout(0.2))



# Second

cnn.add(Conv2D(filters = 64, kernel_size = (3,3),activation ='relu'))

cnn.add(Conv2D(filters = 48, kernel_size = (3,3),activation ='relu'))

cnn.add(Conv2D(filters = 32, kernel_size = (3,3),activation ='relu'))

cnn.add(BatchNormalization())

cnn.add(MaxPool2D(pool_size=(2,2)))

cnn.add(Dropout(0.2))



# Third

cnn.add(Flatten())

cnn.add(Dense(256, activation = "relu"))           # (64 x 7 x 7) x 256

cnn.add(Dense(128, activation = "relu"))           # 256 x 128

cnn.add(Dense(64, activation = "relu"))           # 128 x 64

cnn.add(Dropout(0.4))



#Output

cnn.add(Dense(10, activation = "softmax"))           # 64 x 10

cnn.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
print(cnn.summary())
status = cnn.fit_generator(aug.flow(X_train,y_train, batch_size=56),

                              epochs = 10, validation_data = (X_test,y_test),

                              verbose = 2, steps_per_epoch=660)
plt.figure()

fig,(ax1, ax2)=plt.subplots(1,2,figsize=(15,5))

ax1.plot(status.history['loss'])

ax1.plot(status.history['val_loss'])

ax1.legend(['Training','Validation'])

ax1.set_title('Loss')

ax1.set_xlabel('Epoch')



ax2.plot(status.history['accuracy'])

ax2.plot(status.history['val_accuracy'])

ax2.legend(['Training','Validation'])

ax2.set_title('Acurracy')

ax2.set_xlabel('Epoch')



plt.show()



score = cnn.evaluate(X_test,y_test,verbose = 1)

print('Test Score: ',score[0])

print('Test Accuracy: ',score[1])
res = cnn.predict(test)

res = np.argmax(res,axis = 1)

res = pd.Series(res,name="Label")

final = pd.concat([pd.Series(range(1,28001),name = "ImageId"),res],axis = 1)

final.to_csv("Balaka_Digit_Recognition2.csv",index=False)