import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score

from keras.models import Sequential

from keras.layers import Convolution2D,Dropout,LeakyReLU,Dense,MaxPooling2D,Flatten,BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.constraints import max_norm

from keras.utils import np_utils

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape,test.shape
train.head(3)
X = train.drop('label',axis = 1).values

y = train['label'].values

test = test.values
plt.imshow(X[10].reshape(28,-1),cmap='gray')

plt.show()

print(y[10])
scaler = MinMaxScaler()

scaler.fit(X)

X = scaler.transform(X)

test = scaler.transform(test)
X = X.reshape(-1,28,28,1)

test = test.reshape(-1,28,28,1)
y = np_utils.to_categorical(y,dtype ='int8')



y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,stratify=y,random_state=0)
# Setting up Data Augmentation

datagen = ImageDataGenerator(rotation_range=10,

                             width_shift_range=.1,

                             height_shift_range=.1,

                             shear_range=10,

                             zoom_range=.1)

datagen.fit(X_train)
model = Sequential()



model.add(Convolution2D(64,(3,3),activation='relu', input_shape=(28,28,1),padding='same'))

model.add(Dropout(0.3))

model.add(Convolution2D(64, (3,3), activation='relu',padding='same'))

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.3))

model.add(Convolution2D(128,(3,3),activation='relu',padding='same'))

model.add(Convolution2D(128, (3,3),activation='relu',padding='same'))

model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(256, activation = "relu",kernel_constraint = max_norm(3)))

model.add(Dropout(0.5))

model.add(Dense(10,activation='softmax'))



model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit_generator(datagen.flow(X_train,y_train,batch_size=128),epochs = 2,shuffle = True,validation_data=(X_test,y_test),steps_per_epoch=X_train.shape[0])
predictions = model.predict_classes(test)

my_submission = pd.DataFrame({'ImageId': range(1,len(predictions)+1), 'Label': predictions})

my_submission.to_csv('submission.csv', index=False)