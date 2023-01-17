import numpy as np 

import pandas as pd

from matplotlib import pyplot as plt

from keras.utils.np_utils import to_categorical

from keras.layers.core import Dense, Flatten, Dropout, Lambda

from keras.layers import BatchNormalization

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.optimizers import Adam, SGD

from keras.models import Sequential

import cv2

##df = open(r'E:\ml data\mnist\train.csv')

gh = pd.read_csv('../input/train.csv')

##df.close()
gh.head()
y = gh.label

del gh['label']
gh.shape
fd = gh/255

X= np.array(fd).reshape([-1,28,28, 1])

X_for_sample= np.array(fd).reshape([-1,28,28])
X.shape
plt.imshow(X_for_sample[58])

plt.show()
Y = to_categorical(y, num_classes= 10)

Y.shape
meanm = X.mean().astype(float)

stdm = X.std().astype(float)
def normalise(q):

    return  (q-meanm)/stdm
model= Sequential()

model.add(Lambda(normalise, input_shape = (28,28, 1)))

model.add(Conv2D(32, (3,3), padding= 'same', activation = 'relu'))

model.add(BatchNormalization(axis = 1))

model.add(Conv2D(32, (3,3), padding= 'same', activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization(axis = 1))



model.add(Conv2D(64, (3,3), padding= 'same', activation = 'relu'))

model.add(BatchNormalization(axis = 1))

model.add(Conv2D(64, (3,3), padding= 'same', activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(BatchNormalization(axis = 1))



model.add(Dense(512, activation = 'relu'))

model.add(BatchNormalization(axis = 1))

model.add(Dropout(0.5))

model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = Adam())
model.optimizer.lr = 0.01
model.fit(X, Y, validation_split = 0.25, verbose = 1, batch_size = 32, epochs= 7 )
tr = pd.read_csv('../input/test.csv')
bv = tr/255

Xt= np.array(bv).reshape([-1,28,28, 1])
prediction_test = model.predict(Xt)

prediction_test.shape
S = np.argmax(prediction_test ,axis = 1)



results = pd.Series(S,name="Label")
subs = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

subs.head()
subs.to_csv('my_mnist1.csv', index = False)