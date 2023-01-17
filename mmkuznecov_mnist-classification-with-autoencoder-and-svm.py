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
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

train.shape, test.shape # check dimensions
x_train = np.array(train.drop('label', axis = 1)).reshape(-1,28,28,1)

y_train = np.array(train['label'])



x_test = np.array(test).reshape(-1,28,28,1)
x_train = x_train.astype('float32') / 255.0

x_test = x_test.astype('float32') / 255.0
%matplotlib inline

import matplotlib.pyplot as plt

image = x_train[0].reshape((28,28))

plt.imshow(image)

plt.gray()
# importing keras

from keras import backend as K

from keras.models import Sequential, Model

from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Input, Reshape, Conv2DTranspose, BatchNormalization
# building autoencoder and encoder

inp = Input((28, 28,1))

e = Conv2D(32, (3, 3), activation='relu')(inp)

e = MaxPooling2D((2, 2))(e)

e = Conv2D(64, (3, 3), activation='relu')(e)

e = MaxPooling2D((2, 2))(e)

e = Conv2D(64, (3, 3), activation='relu')(e)

l = Flatten()(e)

latent = Dense(49, activation='softmax')(l) #



d = Reshape((7,7,1))(latent)

d = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same')(d)

d = BatchNormalization()(d)

d = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same')(d)

d = BatchNormalization()(d)

d = Conv2DTranspose(32,(3, 3), activation='relu', padding='same')(d)

decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d)





encoder = Model(inp, latent)

autoencoder = Model(inp, decoded)

autoencoder.summary()
autoencoder.compile(optimizer = 'adam', loss = 'mse')
autoencoder.fit(x_train, x_train,

                epochs=5,

                batch_size=128,

                shuffle=True,

                validation_data=(x_test, x_test))
image = x_test[3].reshape((28,28))

plt.imshow(image)

plt.gray()
decoded_imgs = autoencoder.predict(x_test)
test_img = decoded_imgs[3].reshape((28,28))

plt.imshow(test_img)

plt.gray()
train_vectors = encoder.predict(x_train)

test_vectors = encoder.predict(x_test)
from sklearn.svm import SVC

clf = SVC()

clf.fit(train_vectors, y_train)
pred = clf.predict(test_vectors)

pred = pd.Series(pred,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pred],axis = 1)

submission.head()
submission.to_csv('submission.csv', index = False)