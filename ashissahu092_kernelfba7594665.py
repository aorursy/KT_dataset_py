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
import numpy as np

import pandas as pd

import scipy.io as sc

import glob



import matplotlib.pylab as plt

import matplotlib.cm as cm

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



from keras.models import Sequential, Model

from keras.optimizers import SGD, RMSprop, Adam, Nadam

from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import to_categorical



from keras.layers import Dense, Dropout, LSTM

from keras.layers import Activation, Flatten, Input, BatchNormalization

from keras.layers import Conv1D, MaxPooling1D 

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
mat = sc.loadmat('../input/train_32x32.mat')

test = sc.loadmat('../input/test_32x32.mat')

extra = sc.loadmat('../input/extra_32x32.mat')
print(type(mat["X"]))

mat["X"]
m=mat['X'][:,6:26,:,44]

from matplotlib import pyplot as PLT

PLT.imshow(m)

PLT.show()

mat["y"][44]

len(mat["y"])

len(mat["y"])

import cv2

dim = (32,32)

m=mat['X'][:,6:26,:,1]

resized = cv2.resize(m, dim, interpolation = cv2.INTER_AREA) 



import numpy as np

import matplotlib.pyplot as plt



from skimage import data

from skimage.color import rgb2gray



original = resized

grayscale = rgb2gray(original)



fig, axes = plt.subplots(1, 2, figsize=(8, 4))

ax = axes.ravel()



ax[0].imshow(original)

ax[0].set_title("Original")

ax[1].imshow(grayscale, cmap=plt.cm.gray)

ax[1].set_title("Grayscale")



fig.tight_layout()

plt.show()
mat['X'].shape[:]
b=grayscale.ravel()

train= pd.DataFrame(b.reshape(1,1024), columns=np.arange(0,1024))

for i in range(1,73257,1):

    m=mat['X'][:,6:26,:,i]

    m = cv2.resize(m, dim, interpolation = cv2.INTER_AREA) 

    import numpy as np

    import matplotlib.pyplot as plt

    from skimage import data

    from skimage.color import rgb2gray

    original = m

    grayscale = rgb2gray(original)

    b=grayscale.ravel()

    b= pd.DataFrame(b.reshape(1,1024), columns=np.arange(0,1024))

    train = train.append(b)

x_train=train

train_labels = mat['y'][:73257,]



len(x_train)
test['X'].shape[:]
original =test['X'][:,6:26,:,1]

original = cv2.resize(original, dim, interpolation = cv2.INTER_AREA) 

grayscale = rgb2gray(original)



original1 = extra['X'][:,6:26,:,1]

original1 = cv2.resize(original1, dim, interpolation = cv2.INTER_AREA) 

grayscale1 = rgb2gray(original1)



b=grayscale.ravel()

b1=grayscale1.ravel()

x_test= pd.DataFrame(b.reshape(1,1024), columns=np.arange(0,1024))

x_valid= pd.DataFrame(b1.reshape(1,1024), columns=np.arange(0,1024))





for i in range(1,26032,1):

    m=test['X'][:,6:26,:,i]

    m = cv2.resize(m, dim, interpolation = cv2.INTER_AREA) 

    m1=extra['X'][:,6:26,:,i]

    m1 = cv2.resize(m1, dim, interpolation = cv2.INTER_AREA) 



    from skimage import data

    from skimage.color import rgb2gray

    original = m

    original1 = m1

    grayscale = rgb2gray(original)

    grayscale1 = rgb2gray(original1)

    b=grayscale.ravel()

    b1=grayscale1.ravel()

    b= pd.DataFrame(b.reshape(1,1024))

    x_test = x_test.append(b)

    b1= pd.DataFrame(b1.reshape(1,1024))

    x_valid = x_valid.append(b)

    

len(x_valid)
len(x_test)

len(x_valid)

test_labels = test['y'][:26032,]

extra_labels = extra['y'][:26032,]

x_test.shape[:]
np.empty([len(train_labels), n, 11])
len(extra_labels)
def digit_to_categorical(data):

    n = data.shape[1]

    data_cat = np.empty([len(data), n, 11])    

    for i in range(n):

        data_cat[:, i] = to_categorical(data[:, i], num_classes=11)        

    return data_cat
print(type(test_labels))


train_images = x_train.as_matrix().astype('float32')



test_images = x_test.as_matrix().astype('float32')



extra_images = x_valid.as_matrix().astype('float32')





x_train = np.concatenate((train_images.reshape(-1, 32, 32, 1),

                             test_images.reshape(-1, 32, 32, 1)),

                            axis=0)

y_train = np.concatenate((digit_to_categorical(train_labels),

                             digit_to_categorical(test_labels)),

                            axis=0)



x_valid = extra_images.reshape(-1, 32, 32, 1)

y_valid = digit_to_categorical(extra_labels)



n = int(len(x_valid)/2)

x_test, y_test = x_valid[:n], y_valid[:n]

x_valid, y_valid = x_valid[n:], y_valid[n:]



x_train.shape, x_test.shape, x_valid.shape, 

y_train.shape, y_test.shape, y_valid.shape
y_train
def cnn_model():    

    model_input = Input(shape=(32, 32, 1))

    x = BatchNormalization()(model_input)

        

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(model_input)

    x = MaxPooling2D(pool_size=(2, 2))(x) 

    

    x = Conv2D(32, (3, 3), activation='relu')(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)    

    x = Dropout(0.25)(x)

    

    x = Conv2D(64, (3, 3), activation='relu')(x)       

    x = Conv2D(64, (3, 3), activation='relu')(x)    

    x = Dropout(0.25)(x)

    

    x = Conv2D(196, (3, 3), activation='relu')(x)    

    x = Dropout(0.25)(x)

              

    x = Flatten()(x)

    

    x = Dense(512, activation='relu')(x)    

    x = Dropout(0.5)(x)

    

    y1 = Dense(11, activation='softmax')(x)

    

    model = Model(input=model_input, output=y1)



    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

cnn_model = cnn_model()

cnn_checkpointer = ModelCheckpoint(filepath='weights.best.cnn.hdf5', 

                                   verbose=2, save_best_only=True)
y_train[:,0]
y_train_list = [y_train[:, i] for i in range(1)]

y_test_list = [y_test[:, i] for i in range(1)]

y_valid_list = [y_valid[:, i] for i in range(1)]

cnn_history = cnn_model.fit(x_train, y_train_list, 

                            validation_data=(x_valid, y_valid_list), 

                            epochs=75, batch_size=128, verbose=2, 

                            callbacks=[cnn_checkpointer])
cnn_scores = cnn_model.evaluate(x_test, y_test_list, verbose=0)

cnn_scores[0]
cnn_model.load_weights('weights.best.cnn.hdf5')

cnn_scores = cnn_model.evaluate(x_test, y_test_list, verbose=0)



print("CNN Model 1. \n")

print("Scores: \n" , (cnn_scores))

print("First digit. Accuracy: %.2f%%" % (cnn_scores[1]*100))

print(cnn_model.summary())