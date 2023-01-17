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
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.activations import relu, softmax
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

sample = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
train_data.iloc[1,:]

Xdata = train_data.iloc[:,1:]
XTdata = test_data
XTdata.shape
img_size = 28
num_classes = 10
num_imgs = Xdata.shape[0]
test_num_imgs = XTdata.shape[0]


print("Number of train images: ", num_imgs)
print("Number of test images: ", test_num_imgs)
X = Xdata.to_numpy()
X=X.reshape(num_imgs,img_size,img_size,1) / 255.0

TX = XTdata.to_numpy()
TX=TX.reshape(test_num_imgs,img_size,img_size,1) / 255.0

# one-hot encode
Y = Ydata.to_numpy()
Y = keras.utils.to_categorical(Y, num_classes) 
from matplotlib import pyplot as plt
imshow = X[0].reshape(28,28)
plt.imshow(imshow, cmap='gray')
plt.show()
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# use GPU
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    Smodel = Sequential()

    Smodel.add(Conv2D(filters=100, kernel_size=3, input_shape=(img_size, img_size, 1), activation=relu, padding='same')) # 'same' for not ignoring image border data
    Smodel.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
    
    Smodel.add(Conv2D(filters=70, kernel_size=3, activation=relu, padding='same'))
    Smodel.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='valid'))
    
    Smodel.add(Conv2D(filters=50, kernel_size=3, activation=relu))
    
    Smodel.add(Flatten())
    Smodel.add(Dense(30, activation=relu))
    
    Smodel.add(Dropout(rate=0.5))
    Smodel.add(Dense(20, activation=relu))
    Smodel.add(Dense(num_classes, activation=softmax))
    
    Smodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

Smodel.summary()
        
Smodel.fit(X, Y,
         batch_size=100,
         epochs=10,
         validation_split=0.3)
preds = Smodel.predict(TX, batch_size=100)
labels = range(10)
pred_labels = np.array([np.argmax(p) for p in preds]).reshape(test_num_imgs, 1).astype('int')
ID = np.array(range(1,test_num_imgs+1)).reshape(test_num_imgs, 1).astype('int')

pred_labels.shape, ID.shape
sname = 'submission.csv'
np.savetxt(sname, write2csv, delimiter=',', fmt='%d', header='ImageId,Label', comments='')
# remove spaces, for submission format
fi = open(sname, 'r')
contents = fi.read()
contents = contents.replace(' ', '')
fi.close()

with open(sname, 'w') as fi:
    fi.seek(0)
    fi.truncate()
    fi.write(contents)
# A = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
A = pd.read_csv('submission.csv')
A
# see if some are true 

imshow = TX[27999].reshape(28,28)
plt.imshow(imshow, cmap='gray')
plt.show()
with strategy.scope():
    Dmodel = Sequential()
    Dmodel.add(Flatten())
    Dmodel.add(Dense(num_imgs))
    Dmodel.add(Dense(100, activation=relu))
    Dmodel.add(Dense(50, activation=relu))
    Dmodel.add(Dense(num_classes, activation=softmax))
    
    Dmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])        
Dmodel.fit(X, Y,
         batch_size=100,
         epochs=5,
         validation_split=0.3)
