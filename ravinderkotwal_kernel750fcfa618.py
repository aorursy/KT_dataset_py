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
import numpy as np

images_set=np.asarray(np.load("/kaggle/input/imagekadataset/images_set3.npy").astype(np.float32))

#images_set[0]
for i in range(len(images_set)):

    images_set[i]=np.divide((images_set[i]-images_set[i].mean()),images_set[i].std())
np.save("/images_set4.npy",images_set)
#point=np.load("/kaggle/input/imagekadataset/point.npy",allow_pickle=True)

#mean_point=point.mean()

#std_point=point.std()
#point=np.divide((point-mean_point),std_point)
#images_set.reshape(-1,224,224,1)
#from sklearn.model_selection import train_test_split

#X_train,X_test=train_test_split(images_set,test_size=0.3)

#y_train,y_test=train_test_split(point,test_size=0.3)
#X_train=images_set[0:14000]

#Y_train=point[0:14000]
#X_test=images_set[14000:]

#Y_test=point[14000:]
#import gc

#del images_set

#gc.collect
#import gc

#del point

#gc.collect
#print(X_train.shape)
#import tensorflow

#from tensorflow.keras.models import Sequential

#from tensorflow.keras.layers import Flatten

#from tensorflow.keras.layers import Convolution2D

#from tensorflow.keras.layers import MaxPooling2D

#from tensorflow.keras.layers import Dense

#from tensorflow.keras.layers import Dropout



###model = Sequential()

###model.add(Convolution2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))#

##model.add(Convolution2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))

#model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

#model.add(Convolution2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

#model.add(Convolution2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

#odel.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

#odel.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

#model.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

#model.add(Convolution2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

#model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

#model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

#model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

#model.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

#model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

#odel.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

#odel.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

#odel.add(Convolution2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

#odel.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))#

#model.add(Flatten())

#model.add(Dense(4096,activation="relu"))

#model.add(Dense(4096,activation="relu"))

#3model.add(Dense(32, activation="relu"))

#model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

#model.summary()
#model.fit(X_train,Y_train,batch_size=20,epochs=50,validation_data=(X_test,Y_test))