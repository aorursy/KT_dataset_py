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
df = pd.read_csv('../input/data.csv')
df.head()
df = df.sample(frac=1).reset_index(drop=True)
df.head()
train_df = df[0:80000].copy()
test_df = df[80000:].copy()
train_df.shape,test_df.shape
label = train_df['character'].values
y_train = np.zeros([train_df.shape[0],df['character'].unique().shape[0]])
from sklearn.preprocessing import LabelBinarizer
binencoder = LabelBinarizer()
y_train = binencoder.fit_transform(label)
train_df = train_df.drop(['character'],axis=1)
X_train = train_df.as_matrix()
X_train = np.reshape(X_train,(X_train.shape[0],32,32,1))
import matplotlib.pyplot as plt
plt.subplot(131)
plt.title(str(label[0]))
plt.imshow(X_train[0].reshape((32,32)),cmap='gray')

plt.subplot(132)
plt.title(str(label[5]))
plt.imshow(X_train[5].reshape((32,32)),cmap='gray')

plt.subplot(133)
plt.title(str(label[10]))
plt.imshow(X_train[10].reshape((32,32)),cmap='gray')
plt.show()
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
def alpha_model(input_shape):
    X_in = Input(input_shape)
    
    X = Conv2D(16,kernel_size=(5,5),padding='same',input_shape=input_shape)(X_in)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)
    
    X = Conv2D(16,kernel_size=(5,5),padding='same',input_shape=input_shape)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)
    
    X = Flatten()(X)
    X = Dense(128,activation='relu')(X)
    X = Dense(46,activation='softmax')(X)
    
    model = Model(inputs=X_in,outputs=X,name='devanagari')
    return model
model = alpha_model((32,32,1))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=128,epochs=5)
model.save('my_model.h5')
label_test = test_df['character'].values
y_test = np.zeros([train_df.shape[0],df['character'].unique().shape[0]])
binencoder = LabelBinarizer()
y_test = binencoder.fit_transform(label_test)
test_df = test_df.drop(['character'],axis=1)
X_test = test_df.as_matrix()
X_test = np.reshape(X_test,(X_test.shape[0],32,32,1))

model.evaluate(X_test,y_test)
