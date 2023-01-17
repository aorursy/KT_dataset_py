# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_json('../input/train.json')

test_df = pd.read_json('../input/test.json')

train_df.head()
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])

X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])
X_band = np.zeros([1604,75,75,2])

for t in range(1604):

    X_band[t,:,:,0] = X_band_1[t]

    X_band[t,:,:,1] = X_band_2[t]
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



import keras.backend as K

K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow
def Iceberg_model(input_shape):

    X_in = Input(input_shape)

    

    X = Conv2D(10,kernel_size=(5,5),input_shape=(75,75,2))(X_in)

    X = BatchNormalization()(X)

    X = Activation('relu')(X)

    X = MaxPooling2D(pool_size=(2,2))(X)

    

    X = Conv2D(10,kernel_size=(5,5))(X)

    X = BatchNormalization()(X)

    X = Activation('relu')(X)

    X = MaxPooling2D(pool_size=(2,2))(X)

    

    X = Conv2D(10,kernel_size=(5,5))(X)

    X = BatchNormalization()(X)

    X = Activation('relu')(X)

    X = MaxPooling2D(pool_size=(2,2))(X)

    

    X = Flatten()(X)

    X = Dense(50)(X)

    X = Activation('relu')(X)

    

    X = Dense(1)(X)

    X = Activation('sigmoid')(X)

    

    model = Model(inputs=X_in,outputs=X,name='Iceberg_model')

    return model
IcebergModel = Iceberg_model((75,75,2))
IcebergModel.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
target = train_df['is_iceberg'].values

IcebergModel.fit(x=X_band,y=target,epochs=20,batch_size=128)
IcebergModel.evaluate(x=X_band,y=target)
test_df.shape
X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_1"]])

X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_2"]])

X_test = np.zeros([8424,75,75,2])

for t in range(8424):

    X_test[t,:,:,0] = X_band_test_1[t]

    X_test[t,:,:,1] = X_band_test_2[t]
pred = IcebergModel.predict(x=X_test)
sub_df = pd.DataFrame()

sub_df['id'] = test_df['id']

sub_df['is_iceberg'] = pred

sub_df.to_csv('output.csv',index=False)