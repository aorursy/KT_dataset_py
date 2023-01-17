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
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from sklearn.metrics import confusion_matrix, accuracy_score
from plotly.offline import iplot, init_notebook_mode
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
import plotly.graph_objs as go
from matplotlib.pyplot import cm
from keras.models import Model
import numpy as np
import keras
import h5py

init_notebook_mode(connected=True)
%matplotlib inline
with h5py.File('../input/full_dataset_vectors.h5','r') as dataset:
    x_train = dataset["X_train"][:]
    x_test = dataset["X_test"][:]
    y_train = dataset["y_train"][:]
    y_test = dataset["y_test"][:]
print("x_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("x_test shape", x_test.shape)
print("y_test shape", y_test.shape)
with h5py.File("../input/train_point_clouds.h5", "r") as points_dataset:
    digits = []
    for i in range(10):
        digit = (points_dataset[str(i)]["img"][:], 
                 points_dataset[str(i)]["points"][:], 
                 points_dataset[str(i)].attrs["label"])
        digits.append(digit)
x_c = [r[0] for r in digits[0][1]]
y_c = [r[1] for r in digits[0][1]]
z_c = [r[2] for r in digits[0][1]]

trace = go.Scatter3d(x=x_c, y=y_c, z=z_c, mode='markers', 
                      marker=dict(size=12, color=z_c, colorscale='Viridis', opacity=0.7))
data = [trace]
layout = go.Layout(height=500, width=600, title= "Digit: "+str(digits[0][2]) + " in 3D space")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
#Initialising the channel dimension in the input dataset
xtrain = np.ndarray((x_train.shape[0], 4096, 3))
xtest = np.ndarray((x_test.shape[0], 4096, 3))

def add_rgb_dimen(array):
    scalar_map = cm.ScalarMappable(cmap="Oranges")
    array = scalar_map.to_rgba(array)[:,:-1]
    return array

for i in range(x_train.shape[0]):
    xtrain[i] = add_rgb_dimen(x_train[i])

for i in range(x_test.shape[0]):
    xtest[i] = add_rgb_dimen(x_test[i])
#convert to 4D space
xtrain = xtrain.reshape(x_train.shape[0], 16, 16, 16, 3)
xtest = xtest.reshape(x_test.shape[0], 16, 16, 16, 3)

#one hot for target variable
ytrain = keras.utils.to_categorical(y_train, 10)
ytest = keras.utils.to_categorical(y_test, 10)
print("xtrain shape = ", xtrain.shape)
print("ytrain shape = ", ytrain.shape)
#input layer
input_layer = Input((16,16,16,3))

#convolution layer
conv_layer_one = Conv3D(filters=8, kernel_size=(3,3,3), activation='relu')(input_layer)
conv_layer_two = Conv3D(filters=16, kernel_size=(3,3,3), activation='relu')(conv_layer_one)

#pooling layer
pooling_layer_one = MaxPool3D(pool_size=(2,2,2))(conv_layer_two)

#convolution layer
conv_layer_three = Conv3D(filters=24, kernel_size=(3,3,3), activation='relu')(pooling_layer_one)
conv_layer_four = Conv3D(filters=32, kernel_size=(3,3,3), activation='relu')(conv_layer_three)

#pooling layer
pooling_layer_two = MaxPool3D(pool_size=(2,2,2))(conv_layer_four)
flatten_layer = Flatten()(pooling_layer_two)

#Fully Connected layers
dense_layer_one = Dense(units=2048, activation='relu')(flatten_layer)
dense_layer_one = Dropout(0.4)(dense_layer_one)
dense_layer_two = Dense(units=512, activation='relu')(dense_layer_one)
dense_layer_two = Dropout(0.4)(dense_layer_two)
output_layer = Dense(units=10, activation='softmax')(dense_layer_two)

model = Model(inputs = input_layer, outputs = output_layer)
model.compile(loss=categorical_crossentropy, optimizer=Adadelta(lr=0.1), metrics=['acc'])
#model.fit(x=xtrain, y=ytrain,batch_size=128, epochs=50, validation_split=0.2)
model.fit(x=xtrain, y=ytrain,batch_size=128, epochs=2, validation_split=0.2)
