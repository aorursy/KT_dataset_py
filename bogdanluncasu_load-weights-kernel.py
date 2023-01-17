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
from keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout

from keras.layers.merge import add

from keras.activations import relu, softmax

from keras.models import Model

from keras import regularizers

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import keras
def conv3x3(filters, x= None, stride=1):

    if x is None:

        return Conv2D(filters=filters, kernel_size=3, strides=stride,

                         padding='same', bias=False)

    """3x3 convolution with padding"""

    return Conv2D(filters=filters, kernel_size=3, strides=stride,

                     padding='same', bias=False)(x)



def block(n_output, upscale=False):

    # n_output: number of feature maps in the block

    # upscale: should we use the 1x1 conv2d mapping for shortcut or not

    

    # keras functional api: return the function of type

    # Tensor -> Tensor

    def f(x):

        

        # H_l(x):

        # first pre-activation

        h = conv3x3(n_output,x)

        h = BatchNormalization()(x)

        h = Activation(relu)(h)

        # first convolution

        

        h = conv3x3(n_output)(h)

        

        # second pre-activation

        h = BatchNormalization()(x)

        h = Activation(relu)(h)

        # second convolution

        h = conv3x3(n_output)(h)

        s = add([x, h])

        return Activation(relu)(s)

    

    return f
from keras.layers import MaxPooling2D



def layer(filters,x, strides=1, blocks=2):

    

    for i in range(blocks):

        x=block(filters)(x)

    return BatchNormalization()(x)

# input tensor is the 28x28 grayscale image

input_tensor = Input((28, 28, 1))



# first conv2d with post-activation to transform the input data to some reasonable form

x = Conv2D(kernel_size=7, filters=64, strides=1)(input_tensor)

x = BatchNormalization()(x)

x = Activation(relu)(x)

x = MaxPooling2D()(x)



x = layer(64,x)

x = Conv2D(kernel_size=3, filters=128, strides=1)(x)

x = layer(128,x)

x = Conv2D(kernel_size=3, filters=256, strides=1)(x)

x = layer(256,x)

x = Conv2D(kernel_size=3, filters=512, strides=1)(x)

x = layer(512,x)





# last activation of the entire network's output

x = BatchNormalization()(x)

x = Activation(relu)(x)



# average pooling across the channels

# 28x28x48 -> 1x48

x = GlobalAveragePooling2D()(x)



# dropout for more robust learning

x = Dropout(0.2)(x)



# last softmax layer

x = Dense(units=10)(x)

x = Activation(softmax)(x)



model = Model(inputs=input_tensor, outputs=x)

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001), metrics=['accuracy'])
train_images = np.load(os.path.join('../input/cursive-hiragana-classification','train-imgs.npz'))['arr_0']

test_images = np.load(os.path.join('../input/cursive-hiragana-classification','test-imgs.npz'))['arr_0']

train_labels = np.load(os.path.join('../input/cursive-hiragana-classification','train-labels.npz'))['arr_0']

def data_preprocessing(images):

    num_images = images.shape[0]

    x_shaped_array = images.reshape(num_images, 28, 28, 1)

    out_x = x_shaped_array / np.std(x_shaped_array, axis = 0)

    return out_x

X = data_preprocessing(train_images)

y = keras.utils.to_categorical(train_labels, 10)

X_test = data_preprocessing(test_images)



X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
model.load_weights('../input/weights472019/weights.best_r7.keras')
predicted_classes = model.predict(X_test)

submission = pd.read_csv(os.path.join("../input/cursive-hiragana-classification","sample_submission.csv"))

submission['Class'] = np.argmax(predicted_classes, axis=1)

submission.to_csv(os.path.join(".","submission.csv"), index=False)



new_cols = ["p0","p1","p2","p3","p4","p5","p6","p7","p8","p9"]

new_vals = predicted_classes

submission = submission.reindex(columns=submission.columns.tolist() + new_cols)

submission[new_cols] = new_vals



submission.to_csv(os.path.join(".","submission_arr.csv"), index=False)