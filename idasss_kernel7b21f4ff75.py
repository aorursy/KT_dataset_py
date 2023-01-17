# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import KFold, train_test_split
import keras

from keras.utils import to_categorical

from keras.engine.input_layer import Input

from keras.models import Model

from keras.layers import Dense, Dropout, Flatten, Activation, Reshape, Add, Concatenate

from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, GlobalAveragePooling2D, UpSampling2D, BatchNormalization

from keras.layers import PReLU, LeakyReLU

from keras.optimizers import Adam, Adadelta, Nadam, Adamax

from keras.regularizers import l1, l2, l1_l2

from keras import backend as K

from keras.callbacks import EarlyStopping
df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')

df_train.head()
train_data_data = df_train.drop("label", axis = 1).values / 255

train_data_data = np.expand_dims((train_data_data.reshape((train_data_data.shape[0], 28, 28))), axis = -1)



train_data_labels = to_categorical(df_train["label"].values, 10)



test_data_data = df_test.values / 255

test_data_data = np.expand_dims(test_data_data.reshape((test_data_data.shape[0], 28, 28)), axis = -1)
print(train_data_data.shape)

print(train_data_labels.shape)

print(test_data_data.shape)
'''

def build_model():

    input0 = Input(shape = (28, 28, 1))

    

    x = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(input0)

    x = Activation(activation = 'relu')(x)

    x = BatchNormalization()(x)

    

    x0 = Conv2D(filters = 16, kernel_size = (2, 2), strides = (2, 2), padding = 'same')(x)

    x0 = Activation(activation = 'relu')(x0)

    x1 = Conv2D(filters = 16, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x1 = Activation(activation = 'relu')(x1)

    x2 = Conv2D(filters = 16, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(x)

    x2 = Activation(activation = 'relu')(x2)

    x3 = Conv2D(filters = 16, kernel_size = (5, 5), strides = (2, 2), padding = 'same')(x)

    x3 = Activation(activation = 'relu')(x3)

    

    x = Concatenate()([x0, x1, x2, x3])

    x = BatchNormalization()(x)

    

    x0 = Conv2D(filters = 32, kernel_size = (2, 2), strides = (2, 2), padding = 'same')(x)

    x0 = Activation(activation = 'relu')(x0)

    x1 = Conv2D(filters = 32, kernel_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x1 = Activation(activation = 'relu')(x1)

    x2 = Conv2D(filters = 32, kernel_size = (4, 4), strides = (2, 2), padding = 'same')(x)

    x2 = Activation(activation = 'relu')(x2)

    x3 = Conv2D(filters = 32, kernel_size = (5, 5), strides = (2, 2), padding = 'same')(x)

    x3 = Activation(activation = 'relu')(x3)

    

    x = Concatenate()([x0, x1, x2, x3])

    x = BatchNormalization()(x)

    

    x = Conv2D(filters = 128, kernel_size = (3, 3))(x)

    x = Activation(activation = 'relu')(x0)

    

    x = Conv2D(filters = 256, kernel_size = (3, 3))(x)

    x = Activation(activation = 'relu')(x0)

    

    x = Conv2D(filters = 512, kernel_size = (3, 3))(x)

    

    x = Flatten()(x)

    x = Dense(units = 512)(x)

    x = Activation(activation = 'relu')(x)

    x = Dense(units = 128)(x)

    x = Activation(activation = 'relu')(x)

    x = Dense(units = 10)(x)

    x = Activation(activation = 'softmax')(x)

    model = Model(inputs = input0, outputs = x)

    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['acc'])

    return model

'''
'''0923 model

def build_model():

    input0 = Input(shape = (28, 28, 1))

    x = Conv2D(filters = 32, kernel_size = (5, 5))(input0)

    x = PReLU()(x)

    x = Conv2D(filters = 64, kernel_size = (3, 3))(x)

    x = PReLU()(x)

    x = MaxPooling2D(pool_size = (2, 2))(x)

    

    x = Conv2D(filters = 128, kernel_size = (3, 3))(x)

    x = PReLU()(x)

    x = MaxPooling2D(pool_size = (2, 2))(x)

    x = Dropout(rate = 0.25)(x)

    

    x = Conv2D(filters = 256, kernel_size = (3, 3))(x)

    x = PReLU()(x)

    x = MaxPooling2D(pool_size = (2, 2))(x)

    x = Dropout(rate = 0.25)(x)

    

    x = Flatten()(x)

    x = Dense(units = 512)(x)

    x = PReLU()(x)

    x = Dropout(rate = 0.25)(x)

    x = Dense(units = 128)(x)

    x = PReLU()(x)

    x = Dropout(rate = 0.25)(x)

    x = Dense(units = 10)(x)

    x = Activation(activation = 'softmax')(x)



    model = Model(inputs = input0, outputs = x)

    model.compile(loss = 'categorical_crossentropy', optimizer = Adadelta(), metrics = ['accuracy'])

    

    return model

'''
'''

def build_model():

    input0 = Input(shape = (28, 28, 1))

    x0 = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(input0)

    x1 = Conv2D(filters = 32, kernel_size = (4, 4), padding = 'same')(input0)

    x2 = Conv2D(filters = 48, kernel_size = (5, 5), padding = 'same')(input0)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    x0 = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(x)

    x1 = Conv2D(filters = 32, kernel_size = (4, 4), padding = 'same')(x)

    x2 = Conv2D(filters = 48, kernel_size = (5, 5), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    x0 = Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same')(x)

    x1 = Conv2D(filters = 64, kernel_size = (4, 4), padding = 'same')(x)

    x2 = Conv2D(filters = 96, kernel_size = (5, 5), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    x0 = Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same')(x)

    x1 = Conv2D(filters = 64, kernel_size = (4, 4), padding = 'same')(x)

    x2 = Conv2D(filters = 96, kernel_size = (5, 5), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D(pool_size = (2, 2))(x)

    

    x0 = Conv2D(filters = 64, kernel_size = (2, 2), padding = 'same')(x)

    x1 = Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same')(x)

    x2 = Conv2D(filters = 192, kernel_size = (4, 4), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x0 = Conv2D(filters = 64, kernel_size = (2, 2), padding = 'same')(x)

    x1 = Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same')(x)

    x2 = Conv2D(filters = 192, kernel_size = (4, 4), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x = MaxPooling2D(pool_size = (2, 2))(x)

    x = Dropout(rate = 0.25)(x)

    

    x0 = Conv2D(filters = 128, kernel_size = (2, 2), padding = 'same')(x)

    x1 = Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same')(x)

    x2 = Conv2D(filters = 384, kernel_size = (4, 4), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x0 = Conv2D(filters = 128, kernel_size = (2, 2), padding = 'same')(x)

    x1 = Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same')(x)

    x2 = Conv2D(filters = 384, kernel_size = (4, 4), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    #x = MaxPooling2D(pool_size = (2, 2))(x)

    #x = Dropout(rate = 0.25)(x)

    

    #x = Flatten()(x)

    #x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)

    #x = Dense(units = 1024)(x)

    #x = LeakyReLU()(x)

    x = Dense(units = 512)(x)

    x = LeakyReLU()(x)

    x = Dropout(rate = 0.25)(x)

    x = Dense(units = 256)(x)

    x = LeakyReLU()(x)

    x = Dense(units = 128)(x)

    x = LeakyReLU()(x)

    x = Dropout(rate = 0.25)(x)

    x = Dense(units = 64)(x)

    x = LeakyReLU()(x)

    x = Dropout(rate = 0.25)(x)

    x = Dense(units = 32)(x)

    x = LeakyReLU()(x)

    x = Dense(units = 10)(x)

    x = Activation(activation = 'softmax')(x)



    model = Model(inputs = input0, outputs = x)

    model.compile(loss = 'categorical_crossentropy', optimizer = Adadelta(), metrics = ['accuracy'])

    

    return model

'''
def build_model():

    input0 = Input(shape = (28, 28, 1))

    x0 = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(input0)

    x1 = Conv2D(filters = 32, kernel_size = (4, 4), padding = 'same')(input0)

    x2 = Conv2D(filters = 48, kernel_size = (5, 5), padding = 'same')(input0)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    x0 = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(x)

    x1 = Conv2D(filters = 32, kernel_size = (4, 4), padding = 'same')(x)

    x2 = Conv2D(filters = 48, kernel_size = (5, 5), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    x0 = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(x)

    x1 = Conv2D(filters = 32, kernel_size = (4, 4), padding = 'same')(x)

    x2 = Conv2D(filters = 48, kernel_size = (5, 5), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    x0 = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(x)

    x1 = Conv2D(filters = 32, kernel_size = (4, 4), padding = 'same')(x)

    x2 = Conv2D(filters = 48, kernel_size = (5, 5), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    a0 = Conv2D(filters = 96, kernel_size = (1, 1), padding = 'same')(input0)

    x = Add()([x, a0])

    x = BatchNormalization()(x)

    

    x0 = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(x)

    x1 = Conv2D(filters = 32, kernel_size = (4, 4), padding = 'same')(x)

    x2 = Conv2D(filters = 48, kernel_size = (5, 5), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    x0 = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(x)

    x1 = Conv2D(filters = 32, kernel_size = (4, 4), padding = 'same')(x)

    x2 = Conv2D(filters = 48, kernel_size = (5, 5), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    x0 = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(x)

    x1 = Conv2D(filters = 32, kernel_size = (4, 4), padding = 'same')(x)

    x2 = Conv2D(filters = 48, kernel_size = (5, 5), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    x0 = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(x)

    x1 = Conv2D(filters = 32, kernel_size = (4, 4), padding = 'same')(x)

    x2 = Conv2D(filters = 48, kernel_size = (5, 5), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    a0 = Conv2D(filters = 96, kernel_size = (1, 1), padding = 'same')(input0)

    x = Add()([x, a0])

    x = BatchNormalization()(x)

    

    x0 = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(x)

    x1 = Conv2D(filters = 32, kernel_size = (4, 4), padding = 'same')(x)

    x2 = Conv2D(filters = 48, kernel_size = (5, 5), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    x0 = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(x)

    x1 = Conv2D(filters = 32, kernel_size = (4, 4), padding = 'same')(x)

    x2 = Conv2D(filters = 48, kernel_size = (5, 5), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    x0 = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(x)

    x1 = Conv2D(filters = 32, kernel_size = (4, 4), padding = 'same')(x)

    x2 = Conv2D(filters = 48, kernel_size = (5, 5), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x = BatchNormalization()(x)

    x0 = Conv2D(filters = 16, kernel_size = (3, 3), padding = 'same')(x)

    x1 = Conv2D(filters = 32, kernel_size = (4, 4), padding = 'same')(x)

    x2 = Conv2D(filters = 48, kernel_size = (5, 5), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    a0 = Conv2D(filters = 96, kernel_size = (1, 1), padding = 'same')(input0)

    x = Add()([x, a0])

    x = BatchNormalization()(x)

    

    input1 = MaxPooling2D(pool_size = (2, 2))(x)

    

    x0 = Conv2D(filters = 32, kernel_size = (2, 2), padding = 'same')(input1)

    x1 = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same')(input1)

    x2 = Conv2D(filters = 96, kernel_size = (4, 4), padding = 'same')(input1)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x0 = Conv2D(filters = 32, kernel_size = (2, 2), padding = 'same')(x)

    x1 = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same')(x)

    x2 = Conv2D(filters = 96, kernel_size = (4, 4), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    a1 = Conv2D(filters = 192, kernel_size = (1, 1), padding = 'same')(input1)

    x = Add()([x, a1])

    x = BatchNormalization()(x)

    

    x0 = Conv2D(filters = 32, kernel_size = (2, 2), padding = 'same')(x)

    x1 = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same')(x)

    x2 = Conv2D(filters = 96, kernel_size = (4, 4), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x0 = Conv2D(filters = 32, kernel_size = (2, 2), padding = 'same')(x)

    x1 = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same')(x)

    x2 = Conv2D(filters = 96, kernel_size = (4, 4), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    a1 = Conv2D(filters = 192, kernel_size = (1, 1), padding = 'same')(input1)

    x = Add()([x, a1])

    x = BatchNormalization()(x)

    

    x0 = Conv2D(filters = 32, kernel_size = (2, 2), padding = 'same')(x)

    x1 = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same')(x)

    x2 = Conv2D(filters = 96, kernel_size = (4, 4), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x0 = Conv2D(filters = 32, kernel_size = (2, 2), padding = 'same')(x)

    x1 = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same')(x)

    x2 = Conv2D(filters = 96, kernel_size = (4, 4), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    a1 = Conv2D(filters = 192, kernel_size = (1, 1), padding = 'same')(input1)

    x = Add()([x, a1])

    x = BatchNormalization()(x)

    

    input2 = MaxPooling2D(pool_size = (2, 2))(x)

    

    x0 = Conv2D(filters = 64, kernel_size = (2, 2), padding = 'same')(input2)

    x1 = Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same')(input2)

    x2 = Conv2D(filters = 192, kernel_size = (4, 4), padding = 'same')(input2)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x0 = Conv2D(filters = 64, kernel_size = (2, 2), padding = 'same')(x)

    x1 = Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same')(x)

    x2 = Conv2D(filters = 192, kernel_size = (4, 4), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    a2 = Conv2D(filters = 384, kernel_size = (1, 1), padding = 'same')(input2)

    x = Add()([x, a2])

    x = BatchNormalization()(x)

    

    x0 = Conv2D(filters = 64, kernel_size = (2, 2), padding = 'same')(x)

    x1 = Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same')(x)

    x2 = Conv2D(filters = 192, kernel_size = (4, 4), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x0 = Conv2D(filters = 64, kernel_size = (2, 2), padding = 'same')(x)

    x1 = Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same')(x)

    x2 = Conv2D(filters = 192, kernel_size = (4, 4), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    a2 = Conv2D(filters = 384, kernel_size = (1, 1), padding = 'same')(input2)

    x = Add()([x, a2])

    x = BatchNormalization()(x)

    

    x0 = Conv2D(filters = 64, kernel_size = (2, 2), padding = 'same')(x)

    x1 = Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same')(x)

    x2 = Conv2D(filters = 192, kernel_size = (4, 4), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    x0 = Conv2D(filters = 64, kernel_size = (2, 2), padding = 'same')(x)

    x1 = Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same')(x)

    x2 = Conv2D(filters = 192, kernel_size = (4, 4), padding = 'same')(x)

    x = Concatenate()([x0, x1, x2])

    x = LeakyReLU()(x)

    #x = MaxPooling2D(pool_size = (2, 2))(x)

    #x = Dropout(rate = 0.25)(x)

    

    x = Flatten()(x)

    x = BatchNormalization()(x)

    #x = GlobalAveragePooling2D()(x)

    x = Dense(units = 1024)(x)

    x = LeakyReLU()(x)

    x = Dense(units = 512)(x)

    x = LeakyReLU()(x)

    x = Dropout(rate = 0.25)(x)

    x = Dense(units = 256)(x)

    x = LeakyReLU()(x)

    x = Dense(units = 128)(x)

    x = LeakyReLU()(x)

    x = Dropout(rate = 0.25)(x)

    x = Dense(units = 64)(x)

    x = LeakyReLU()(x)

    x = Dropout(rate = 0.25)(x)

    x = Dense(units = 32)(x)

    x = LeakyReLU()(x)

    x = Dense(units = 10)(x)

    x = Activation(activation = 'softmax')(x)



    model = Model(inputs = input0, outputs = x)

    model.compile(loss = 'categorical_crossentropy', optimizer = Adadelta(), metrics = ['accuracy'])

    

    return model
'''

X_train, X_test, Y_train, Y_test = train_test_split(train_data_data, train_data_labels, test_size = 0.1)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)

'''
'''

kf = KFold(n_splits = 10, shuffle = True)

all_loss = []

all_val_loss = []

all_acc = []

all_val_acc = []

epochs = 300



for train_index, val_index in kf.split(X_train, Y_train):

    train_data = X_train[train_index]

    train_label = Y_train[train_index]

    val_data = X_train[val_index]

    val_label = Y_train[val_index]

    

    earlystopping_callback = EarlyStopping(monitor = 'val_loss', patience=10, verbose=1, mode='auto')

    

    model = build_model()

    history = model.fit(x = train_data, y = train_label, epochs = epochs, batch_size = 256, validation_data = (val_data, val_label), callbacks = [earlystopping_callback])

    

    loss = history.history["loss"]

    val_loss = history.history["val_loss"]

    acc = history.history["acc"]

    val_acc = history.history["val_acc"]

    

    all_loss.append(loss)

    all_val_loss.append(val_loss)

    all_acc.append(acc)

    all_val_acc.append(val_acc)

    



average_all_loss = np.mean([i[-1] for i in all_loss])

average_all_val_loss = np.mean([i[-1] for i in all_val_loss])

average_all_acc = np.mean([i[-1] for i in all_acc])

average_all_val_acc = np.mean([i[-1] for i in all_val_acc])



print("Loss: {}, Val_Loss: {}, Accuracy: {}, Val_Accuracy: {}".format(average_all_loss, average_all_val_loss, average_all_acc, average_all_val_acc))

'''
'''

epochs = 300

earlystopping_callback = EarlyStopping(monitor = 'val_loss', patience=10, verbose=1, mode='auto')

    

model = build_model()

history = model.fit(x = train_data_data, y = train_data_labels, epochs = epochs, batch_size = 256, validation_split = 0.1, callbacks = [earlystopping_callback])

'''
epochs = 300

max_patience = 10

patience = 0

weights = None

model = build_model()

min_val_loss = None

max_val_acc = 0

max_val_acc_loss = None



for epoch in range(epochs):

    print("Epoch {}/{}".format(epoch + 1, epochs))

    history = model.fit(x = train_data_data, y = train_data_labels, epochs = 1, batch_size = 256, validation_split = 0.1)

    if epoch == 0:

        min_val_loss = history.history["val_loss"][0]

        max_val_acc = history.history["val_acc"][0]

        max_val_acc_loss = history.history["val_loss"][0]

        weights = model.get_weights()

    else:

        if history.history["val_loss"][0] < min_val_loss:

            patience = 0

        else:

            patience = patience + 1

        

        if history.history["val_acc"][0] > max_val_acc:

            weights = model.get_weights()

        elif history.history["val_acc"][0] == max_val_acc:

            if history.history["val_loss"][0] < max_val_acc_loss:

                weights = model.get_weights()

    

    if patience >= max_patience:

        break

        

model.set_weights(weights)
prediction = np.expand_dims(model.predict(test_data_data).argmax(axis = -1).astype(np.uint8), axis = -1)

na_imageId = np.expand_dims(np.arange(1, test_data_data.shape[0] + 1), axis = -1)
df_submission = pd.DataFrame(np.concatenate([na_imageId, prediction], axis = 1), columns = ["ImageId", "Label"])

df_submission.head()
df_submission.to_csv("submission.csv", index = False)