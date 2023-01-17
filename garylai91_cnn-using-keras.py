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
# Import data
data = pd.read_csv("../input/handwritten_data_785.csv", encoding = 'utf8')
print("DataFrame shape:" + str(data.shape))
target = data.iloc[:,0].values.reshape(-1,1)
features = data.iloc[:, 1:]
print("Target shape:" + str(target.shape))
print("Features shape:" + str(features.shape))
pd.DataFrame(target)[0].value_counts(normalize=True).sort_values(ascending=False).plot(kind='bar')
# Import pacakges
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

target2 = to_categorical(target.copy())

X_train, X_test, y_train, y_test = train_test_split(features, target2, test_size=0.3, random_state=42)

X_train = X_train.values.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.values.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

def model(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3,3))(X_input)
    X = Conv2D(32, (3,3), strides=(1,1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2), name='maxpool1')(X)
    X = Flatten()(X)
    X = Dense(26, activation='softmax', name = 'fc')(X)
    model = Model(inputs = X_input, outputs=X, name = 'ConvModel')
    return model
firstModel = model(X_train.shape[1:])
firstModel.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
firstModel.fit(X_train, y_train, epochs = 5, batch_size = 256)
firstModel.evaluate(X_test, y_test)
firstModel.summary()

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
inputs = X_train.shape[1]
def logit_model(inputs):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=inputs))
    model.add(Dense(256, activation='relu', input_dim=512))
    model.add(Dense(128, activation='relu', input_dim=256))
    model.add(Dense(26, activation='softmax', input_dim=128))
    return model
log_reg = logit_model(inputs)
log_reg.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model = log_reg.fit(X_train, y_train, epochs = 5, batch_size = 128)
test_score = log_reg.evaluate(X_test, y_test)
print('Test cost: ' + str(test_score[0]))
print('Test cost: ' + str(test_score[1]))
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(range(len(model.history['loss'])), model.history['loss'], label='Training cost')
plt.title('Training Cost')

plt.plot(range(len(model.history['acc'])), model.history['acc'], 'r', label='Training Accuracy')
plt.title('Training Accuracy')
