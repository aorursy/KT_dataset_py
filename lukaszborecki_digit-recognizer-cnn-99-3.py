# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/digit-recognizer/train.csv')
X = df.iloc[:, 1:]
y = df.iloc[:, 0]
X/=255
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.015, stratify=y)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = X_train.shape[1:]
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model

X_input = Input(input_shape)

X = Conv2D(16, (5, 5), strides = (1, 1), name = 'conv0')(X_input)
X = BatchNormalization(axis = -1, name = 'bn1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((2, 2), name='max_pool')(X)
X = Conv2D(32, (5, 5), strides = (1, 1), name = 'conv1')(X)
X = BatchNormalization(axis = -1, name = 'bn2')(X)
X = Activation('relu')(X)
X = MaxPooling2D((2, 2), name='max_pool2')(X)
X = Flatten()(X)

X = Dense(256, name = 'lay3')(X)
X = BatchNormalization(axis = -1, name = 'bn3')(X)
X = Activation('relu')(X)
X = Dropout(0.2)(X)
X = Dense(128, name = 'lay4')(X)
X = BatchNormalization(axis = -1, name = 'bn4')(X)
X = Activation('relu')(X)
X = Dropout(0.2)(X)
X = Dense(10, activation='sigmoid', name='output')(X)
model = Model(inputs = X_input, outputs = X, name='MnistDigit')
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(x=X_train, y = y_train, validation_data=(X_test,y_test), epochs = 450, batch_size=128)
df_test = pd.read_csv('../input/digit-recognizer/test.csv')
X_pred = np.array(df_test)
X_pred = X_pred/255.0
X_pred = X_pred.reshape((X_pred.shape[0], 28, 28, 1))
y_pred = model.predict(X_pred)
y_pred = np.argmax(y_pred, axis=1)
y_pred[:5]
result = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
result['Label'] = y_pred
result.head()
result.to_csv('result.csv', index=False)