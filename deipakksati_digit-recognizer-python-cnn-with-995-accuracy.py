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
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
%matplotlib inline
import tensorflow as tf

from sklearn.model_selection import train_test_split
print("We're using TF", tf.__version__)

train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
print(train.shape,test.shape)
y = train['label']
train=train.drop(['label'],axis =1)
train = train/255
test = test/255

df_train ,df_val,y_train,y_val = train_test_split(train,y,test_size=0.2, random_state=111)
print(df_train.shape,df_val.shape,y_train.shape,y_val.shape)

import keras
y_train_oh = keras.utils.to_categorical(y_train, 10)
y_val_oh = keras.utils.to_categorical(y_val, 10)
print(df_train.shape,df_val.shape,y_train_oh.shape,y_val_oh.shape)
dims = df_train.shape
print(dims[0],dims)
dize2  = (dims[0],28,28,1)
df_train_fn = df_train.to_numpy()
df_train_f = df_train_fn.reshape((dize2))
df_val_fn = df_val.to_numpy()
df_val_f = df_val_fn.reshape((df_val_fn.shape[0],28,28,1))
df_test_fn = test.to_numpy()
df_test_f = df_test_fn.reshape((df_test_fn.shape[0],28,28,1))
print(df_train_f.shape, df_val_f.shape,df_test_f.shape)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, Flatten, Input, Dense

def create_model():
    
    def add_conv_block(model, num_filters):
        
        model.add(Conv2D(num_filters, 3, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(num_filters, 3, activation='relu', padding='valid'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.2))

        return model
    
    model = tf.keras.models.Sequential()
    model.add(Input(shape=(28, 28, 1)))
    
    model = add_conv_block(model, 32)
    model = add_conv_block(model, 64)
    model = add_conv_block(model, 128)

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = create_model()
model.summary()
h = model.fit(
    df_train_f, y_train_oh,
    validation_data=(df_val_f, y_val_oh),
    epochs=20, batch_size=4096
)
preds = model.predict(df_test_f)
print(preds.shape)
pred1 = np.argmax(preds, axis=1)
print(pred1.shape)
print(pred1)
df_final = pd.DataFrame(columns = ['ImageId','Label'])
df_final['Label'] = pred1
df_final['ImageId'] = df_final.index + 1
np.savetxt('final_1.csv',pred1)
df_final.to_csv("final_2.csv")
