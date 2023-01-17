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
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv', dtype=np.float32)

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv',dtype= np.float32)
train_data.head()
test_data.head()
train_data.describe()
train_data.info()
test_data.info()
train_data.ndim#for dimensions
train_data.isnull().sum()
test_data.isnull().sum()
y_train = train_data['label'].values

print(y_train)

x_train = train_data.loc[:,train_data.columns!='label'].values/255

print(x_train)
x_test = test_data.values/255

print(x_test)
import tensorflow as tf

import matplotlib.pyplot as plt



def visual(data, index):

    plt.imshow(data[index].reshape(28,28))

    plt.show()

    

visual(x_train,3)    
x_train = x_train.reshape(x_train.shape[0],28,28)

x_train.shape
x_test = x_test.reshape(x_test.shape[0],28,28)

x_test.shape
from tensorflow import keras

from keras import layers

from keras import models

print(tf.__version__)



model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape=(28,28)),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10, activation='softmax')

])

model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train, epochs=30,batch_size=6000)
preds = model.predict_classes(x_test, verbose=0)
submission_df = pd.DataFrame({'ImageId': list(range(1,len(preds)+1)),'Label': preds})

print(submission_df)
submission_df.to_csv('Submission.csv', index=False)