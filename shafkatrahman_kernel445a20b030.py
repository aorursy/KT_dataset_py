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
training_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
print(training_data.shape)     
training_data.head()
labels = training_data['label']
training_data.drop('label',axis=1,inplace=True)
labels = labels.to_numpy()
training_data = training_data.to_numpy()
training_data = training_data.reshape(-1,28,28,1)
print(labels.shape,training_data.shape)
test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
test_data = test_data.to_numpy()
test_data = test_data.reshape(-1,28,28,1)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
def make_model(shape):
    W,H,C = shape
    input_layer= Input(shape)
    
   
    
    x = Conv2D(16,(3,3),strides=(1,1),padding='valid')(input_layer)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(1, 1))(x)
    
    x = Conv2D(32,(3,3),strides=(1,1),padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(1, 1))(x)
    
    x = Conv2D(64,(3,3),strides=(1,1),padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(1, 1))(x)
    
    x = Conv2D(128,(3,3),strides=(1,1),padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((3, 3), strides=(1, 1))(x)
    
    x = Flatten()(x)
    x = Dense(10,activation = 'softmax')(x)
    
    model = Model(input_layer,x)
    
    return model
    
model = make_model((28,28,1))
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
model.summary()
history = model.fit(training_data,labels,epochs= 20,validation_split=0.2,shuffle=True)
history = model.fit(training_data,labels,epochs= 20,shuffle=True)
prediction = model.predict(test_data)
print(prediction.shape)
pred = np.argmax(prediction,axis=1)
df = pd.DataFrame(pred, columns=['Label'])
df.insert(0, 'ImageId', df.index + 1)
df.to_csv('submission.csv', index=False)
