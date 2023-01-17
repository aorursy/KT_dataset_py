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
# Importing the Required Library
import pandas as  pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Dense,Activation,Conv2D,MaxPooling2D,Flatten
from kerastuner.tuners import RandomSearch
# Reading the Data
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
test.columns
# taking the independent feature and the dependent feature in xtrain and ytrain of the traning data
xtrain = train.drop(columns = 'label')
ytrain = train['label']
# The shape of train and test dataset
print('test : ',test.shape,'xtrain : ',xtrain.shape)
# Scaling the Data
xtrain = xtrain/255.0
test = test/255.0
# Reshaping the Data
xtrain = xtrain.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
# Converting the Categorical data  
ytrain = to_categorical(ytrain,num_classes=10)

xtrain_t,xtest_t,ytrain_t,ytest_t = train_test_split(xtrain,ytrain,test_size= 0.1,random_state = 42)

# Building the Model
ypred = None
def model_builder():
    model = Sequential([
          Conv2D(28,kernel_size = 3,padding='same',input_shape =(28,28,1),activation ='relu'),
          MaxPooling2D(pool_size = (2,2)),
          Conv2D(28,kernel_size=(2,2),padding='valid',activation='relu'),
          MaxPooling2D(pool_size =(2,2)),
          Conv2D(28,kernel_size=(2,2),padding='valid',activation='relu'),
          Dropout(0.2),
          Flatten(),
          Dense(512,activation='relu'),
          Dropout(0.5),
          Dense(256,activation='relu'),
          Dense(ytrain_t.shape[1],activation = 'softmax')
  ])
    model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics = ['accuracy'])
    model.fit(xtrain_t,ytrain_t,epochs = 20,validation_data = (xtest_t,ytest_t),batch_size = 64)
    ypred = model.predict(test)
    return ypred,model
  # return model
# Running
pred,model = model_builder()
model.summary()
# checking the predictions
predictions = np.argmax(pred, axis=1)
predictions

ImageId = np.arange(1,28001)
output = pd.DataFrame({'ImageId':ImageId, 'Label':predictions})
output.to_csv('output.csv', index=False)
print(output)
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
submission.head()
