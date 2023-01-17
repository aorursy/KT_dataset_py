# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/digit-recognizer/train.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
digits = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
digits.head(25)
X = digits.drop('label',axis=1)
y= digits['label']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size =0.2,random_state=1) 
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
plt.rcParams['font.size']=14
x_train, x_test = X_train/255.0,  X_test/255.0
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
#Defining layers of the model
model.add(Dense(128, activation= 'sigmoid', input_shape = (784,)))
model.add(Dense(64, activation= 'sigmoid'))
model.add(Dense(10, activation= 'softmax'))
model.summary()
#Compiling the Model
model.compile(optimizer='sgd', loss= 'categorical_crossentropy', metrics=['accuracy'])
%%time
#Training the Model
history = model.fit(x_train, y_train, batch_size= 32, epochs=20, validation_split=0.2)
pd.DataFrame(history.history).plot(figsize=(10,7))
plt.grid(True)
plt.show()
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.regularizers import  l1, l2
from tensorflow.keras.layers import Dropout
model12 = Sequential()
model12.add(Dense(256, activation= 'elu', input_shape = (784,)))
# model12.add(BatchNormalization())
model12.add(Dropout(0.25))
model12.add(Dense(128, activation= 'elu'))
# model12.add(BatchNormalization())
model12.add(Dropout(0.25))
model12.add(Dense(64, activation= 'elu'))
# model12.add(BatchNormalization())
model12.add(Dense(32, activation= 'elu'))
# model12.add(BatchNormalization())
model12.add(Dense(10, activation= 'softmax'))
model12.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics=['accuracy'])
history12 = model12.fit(x_train, y_train, batch_size= 64, epochs= 20, validation_split=0.2)
pd.DataFrame(history12.history).plot(figsize=(10,7))
plt.grid(True)
plt.show()
model11 = Sequential()
#Defining layers of the model
model11.add(Flatten(input_shape  = (784,)))
model11.add(Dense(256))
model11.add(BatchNormalization())
# model11.add(Dropout(0.25))
model11.add(Activation('elu'))
model11.add(Dense(128))
# model11.add(BatchNormalization())
# model11.add(Dropout(0.25))
model11.add(Activation('elu')) 
model11.add(Dense(64))
model11.add(BatchNormalization())
model11.add(Activation('elu'))
model11.add(Dense(32))
model11.add(BatchNormalization())
model11.add(Activation('elu'))
model11.add(Dense(10, activation= 'softmax'))
model11.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['accuracy']) 
history11 = model11.fit(x_train, y_train ,batch_size= 64, epochs=32, validation_split=0.3)
pd.DataFrame(history11.history).plot(figsize=(10,7))
plt.grid(True)
plt.show()
model11.evaluate(x_test, y_test, verbose=0)
model1 = Sequential()
#Defining layers of the model
model1.add(Flatten(input_shape  = (784,)))
model1.add(Dense(256))
# model1.add(BatchNormalization())
# model11.add(Dropout(0.25))
model1.add(Activation('elu'))
model1.add(Dense(128))
# model11.add(BatchNormalization())
# model11.add(Dropout(0.25))
model1.add(Activation('elu')) 
model1.add(Dense(64))
model1.add(BatchNormalization())
model1.add(Activation('elu'))
model1.add(Dense(32))
# model1.add(BatchNormalization())
model1.add(Activation('elu'))
model1.add(Dense(10, activation= 'softmax'))
model1.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['accuracy']) 
history1 = model1.fit(x_train, y_train ,batch_size= 64, epochs=32, validation_split=0.3)
model1.evaluate(x_test, y_test, verbose=0)
model2 = Sequential()
#Defining layers of the model
model2.add(Flatten(input_shape  = (784,)))
model2.add(Dense(512))
model2.add(Activation('relu'))
model2.add(Dense(256))
model2.add(Activation('relu'))
model2.add(Dense(128))
model2.add(Activation('relu')) 
model2.add(Dense(128))
model2.add(Activation('relu')) 
model2.add(Dense(64))
model2.add(Activation('relu'))
model2.add(Dense(32))
model2.add(Activation('relu'))
model2.add(Dense(10, activation= 'softmax'))
model2.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics=['accuracy']) 
history2 = model2.fit(x_train, y_train ,batch_size= 64, epochs=50, validation_split=0.2)
model2.evaluate(x_test, y_test, verbose=0)
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test.head()
test = test/255.0
pred=model2.predict(test)
pred_classes = np.argmax(pred,axis = 1) 
results = pd.Series(pred_classes,name="Label")
results
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission_digit_divya.csv",index=False)
