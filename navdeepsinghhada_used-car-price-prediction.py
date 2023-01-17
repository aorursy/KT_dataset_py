# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#read data file
df = pd.read_csv(os.path.join(dirname, filename))
#data description
df.describe()
#fraction of data
df.head(10)
#data shape
df.shape
#data info
df.info()
#regression plot for current price and km
ax=sns.regplot(x='km',y='current price',data=df)
#regression plot betwwen current price and torque
ax=sns.regplot(x='torque',y='current price',data=df)
x=df[['on road old','on road now','years','km','rating','condition','economy','top speed','hp','torque']] #features
y=df['current price'] #label
from sklearn.model_selection import train_test_split
#split the data into training and testing sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
#using single unit Dense layer with input_shape=10(for 10 features)
model=keras.Sequential([keras.layers.Dense(10,activation=tf.nn.relu,input_shape=[10,]),
                        keras.layers.Dense(1)])
#using Adam optimizer with learning rate 0.01 to train model
model.compile(optimizer=Adam(0.01),loss='mean_squared_error')
history=model.fit(x_train,y_train,validation_split=0.25,epochs=30)
#plotting train and validation loss 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
print(model.predict([[535651,798186,3,78945,1,2,14,177,73,123]]))
#coefficients of the model
print('Coefficients: \n',model.get_weights())
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
y_prediction=model.predict(x_test)
print(r2_score(y_prediction,y_test))
print(mean_absolute_error(y_prediction,y_test))
print(mean_squared_error(y_prediction,y_test))
# Predict Test Data
#y_prediction=model.predict(x_test)
#scatterplot of the Y test versus the Y predicted values
plt.scatter(y_test,y_prediction)
plt.xlabel('Y Test')
plt.ylabel('Y Predicted')