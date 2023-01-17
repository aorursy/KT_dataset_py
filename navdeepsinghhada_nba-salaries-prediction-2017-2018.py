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
#finding missing values
null_values_col =df.isnull().sum()
null_values_col = null_values_col[null_values_col != 0].sort_values(ascending = False).reset_index()
null_values_col.columns = ["variable", "number of missing"]
null_values_col.head()
# fill missing values with median 
def fillWithMedian(df):
    return df.fillna(df.median(), inplace=True)

fillWithMedian(df)
#checking null values
df.isnull().any()
#data description
df.describe()
#data head
df.head(10)
#data shape
df.shape
#data info
df.info()
#regression plot for NBA_DraftNumber and Salary
ax=sns.regplot(x='NBA_DraftNumber',y='Salary',data=df)
#regression plot for Age and Salary
ax=sns.regplot(x='Age',y='Salary',data=df)
#regression plot for WS and Salary
ax=sns.regplot(x='WS',y='Salary',data=df)
#regression plot for BPM and Salary
ax=sns.regplot(x='BPM',y='Salary',data=df)
x = df[['NBA_DraftNumber', 'Age', 'WS', 'BPM']] #features
y = df[['Salary']] #label
from sklearn.model_selection import train_test_split
#spliting data into training and testing sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam  
#Using a three-layer neural network with 4 neurons in the input and hidden layers and 1 neuron in output layer
model=keras.Sequential([keras.layers.Dense(4,activation=tf.nn.relu,input_shape=[4,]),
                        keras.layers.Dense(4,activation=tf.nn.relu),
                        keras.layers.Dense(1)])
# #using Adam optimizer with learning rate 1
model.compile(loss='mean_squared_error', optimizer=Adam(0.1))
history = model.fit(x_train,y_train,validation_split=0.25, epochs=70)
#plotting train and validation loss 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
print(model.predict([[23,30,3.1,-1.0]]))
#Model Coefficients
print('Coefficients: \n',model.get_weights())
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
#predict x_test data
y_prediction=model.predict(x_test)
#printing different errors
print(r2_score(y_prediction,y_test))
print(mean_absolute_error(y_prediction,y_test))
print(mean_squared_error(y_prediction,y_test))
# Predict Test Data
#scatterplot of the Y test versus the Y predicted values
plt.scatter(y_test,y_prediction)
plt.xlabel('Y Test')
plt.ylabel('Y Predicted')