# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_excel('../input/Data_OpelCorsa.xlsx')
data.head()
#Preprocessing Adım 2: phone number, area code, state özniteliklerinin kaldırılması
data.drop('Column1', axis = 1, inplace = True)
data.drop('baslik', axis = 1, inplace = True)
data.drop('ilce', axis = 1, inplace = True)
data.drop('mahalle', axis = 1, inplace = True)
data.drop('para_birimi', axis = 1, inplace = True)
data.drop('ilan_no', axis = 1, inplace = True)
data.drop('ilan_tarihi', axis = 1, inplace = True)
data.drop('marka', axis = 1, inplace = True)
print("After Dataset preprocessing: " + str(data.shape))
data.head()

print('Original Features:\n', list(data.columns), '\n')
data= pd.get_dummies(data)
print('Features after One-Hot Encoding:\n', list(data.columns))
# Scatter Plot 
data.plot(kind='scatter', x='fiyat', y='yil',alpha = 0.5,color = 'red')
plt.xlabel('price')              # label = name of label
plt.ylabel('year')
plt.title('Fiyat ve yil Scatter Plot') 

data.plot(kind='scatter', x='fiyat', y='km',alpha = 0.5,color = 'grey')
plt.xlabel('price')              # label = name of label
plt.ylabel('km')
plt.title('Fiyat ve km Scatter Plot') 
data.plot(kind='scatter', x='fiyat', y='motor_gucu_hp',alpha = 0.5,color = 'green')
plt.xlabel('price')              # label = name of label
plt.ylabel('machine power')
plt.title('fiyat ve motor_gucu_hp Scatter Plot') 


# Importing the dataset
X = data.iloc[:, data.columns != 'fiyat']
y = data.fiyat

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=120, kernel_initializer='normal', activation='relu'))
    model.add(Dense(output_dim = 120, init = 'uniform', activation = 'relu'))
    model.add(Dense(output_dim = 120, init = 'uniform', activation = 'relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mse',
                optimizer='adam',
                metrics=['mae'] )
    return model

model = baseline_model()
model.summary()

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

# Store training stats
history = model.fit(X_train, y_train, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])
test_predictions = model.predict(X_test).flatten()

plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])
error = test_predictions - y_test
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [1000 TL]")
_ = plt.ylabel("Count")
error = np.sum(np.sqrt(test_predictions - y_test))
error/len(data)
from sklearn.metrics import r2_score
r2_score(y_test, test_predictions)
