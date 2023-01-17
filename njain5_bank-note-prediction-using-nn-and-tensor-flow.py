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
bank_note_data=pd.read_csv('/kaggle/input/bank-note-authentication-uci-data/BankNote_Authentication.csv')
import pathlib, os

import matplotlib.pyplot as plt

import seaborn as sns



import tensorflow as tf

print(tf.__version__)
bank_note_data.head()
bank_note_data.info()
bank_note_data.describe()
bank_note_data.hist(figsize=(20,10), grid =False, layout = (2,4), bins=30)
plt.figure(figsize = (20,7))

sns.swarmplot(x = 'class', y = 'curtosis', data = bank_note_data, hue = 'class')

sns.violinplot(x = 'class', y = 'curtosis', data = bank_note_data)
#Standardize rows into uniform scale.



X= bank_note_data.drop(['class'], axis=1)

y= bank_note_data['class']



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(X)



#Scale and centre the data



bank_note_data_normalized = scaler.transform(X)



#Create a pandas dataframe



bank_note_data_normalized = pd.DataFrame(data = X, index= X.index, columns= X.columns)
bank_note_data_normalized.describe()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0,stratify=bank_note_data['class'])
X_train.shape
X_test.shape
y_train.shape
y_test.shape
#Create a Sequential model

from keras.layers import Dropout

model = tf.keras.Sequential()

model.add(Dropout(0.2, input_shape=(4,)))

model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))



#Output Layer

model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))



#Create a Keras version Optimiser

optimizer = tf.keras.optimizers.Adam()



#Compile and print the summary of model

model.compile(loss='binary_crossentropy',

              optimizer=optimizer,

              metrics =['accuracy'])



# Model summary can be created by calling the summary() function on the model that returns a string that in turn can be printed.

model.summary()
#Plot model summary



tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
fitted_model = model.fit(

        X_train, y_train,

        epochs=50  

        )
eval = model.evaluate(X_test, y_test, verbose=0) 

print("\nLoss, accuracy on test data: ")

print("%0.4f %0.2f%%" % (eval[0],eval[1]*100))