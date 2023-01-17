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
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.sample(4)
# checking missing value
df.isna().sum()
# target valriable
df['quality'].value_counts()
import seaborn as sns
sns.countplot(df['quality'])
x = df.drop('quality' , axis=1).values
y = df['quality'].values.reshape(-1 ,1)
x.shape ,y.shape

# spliting the dataset into train and test
from sklearn.model_selection import train_test_split
x_train ,x_test, y_train ,y_test = train_test_split(x ,y , test_size =.20 ,random_state =101)
x_train.shape ,x_test.shape, y_train.shape ,y_test.shape
# Features scaling
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
x_train =sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# implementinf DEEP LEARNING
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units = 6 , kernel_initializer = 'he_uniform' ,activation = 'relu' , input_dim = 11))
model.add(Dense(units = 6 , kernel_initializer = 'he_uniform' , activation='relu'))
model.add(Dense(units = 1 , kernel_initializer = 'glorot_uniform' , activation ='sigmoid'))


model.compile(optimizer='adamax' , loss= 'binary_crossentropy' ,metrics= ['accuracy'])
model_history = model.fit(x_train ,y_train ,validation_split= .20 ,batch_size= 10 ,nb_epoch =5)
# summary of MOdeL
model_history.history['accuracy']
model_history.history['val_accuracy']
import matplotlib.pyplot as plt
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['train' ,'test'])
pred = model.predict(x_test)
pred
from sklearn.metrics import accuracy_score
pred = (pred>0.5)
accuracy_score(y_test ,pred)
