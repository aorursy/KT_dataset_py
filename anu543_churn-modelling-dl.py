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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df =pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')

df.shape
df.sample(5)
df.isna().sum()
df.select_dtypes(include='object').columns
for i in ['Geography' ,'Gender']:

    print(df[i].value_counts())

# encoding for [''Geography' ,]Gneder']



geography = pd.get_dummies(df['Geography'], drop_first=True)

gender = pd.get_dummies(df['Gender'], drop_first=True)
final = pd.concat([df,geography ,gender] ,axis =1)

final.head(2)
# Remove unwanted columns

final =final.drop(['RowNumber', 'CustomerId', 'Surname' ,'Geography','Gender'] ,axis =1)
X = final.drop('Exited' ,axis=1)

Y = final['Exited']

X.shape ,Y.shape
from sklearn.model_selection import train_test_split

x_train ,x_test ,y_train ,y_test = train_test_split(X,Y,test_size =.20 ,random_state = 101)

x_train.shape ,x_test.shape ,y_train.shape ,y_test.shape
from sklearn.preprocessing import StandardScaler

sc =StandardScaler()

x_train = sc.fit_transform(x_train)

#y_train = sc.fit_transform(y_train)

x_test = sc.fit_transform(x_test)
import tensorflow as tf

from tensorflow import keras

import keras

from keras.models import Sequential

from keras.layers import Dense ,Dropout
model = Sequential()



# creating Neural n/w

model.add(Dense(output_dim = 8 ,kernel_initializer = 'he_uniform' , activation ='relu', input_dim = 11))



#creating hidden layer

model.add(Dense(output_dim = 6 , kernel_initializer = 'he_uniform' ,activation = 'relu' ))

#creating O/p layers

model.add(Dense(output_dim =1 , kernel_initializer = 'glorot_uniform' , activation ='sigmoid'))
model.compile(optimizer='Adamax' , loss= 'binary_crossentropy' ,metrics=['accuracy'])
model_history  = model.fit(x_train ,y_train ,validation_split= .20 ,batch_size= 10 ,nb_epoch = 6)
print(model_history.history.keys)
plt.plot(model_history.history['accuracy'])

plt.plot(model_history.history['val_accuracy'])

plt.title("ACCURACY")

plt.ylabel("Accuracy")

plt.xlabel("epoch")

plt.legend(['train' ,'test'],loc = 'upper_left')
pred = model.predict(x_test)

pred
pred = (pred>0.5)

pred
from sklearn.metrics import confusion_matrix ,accuracy_score

confusion_matrix(y_test ,pred)
accuracy_score(y_test ,pred)