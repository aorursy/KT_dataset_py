# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/titanic/train.csv") 

df.head(100)
import string

def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if str.find(big_string, substring) != -1:

            return substring

    print (big_string)

    return np.nan



def replace_nan(substring):

    if (type(substring)==float):

        return "Unknown"

    else:

        return substring

def replace_titles(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Countess', 'Mme']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title
title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev','Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess', 'Don', 'Jonkheer']

df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))

df['Title']=df.apply(replace_titles, axis=1)

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']

df['Cabin']=df['Cabin'].map(lambda x: replace_nan(x))

df['Deck']=df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

#Creating new family_size column

df['Family_Size']=df['SibSp']+df['Parch']

df['Age*Class']=df['Age']*df['Pclass']

df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)

y_tr=df['Survived'].values

df=df.drop(['Name', 'Ticket','Cabin','Survived','PassengerId'], axis=1)

# turn X into dict

df_dict = df.to_dict(orient='records') # turn each row as key-value pairs

# show X_dict



# DictVectorizer

from sklearn.feature_extraction import DictVectorizer

# instantiate a Dictvectorizer object for X

dv_X = DictVectorizer(sparse=False) 

# sparse = False makes the output is not a sparse matrix

# apply dv_X on X_dict

dv_X.fit(df_dict)

df_encoded = dv_X.transform(df_dict)



# show X_encoded

df_encoded

# vocabulary

vocab = dv_X.vocabulary_

# show vocab

vocab

import fancyimpute

x_tr=fancyimpute.KNN(3).fit_transform(df_encoded)
vocab
df = pd.read_csv("/kaggle/input/titanic/test.csv") 

df['Title']=df['Name'].map(lambda x: substrings_in_string(x, title_list))

df['Title']=df.apply(replace_titles, axis=1)

df['Cabin']=df['Cabin'].map(lambda x: replace_nan(x))

df['Deck']=df['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

#Creating new family_size column

df['Family_Size']=df['SibSp']+df['Parch']

df['Age*Class']=df['Age']*df['Pclass']

df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)

df=df.drop(['Name', 'Ticket','Cabin','PassengerId'], axis=1)

# turn X into dict

df_dict = df.to_dict(orient='records') # turn each row as key-value pairs

df_encoded = dv_X.transform(df_dict)

# show X_encoded

df_encoded

import fancyimpute

x_tst=fancyimpute.KNN(3).fit_transform(df_encoded)
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler().fit(x_tr)

x_tr=scaler.transform(x_tr)

x_tst=scaler.transform(x_tst)
#Módulos de keras

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten,GaussianNoise

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import model_from_json

num_classes = 1

batch_size = 10

epochs = 500

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience= 50) #Implementa EarlyStopping para detener el algoritmo si éste no mejorar sus prestaciones, en base al validation loss

filepath1="weights.best.acc.hdf5"  

filepath2="weights.best.loss.hdf5"

checkpointacc = ModelCheckpoint(filepath1, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') #Guarda la información de los pesos del modelo que obtuvo un mejor accuracy de validación

checkpointloss = ModelCheckpoint(filepath2, monitor='val_loss', verbose=1, save_best_only=True, mode='min') #Guarda la información de los pesos del modelo que obtuvo un mejor loss de validación

callbacks_list = [checkpointacc,checkpointloss,es]

model = Sequential() #Genera un modelo secuencial y añade las distintas capas de neuronas

input_shape=(np.shape(x_tr)[1],)

model.add(Dense(200, activation='relu',input_shape=input_shape))

model.add(Dropout(0.3))

model.add(GaussianNoise(0.1))

#model.add(Dense(200, activation='relu'))

#model.add(Dropout(0.3))

#model.add(GaussianNoise(0.1))

#model.add(Dense(200, activation='relu'))

#model.add(Dropout(0.3))

#model.add(GaussianNoise(0.1))

model.add(Dense(200, activation='relu'))

#model.add(Dropout(0.3))

#model.add(GaussianNoise(0.1))

model.add(Dense(num_classes, activation='sigmoid'))

print(model.summary())

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#model.load_weights("./nn/weights.best.acc.hdf5") #Para cargar los pesos guardados de un modelo

hist = model.fit(x_tr, y_tr,batch_size=batch_size,epochs=epochs,verbose=1,callbacks=callbacks_list,validation_split=0.3) #Entrenamiento de la red neuronal, equiparable a la carga de pesos del modelo

#y_pred = model.predict(x_tst) #Predicción en base al dataset de test

#y_pred[y_pred>=0.5] = 1  #Se genera la frontera de decisión

#y_pred[y_pred<0.5] = 0

model.load_weights("weights.best.acc.hdf5")

y_pred = model.predict(x_tst) #Predicción en base al dataset de test

y_pred[y_pred>=0.5] = 1  #Se genera la frontera de decisión

y_pred[y_pred<0.5] = 0

df = pd.read_csv("/kaggle/input/titanic/test.csv") 

column=df["PassengerId"].values
column=column.reshape(np.shape(column)[0],1)

x_csv=np.concatenate((column, y_pred), axis=1).astype(int)
pd_csv=pd.DataFrame(data=x_csv,columns=["PassengerId","Survived"]) 

pd_csv.to_csv('mysubmission.csv',index=False)