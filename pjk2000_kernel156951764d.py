# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import tensorflow.keras.backend  as kb



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
data.head()
type(data)
data.columns

df.drop(['gameId','redFirstBlood','blueTotalGold','redTotalGold','blueTotalExperience','redTotalExperience','redGoldDiff','redExperienceDiff','redKills','redDeaths'], axis=1, inplace=True)
data.drop(['gameId','redFirstBlood','blueTotalGold','redTotalGold','blueTotalExperience','redTotalExperience','redGoldDiff','redExperienceDiff','redKills','redDeaths'], axis=1, inplace=True)
data['blueWins']
data.length()
data.length
len(data)
data[['blueWins']]
targets=data[['blueWins']].values
targets
type(data['blueWins'].values)
features=data.drop('blueWins',axis=1).values
features
len(features)
features.shape

import sklearn.model_selection as model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(features, targets, train_size=0.75,test_size=0.25, random_state=101)
len(X_test)





X_train=kb.constant(X_train)
X_test=kb.constant(X_test)

y_train=kb.constant(y_train)

y_test=kb.constant(y_test)


from keras.layers import Dense

def model():

    model=tf.keras.models.Sequential()

    model.add(Dense(12,input_dim=29,kernel_initializer='normal',activation='relu'))

    model.add(Dense(7,kernel_initializer='normal',activation='relu'))

    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))

    

    

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    return model

    
model=model()
history=model.fit(X_train,y_train,batch_size=64,epochs=20,validation_split=0.2)