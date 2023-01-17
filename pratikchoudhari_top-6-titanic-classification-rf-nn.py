# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np 

np.random.seed(0)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

gs = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')



#drop not much useful columns

train = train.drop(['Cabin','Ticket'],axis=1)

test = test.drop(['Cabin','Ticket'],axis=1)
def binAges(x):

    if x<18:

        return 1

    elif 17<x<35:

        return 2

    elif 34<x<51:

        return 3

    elif 50<x<66:

        return 4

    elif x<65:

        return 5



#get Mr mrs etc

def stripit(x):

    return x.split(',')[1].split('.')[0].strip()    
title_dict = {"Capt": "Rare","Col": "Rare","Major": "Rare","Jonkheer": "Rare","Don": "Rare","Dona": "Miss",

    "Sir" : "Rare","Dr": "Rare","Rev": "Rare","the Countess":"Rare","Mme": "Mrs","Mlle": "Miss",

    "Ms": "Mrs","Mr" : "Mr","Mrs" : "Mrs","Miss" : "Miss","Master" : "Master","Lady" : "Rare"}

title_map = {"Mr": 1, "Master": 2, 

    "Mrs": 3, "Miss": 4, 

    "Rare": 5}
def preprocess(df):

    df['desig'] = df.Name.apply(stripit)

    df.desig = df.desig.map(title_dict)

    df.desig = df.desig.map(title_map)



    #bin ages and fill with mean

    mean = df.Age.mean()

    df.Age = df.Age.fillna(mean)

    df.Age = df.Age.apply(binAges)



    #create family and individual features

    df['Family'] = df.Parch + df.SibSp

    return df
processed_train = preprocess(train).dropna(axis=0)
X = processed_train.drop(['PassengerId','Name','Survived','SibSp','Parch'],axis=1)

y = processed_train.Survived

train.corr().style.background_gradient()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler



X_d = pd.get_dummies(X)

mms = MinMaxScaler().fit(X_d)

X_d = mms.transform(X_d)

X_train, X_val, y_train, y_val = train_test_split(X_d,y,test_size=0.2,random_state=0)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(random_state=111)

rf.fit(X_train, y_train)

rf.score(X_val, y_val)
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score



param_grid = { 

    'n_estimators': [100, 200, 300],

    'max_depth' : [5,6,7,8],

    'criterion' :['gini', 'entropy']

}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, scoring='accuracy', cv=5)

grid_search.fit(X_train, y_train)

print('best params:{}\nbest score:{}'.format(str(grid_search.best_params_), str(grid_search.best_score_)))
rf = RandomForestClassifier(n_estimators=100,max_depth=5,criterion='gini',random_state=111)

rf.fit(X_train, y_train)

rf.score(X_val, y_val)
processed_test = preprocess(test).fillna(0)

X_test = processed_test.drop(['PassengerId','Name','SibSp','Parch'],axis=1)

X_test = pd.get_dummies(X_test)

X_test = mms.transform(X_test)

pred = rf.predict(X_test)

submit = pd.DataFrame({'PassengerId':test.PassengerId,'Survived':sample}).reset_index(drop=True)

submit.to_csv('Submission.csv',index=False)
from keras.models import Sequential 

from keras.layers import Activation, Dense 

model = Sequential()

model.add(Dense(50, input_shape=(10,), activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.summary()
model.fit(X_train,y_train,validation_split=0.33,epochs=400,validation_data=(X_val, y_val),batch_size=32)
processed_test = preprocess(test).fillna(0)

X_test = processed_test.drop(['PassengerId','Name','SibSp','Parch'],axis=1)

X_test = pd.get_dummies(X_test)

X_test = mms.transform(X_test)

sample = map(int,np.round(model.predict(X_test)).reshape(418,))

submit = pd.DataFrame({'PassengerId':test.PassengerId,'Survived':sample}).reset_index(drop=True)

submit.to_csv('Submission.csv',index=False)