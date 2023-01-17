import tensorflow as tf

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

%matplotlib inline

import seaborn as sns

import re

#from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import keras

from keras.models import Sequential

from keras.layers import Dense

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.shape)

print(test.shape)

train.head()
t_data = [train,test]

for dataset in t_data:

    dataset['Name_length'] = dataset['Name'].apply(len)

    dataset['Family_Size'] = dataset['SibSp'] + dataset['Parch'] +1

for dataset in t_data:    

    dataset['IsAlone'] = 0

    dataset.loc[dataset['Family_Size']==1,'IsAlone'] =1

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    dataset['Has_Cabin'] = dataset['Cabin'].apply(lambda x: 0 if type(x)==float else 1)



for dataset in t_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    

train['Categorical_Fare'] = pd.qcut(train['Fare'],4)  

for dataset in t_data:

    avg_age = dataset['Age'].mean()

    std_age = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()



    age_null_randomlist = np.random.randint(avg_age-std_age,avg_age+std_age,size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_randomlist

    dataset['Age'] = dataset['Age'].astype(int)

train['Categorical_Age'] = pd.qcut(train['Age'],5)



def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    else:

        return " "



for dataset in t_data:    

    dataset['Title'] = dataset['Name'].apply(get_title)

    

for dataset in t_data:    

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')

    dataset['Title'] = dataset['Title'].replace('Ms','Miss')

    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

    

for dataset in t_data:

    dataset['Sex'] = dataset['Sex'].map({'female':0,'male':1}).astype(int)

    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2})

    dataset['Title'] = dataset['Title'].map({'Miss':0,'Master':1,'Rare':2,'Mrs':3,'Mr':4})

 

    dataset.loc[dataset['Fare']<=7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] >7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] >14.454) & (dataset['Fare'] <=31), 'Fare'] = 2

    dataset.loc[dataset['Fare'] >31 ,'Fare'] = 3

    

    dataset.loc[dataset['Age']<=19, 'Age'] = 0

    dataset.loc[(dataset['Age'] >19) & (dataset['Age'] <=25) ,'Age'] = 1

    dataset.loc[(dataset['Age'] >25) & (dataset['Age'] <=31) ,'Age']  =2

    dataset.loc[(dataset['Age'] >31) & (dataset['Age'] <=40) ,'Age'] =3

    dataset.loc[(dataset['Age'] >40) & (dataset['Age'] <=80) ,'Age'] =4

    

    

    

    





#for dataset in t_data:

    
dropelements = ['PassengerId','Name','SibSp','Parch','Ticket','Cabin']

train = train.drop(dropelements ,axis=1)
#test = test.drop(dropelements ,axis =1)

train = train.drop(['Categorical_Fare','Categorical_Age'],axis=1)
train.head(20)
test = test.drop(dropelements ,axis=1)
train = train.drop('Name_length',axis=1)

test = test.drop('Name_length',axis=1)
test.head(10)
sns.countplot(x = 'Survived',hue='Sex',data = train)
sns.countplot(x='Survived',hue='Family_Size',data=train)
sns.countplot(x='Survived',hue='Pclass',data=train)
train.isnull().sum()
test.isnull().sum()
y_train = train['Survived']

x_train = train.drop(['Survived'], axis=1)

y_train = y_train.values

x_train= x_train.values

x_test = test.values
y_train = y_train.reshape(891,1)

print(y_train.shape)

print(x_train.shape)

print(x_test.shape)
sample_sub = pd.read_csv('../input/gender_submission.csv')

sc = StandardScaler()

X = sc.fit_transform(x_train)

test = sc.fit_transform(x_test)
model = Sequential()

model.add(Dense(9,kernel_initializer='uniform',activation='relu',input_dim=9))

model.add(Dense(9,kernel_initializer='uniform',activation='relu'))

model.add(Dense(5,kernel_initializer='uniform',activation='relu'))

model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

          
model.summary()
model.compile(optimizer='adam',loss ='binary_crossentropy',metrics=['accuracy'])

model.fit(X,y_train,batch_size=32,nb_epoch=300)
y_pred = model.predict(test)

y_final = (y_pred>0.5).astype(int).reshape(test.shape[0])
sample_sub['Survived'] = y_final

sample_sub.to_csv('sample_final.csv',index=False)

sample_sub.head()