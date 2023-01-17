import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

train = pd.read_csv("../input/train.csv",index_col = 'PassengerId')

test = pd.read_csv("../input/test.csv",index_col = 'PassengerId')

train.describe()

test.describe()
train['Age'].fillna(train['Age'].median(), inplace=True)
train['Mr'] = 0

train['Mrs'] = 0

train['Miss'] = 0

train['royalty'] = 0

train['officer'] = 0


for index,row in train.iterrows():

    name = row['Name']

    if 'Mr.' in name:

        train.set_value(index,'Mr',1)

    elif 'Mrs.' in name:

        train.set_value(index,'Mrs',1)

    elif 'Miss.' in name:

        train.set_value(index,'Miss',1)

    elif 'Lady' or 'Don' or 'Dona' or 'sir' or 'master' in name:

        train.set_value(index,'royalty',1)

    elif 'rev' in name:

        train.set_value(index,'officer',1)

        

train.head()
train.drop('Name',inplace=True, axis=1)

train.head() 


train['Embarked_S'] = 0

train['Embarked_C'] = 0

train['Embarked_Q'] = 0

train['Embarked_unknown'] = 0



for index,row in train.iterrows():

    embarkment = row['Embarked']

    if embarkment == 'S':

        train.set_value(index,'Embarked_S',1)

    elif embarkment == 'C':

        train.set_value(index,'Embarked_C',1)

    elif embarkment == 'Q':

        train.set_value(index,'Embarked_Q',1)

    else:

        train.set_value(index,'Embarked_unknown',1)

   



train.head()

train.drop('Embarked', inplace = True, axis = 1) 


for index,row in train.iterrows():

    if row['Sex'] == 'male':

        train.set_value(index, 'Sex', 1)

    else:

        train.set_value(index,'Sex',0)

train.head()
train.drop('Ticket', inplace= True, axis = 1)

train.head()
train['Fare_cheap']=0

train['Fare_average']=0

train['Fare_expensive']=0



for index,row in train.iterrows():

    if row['Fare'] <= 30.0 :

        train.set_value(index, 'Fare_cheap', 1)

    elif row['Fare'] > 30 and  row['Fare'] <= 50.0:

        train.set_value(index,'Fare_average',1)

    else:

        train.set_value(index, 'Fare_expensive',1)

        

train.head()
train.drop('Fare',inplace = True, axis =1) 

train.head()


train.drop('Cabin',inplace = True, axis = 1)

train.head()
train.describe() 
X = train[['Pclass','Sex','Age','SibSp','Parch','Mr','Mrs','Miss','royalty','officer','Embarked_S','Embarked_C','Embarked_Q','Embarked_unknown','Fare_cheap','Fare_average','Fare_expensive']]

y = train.Survived



X.shape
y.shape
from sklearn.svm import SVC

from sklearn.cross_validation import cross_val_score 



svm_model = SVC() 

svm_model.kernel= 'linear'

score_svm = cross_val_score(svm_model,X,y,cv=10, scoring= 'accuracy')

print(score_svm.mean())