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
df=pd.read_csv("/kaggle/input/titanic/train.csv")

db=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

dc=pd.read_csv("/kaggle/input/titanic/test.csv")
df['Sex'] = df['Sex'].replace({'male':1, 'female':2})

df['EmS'] = df['Embarked'].replace({'S':1, 'Q':0, 'C':0})  

df['EmQ'] = df['Embarked'].replace({'S':0, 'Q':1, 'C':0})

df['EmC'] = df['Embarked'].replace({'S':0, 'Q':0, 'C':1})

df['Age']=df['Age'].fillna(30)

df['Ticket']=df['Ticket'].apply(lambda x: len(x))

df.pop('Name')

df.pop('PassengerId')

df['Cabin']=df['Cabin'].fillna('A')

df['Cabin']=df['Cabin'].apply(lambda x: len(x))

df.pop('Embarked')
#Запасное восстановление

dc=pd.read_csv("/kaggle/input/titanic/test.csv")
dc['Sex'] = dc['Sex'].replace({'male':1, 'female':2})

dc['EmS'] = dc['Embarked'].replace({'S':1, 'Q':0, 'C':0})  

dc['EmQ'] = dc['Embarked'].replace({'S':0, 'Q':1, 'C':0})

dc['EmC'] = dc['Embarked'].replace({'S':0, 'Q':0, 'C':1})

dc['Age']=dc['Age'].fillna(30)

dc['Ticket']=dc['Ticket'].apply(lambda x: len(x))

dc.pop('Name')

dc.pop('PassengerId')

dc['Cabin']=dc['Cabin'].fillna('A')

dc['Cabin']=dc['Cabin'].apply(lambda x: len(x))

dc.pop('Embarked')
df['AgeGroup'] = pd.cut(df['Age'], [0,4,17,65,75,100], right=False)

onehotencoding = pd.get_dummies(df['AgeGroup'])

onehotencoding.columns = ['age1','age2','age3','age4','age5']

df = pd.concat([df, onehotencoding], axis=1)
df.pop('AgeGroup')

df.pop('Age')
dc['AgeGroup'] = pd.cut(dc['Age'], [0,4,17,65,75,100], right=False)

onehotencoding = pd.get_dummies(dc['AgeGroup'])

onehotencoding.columns = ['age1','age2','age3','age4','age5']

dc = pd.concat([dc, onehotencoding], axis=1)
dc['FareGroup'] = pd.cut(dc['Fare'], [0,8,31,56,75,100], right=False)

onehotencoding = pd.get_dummies(dc['FareGroup'])

onehotencoding.columns = ['fare1','fare2','fare3','fare4','fare5']

dc = pd.concat([dc, onehotencoding], axis=1)

dc.pop('FareGroup')

dc.pop('Fare')
df['FareGroup'] = pd.cut(df['Fare'], [0,8,31,56,75,100], right=False)

onehotencoding = pd.get_dummies(df['FareGroup'])

onehotencoding.columns = ['fare1','fare2','fare3','fare4','fare5']

df = pd.concat([df, onehotencoding], axis=1)

df.pop('FareGroup')

df.pop('Fare')
dc.pop('AgeGroup')

dc.pop('Age')
df
df.groupby('Survived')['PC'].hist()
list(df['PC'])
list(v['Ticket'])
def flagPC(string):

    if 'PC' in string:

        return 1

    else:

        return 0
df['PC']=df['Ticket'].apply(flagPC)
dc['PC']=dc['Ticket'].apply(flagPC)
from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.metrics import accuracy_score



y = df['Survived']

x = df[df.columns[1:]]

best = 0 

average = 0

bestdepth=0

bestestim=0

bestlearn=0

total_for_average = 0



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

for i in range(1,10):

    for j in range(1,20):

        model = xgb.XGBClassifier(max_depth=i*1, n_estimators=j*10, learning_rate=0.01*j)

        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        #print(accuracy_score(y_test, y_pred))

        total_for_average += 1

        average += accuracy_score(y_test, y_pred)

        if (accuracy_score(y_test, y_pred) > best): 

            best = accuracy_score(y_test, y_pred)

            bestdepth=i*1

            bestestim=j*10

            bestlearn=0.01*j

print("\nThe Best is", best)

print("\nThe Best things are", bestdepth, bestestim, bestlearn)

print("The Average is", average/total_for_average)
y = df['Survived']

x = df[df.columns[1:]]

best = 0 

average = 0

total_for_average = 0



models=[]

for i in range(50):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)

    model = xgb.XGBClassifier(max_depth=3, n_estimators=180, learning_rate=0.18)

    model.fit(x_train, y_train)

    models.append(model)

    y_pred = model.predict(x_test)





print(accuracy_score(y_test, y_pred))

total_for_average += 1

average += accuracy_score(y_test, y_pred)
db=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
dc


y_pred = np.zeros(len(dc))

for model in models:

    y_pred += model.predict(dc)



for i in range(len(dc)):    

    if y_pred[i]>23:

         y_pred[i]=1 

    else:    

         y_pred[i]=0

        

y_pred = pd.DataFrame(y_pred).astype(int)

db['Survived'] = y_pred
db
db.to_csv("BJladikaSumbmission.csv", index = False)