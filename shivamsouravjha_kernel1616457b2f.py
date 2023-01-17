# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

b = '/kaggle/input/titanic/train.csv'

a = pd.read_csv(b)

c = '/kaggle/input/titanic/test.csv'

d = pd.read_csv(c)



e = [a,d]

a
def bar_char(feature):

    surv = a[a["Survived"]==1][feature].value_counts()

    dead = a[a["Survived"]==0][feature].value_counts()

    df = pd.DataFrame([surv,dead])

    df.index = ['surv','dead']

    df.plot(kind ='bar',stacked ='True',fig=(10,5))



bar_char('Pclass')








a['Age'].fillna(inplace = True,value  = a['Age'].mean())

d['Age'].fillna(inplace = True,value  = d['Age'].mean())







for dataset in e:

    

     dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} )

for dataset in e:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

for dataset in e:

    #print(dataset.Embarked.unique())

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

for dataset in e:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4



for dataset in e:

        dataset['Fare'].fillna(inplace = True,value  = dataset['Fare'].mean())

for dataset in e:

    dataset.loc[ dataset['Fare'] <= 7, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7) & (dataset['Fare'] <= 14), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14) & (dataset['Fare'] <= 31), 'Fare'] = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 100), 'Fare'] = 3

    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 4





    

    

    

    

    

    

    

    

    

    

a['first'] = a['Pclass'].apply(lambda p: 1 if p == 1 else 0 )    

a['second'] = a['Pclass'].apply(lambda p: 1 if p==2 else 0 )    

a['third'] = a['Pclass'].apply(lambda p: 1 if p==3 else 0 )    

    

    

    

    

    

    

    

    

d['first'] = d['Pclass'].apply(lambda p: 1 if p == 1 else 0 )    

d['second'] = d['Pclass'].apply(lambda p: 1 if p==2 else 0 )    

d['third'] = d['Pclass'].apply(lambda p: 1 if p==3 else 0 )    

    

feature = ['Age','Sex','Embarked','SibSp','Fare','second','third','Parch']    

trainx = a[feature]



trainy = a['Survived']





model = RandomForestClassifier(n_estimators=500)





model.fit(trainx,trainy)





testX  = d[feature]















y = model.predict(testX)

















#print(y)

output = pd.DataFrame({'Passengerid': d.PassengerId, 'Survived': y})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")




