# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')



print(data)
data['with_children']=data['Name'].apply(lambda x: '(' in x)

print(data['with_children'])
print(data.columns)

columns=data.columns

for i in columns:

    print('{} null number: {}'.format(i,np.sum(data[i].isnull())))
import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('fivethirtyeight')
print(data.pivot_table(values='Survived', index='with_children',columns=['Sex'], aggfunc='count'))

print(data[data['Survived']==1].pivot_table(values='Survived', index='with_children',columns=['Sex'], aggfunc='count'))
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=500, max_depth=None,

    min_samples_split=2, random_state=0,verbose=3)
features=['Pclass', 'Sex', 'Age', 'SibSp',

       'Parch', 'Fare', 'Embarked', 'with_children']
data['Age'].fillna(data['Age'].mean(),inplace=True)

print(data.groupby('Embarked').PassengerId.count())

data['Embarked'].fillna('S',inplace=True)

data['Embarked'].replace(to_replace={'C':0,'Q':1,'S':2},inplace=True)
data['Sex'].replace(to_replace={'male':0,'female':1},inplace=True)

learnData=data[features]

learnTarget=data['Survived']

clf.fit(learnData,learnTarget)
test['Age'].fillna(test['Age'].mean(),inplace=True)

test['Embarked'].fillna('S',inplace=True)

test['Embarked'].replace(to_replace={'C':0,'Q':1,'S':2},inplace=True)

test['with_children']=test['Name'].apply(lambda x: '(' in x)

test['Sex'].replace(to_replace={'male':0,'female':1},inplace=True)
testData=test[features]

testData['Fare'].fillna(testData['Fare'].mean(),inplace=True)

for i in testData.columns:

    print('{} null {}'.format(i,np.sum(testData[i].isnull())))

target=clf.predict(testData)
print(target)
submit=pd.read_csv('../input/genderclassmodel.csv')

print(submit)
print(test)
submit['Survived']=target

print(submit)
submit.to_csv('submit.csv',index=False)