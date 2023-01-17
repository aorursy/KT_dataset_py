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
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
def extract(m):

    m = m.split(',')[1]

    m = m.split('.')[0]

    return m[1:]
path = '/kaggle/input/titanic/train.csv'

df = pd.read_csv(path)

df['Name'] = df['Name'].apply(extract)

df['Name'] = df['Name'].apply(lambda x: x if x in ['Mr','Mrs','Miss','Master'] else 'Others')

df['Parch'] = df['Parch'].apply(lambda x: x if x in [0,1,2] else 4.5)

df = df.fillna(df.mean())

df['Embarked'] = df['Embarked'].apply(lambda x : x if (x=='C' or x=='Q') else 'S')

df['Embarked'].unique()



print(df.count())

le = LabelEncoder()



le.fit(df['Sex'])

df['Sex'] = le.transform(df['Sex'])



le.fit(df['Name'])

df['Name'] = le.transform(df['Name'])



le.fit(df['Embarked'])

df['Embarked'] = le.transform(df['Embarked'])
features = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Name']

y = df['Survived']

X = df[features]
for i in range(1):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(n_estimators = 200, max_depth = 3)

    model.fit(X_train,y_train)



    print(i,(model.predict(X_train)-y_train==0).sum()*100/len(y_train))

    print((model.predict(X_test)-y_test==0).sum()*100/len(y_test),'\n')
df3 = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

df3
model.fit(X,y)

df2 = pd.read_csv('/kaggle/input/titanic/test.csv')

df2['Name'] = df2['Name'].apply(extract)

df2['Name'] = df2['Name'].apply(lambda x: x if x in ['Mr','Mrs','Miss','Master'] else 'Others')

df2['Parch'] = df2['Parch'].apply(lambda x: x if x in [0,1,2] else 4.5)

df2 = df2.fillna(df.mean())

df2['Embarked'] = df2['Embarked'].apply(lambda x : x if (x=='C' or x=='Q') else 'S')

df2['Embarked'].unique()



le = LabelEncoder()



le.fit(df2['Sex'])

df2['Sex'] = le.transform(df2['Sex'])



le.fit(df2['Name'])

df2['Name'] = le.transform(df2['Name'])



le.fit(df2['Embarked'])

df2['Embarked'] = le.transform(df2['Embarked'])



X_sub = df2[features]

df2['Survived'] = model.predict(X_sub)

sub = pd.concat([df2['PassengerId'],df2['Survived']],axis = 1)

sub
sub.to_csv('Submission.csv',index=False)