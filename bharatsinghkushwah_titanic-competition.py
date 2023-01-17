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
path="../input/titanic/train.csv"
df=pd.read_csv(path)
df.head()
df.index=df["PassengerId"]
df=df.drop(['PassengerId'],axis=1)
df.head()
df['Pclass'].unique()
L=[]

for i in df['Sex']:

    if(i=='male'):

        L.append(0)

    else:

        L.append(1)
df['Sex']=L
df['Sex']
df['Age'].fillna(df['Age'].mean(),inplace=True)
df['Sex'].fillna(0)
df.isnull().sum()
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score 
feat_set=['Sex','Age','SibSp','Parch']
dtree=DecisionTreeClassifier(criterion="entropy")

X = df[feat_set]

Y=df['Survived']
dtree.fit(X,Y)
score=accuracy_score(dtree.predict(X),Y)
score
test_data=pd.read_csv("/kaggle/input/titanic/test.csv")
test_data
X_test=test_data[['Sex','Age','SibSp','Parch']]
test_data.count()
X_test.isnull().sum()
X_test['Age'].fillna(X_test['Age'].mean(),inplace=True)
L=[]

for i in X_test['Sex']:

    if(i=='male'):

        L.append(0)

    else:

        L.append(1)
X_test['Sex']=L
X_test.count()
X_test.isnull().sum()
y_res=dtree.predict(X_test)
y_res
gen_sub=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
gen_sub
test_data['Survived']=y_res
sub=test_data[['PassengerId','Survived']]
sub
sub.to_csv("submission",index=False)

print("Your submission was successfully saved!")