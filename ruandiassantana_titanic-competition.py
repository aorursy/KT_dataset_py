# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df2 = pd.read_csv('/kaggle/input/titanic/test.csv')
df.head()
df.info()
df.drop(['Ticket','Cabin'],axis=1,inplace=True)

df2.drop(['Ticket','Cabin'],axis=1,inplace=True)
df['Embarked'].nunique()
embarked = pd.get_dummies(df['Embarked'],drop_first=True)

embarked2 = pd.get_dummies(df2['Embarked'],drop_first=True)
df = pd.concat((df,embarked),axis=1)

df.drop('Embarked',axis=1,inplace=True)

df2 = pd.concat((df2,embarked2),axis=1)

df2.drop('Embarked',axis=1,inplace=True)
df.head()
sex = pd.get_dummies(df['Sex'],drop_first=True)

sex2 = pd.get_dummies(df2['Sex'],drop_first=True)
df = pd.concat((df,sex),axis=1)

df.drop('Sex',axis=1,inplace=True)

df2 = pd.concat((df2,sex2),axis=1)

df2.drop('Sex',axis=1,inplace=True)
df.head()
df.info()
df['Mr'] = df['Name'].apply(lambda x: 1 if ('Mr' in x) else 0)

df2['Mr'] = df2['Name'].apply(lambda x: 1 if ('Mr' in x) else 0)
df.head()
df.groupby('Mr')['Survived'].mean()
df.drop('Name',axis=1,inplace=True)

df2.drop('Name',axis=1,inplace=True)
df.info()
from sklearn.model_selection import train_test_split
df.fillna(df.mean(),inplace=True)
df.corr()['Survived'].sort_values()
df.isnull().sum().sort_values()
df.drop('PassengerId',axis=1,inplace=True)
df.corr()['Survived'].sort_values()
X = df.drop('Survived',axis=1)

y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))

print('\n')

print(confusion_matrix(y_test,predictions))
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,y_train)
predictions = svm.predict(X_test)
print(classification_report(y_test,predictions))

print('\n')

print(confusion_matrix(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(verbose=3)
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
print(classification_report(y_test,predictions))

print('\n')

print(confusion_matrix(y_test,predictions))
df2.fillna(df2.mean(),inplace=True)
df2.info()
df2.drop('PassengerId',axis=1,inplace=True)
rfc = RandomForestClassifier()
rfc.fit(X,y)
predictions = rfc.predict(df2)
df3 = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df3.head()
df3['Survived'] = predictions
df3.head()
df3.to_csv('predict.csv',index=False,line_terminator='\r\n')
pd.read_csv('/kaggle/working/predict.csv')