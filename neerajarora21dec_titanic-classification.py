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
train_csv = pd.read_csv("/kaggle/input/titanic/train.csv")
train_csv.head()
train_csv.describe()
import matplotlib.pyplot as plt 
import seaborn as sns

%matplotlib inline
sns.countplot(x="Sex", data=train_csv)
train_csv.info()
train_csv.columns
train_csv.head()
train_csv.drop(['Cabin'],axis=1,inplace=True)
train_csv.isna().sum()
train_csv.dropna(inplace=True)
train_csv.info()
cat_feats = ['Sex']
final_train_data = pd.get_dummies(train_csv,columns=cat_feats,drop_first=True)
final_train_data.head()
sns.countplot(x="Embarked", data=final_train_data)
from sklearn.preprocessing import LabelEncoder
final_train_data['Embarked_encoded'] = LabelEncoder().fit_transform(final_train_data['Embarked'])
sns.countplot(x="Embarked_encoded", data=final_train_data)
final_train_data.drop(['Name','Ticket','Embarked','PassengerId'],1,inplace=True)
final_train_data.head()
sns.countplot(x="Survived", data=final_train_data)
final_train_data.info()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
test_csv = pd.read_csv("/kaggle/input/titanic/test.csv")
test_csv.head()
test_csv.drop(['Cabin'],axis=1,inplace=True)
test_csv.isna().sum()
final_test_data = pd.get_dummies(test_csv,columns=['Sex'],drop_first=True)
final_test_data.head()
final_test_data['Embarked_encoded'] = LabelEncoder().fit_transform(final_test_data['Embarked'])
final_test_data.drop(['Name','Ticket'],1,inplace=True)
final_test_data.head()
final_test_data.drop(['PassengerId','Embarked'],1,inplace=True)
final_test_data.head()
final_test_data.isna().sum()
final_test_data['Age'].fillna((final_test_data['Age'].mean()), inplace=True)
final_test_data.isna().sum()
final_test_data['Fare'].fillna((final_test_data['Fare'].mean()), inplace=True)
final_test_data.head()
final_test_data.head()
X = final_train_data.drop(['Survived'],axis=1)
y = final_train_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
rfc3 = RandomForestClassifier(n_estimators=1000)
rfc3.fit(X_train,y_train)
predictions3 = rfc3.predict(X_test)
print(classification_report(y_test,predictions3))
test_predictions = rfc3.predict(final_test_data)
test_predictions
final_output =  pd.DataFrame(data=test_predictions,columns=['Survived'])
final_output.head()
df2 = test_csv.join(final_output)
df2.head()
df3 = df2[['PassengerId','Survived']]
df3.head()
df3.to_csv('titanic_output.csv',index=False)
