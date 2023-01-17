# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset_train = pd.read_csv('../input/train.csv')

dataset_train.columns

dataset_train.head(10)
dataset_train.isnull().sum()

dataset_train = dataset_train.drop(columns = ['PassengerId','Name','Ticket','Cabin'],axis = 1)

dataset_train.columns
import seaborn as sns

sns.catplot(data = dataset_train,kind = 'box',y = 'Age',x = 'Sex',hue = 'Pclass')
sns.catplot(data = dataset_train,kind = 'count',x ='Pclass')

sns.catplot(data = dataset_train,kind = 'count',x ='Pclass',hue = 'Sex')
sns.catplot(data = dataset_train,kind = 'count',x ='Pclass',hue = 'Survived',col = 'Sex')

dataset_train[['Age']] = dataset_train[['Age']].fillna(method = 'ffill')

dataset_train[['Embarked']] = dataset_train[['Embarked']].fillna(method = 'ffill')           



y_train = dataset_train.iloc[:,0:1].values 

X_train = dataset_train.iloc[:,1:].values

from sklearn.preprocessing import LabelEncoder

lab = LabelEncoder()

X_train[:,1] = lab.fit_transform(X_train[:,1])

lab.classes_
X_train[:,6]=lab.fit_transform(X_train[:,6])

lab.classes_

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features = [1,6])

X_train = ohe.fit_transform(X_train)



X_train = X_train.toarray()

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 40)

rf.fit(X_train,y_train)

dataset_test = pd.read_csv('../input/test.csv')

dataset_test.isnull().sum()
passenger_id_df = dataset_test['PassengerId']

passenger_id_df = pd.DataFrame(passenger_id_df)



dataset_test = dataset_test.drop(columns = ['PassengerId','Name','Ticket','Cabin'],axis = 1)

dataset_test.isnull().sum()
dataset_test[['Age']] = dataset_test[['Age']].fillna(method = 'ffill')

dataset_test.isnull().sum()

from sklearn.impute import SimpleImputer

imp = SimpleImputer()

dataset_test[['Fare']] = imp.fit_transform(dataset_test[['Fare']])



dataset_test.isnull().sum()
X_test = dataset_test.values

#apply LabelEncoder

X_test[:,1] = lab.fit_transform(X_test[:,1]) 

lab.classes_

X_test[:,6]=lab.fit_transform(X_test[:,6])

lab.classes_



from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features = [1,6])

X_test = ohe.fit_transform(X_test)



X_test = X_test.toarray()





#apply StandardScaler

X_test = sc.fit_transform(X_test)
y_test_pred_rf = rf.predict(X_test)

rf.score(X_train,y_train)#96.96%

my_submission_file = pd.DataFrame({ 'PassengerId': passenger_id_df['PassengerId'],'Survived': y_test_pred_rf })

my_submission_file.to_csv("my_submission_file.csv", index=False)