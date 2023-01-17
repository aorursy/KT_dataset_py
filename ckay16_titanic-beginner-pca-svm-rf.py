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
import matplotlib as pyplot
#import pandas_profiling
data = pd.read_csv("../input/titanic/train.csv")
data.head()
data.info()
data.columns
data.isnull().sum()
data['Cabin'].nunique()
data['Ticket'].isnull().sum()
target = data['Survived']
data.drop(['Cabin','Name','Ticket','PassengerId','Survived'],axis=1,inplace=True)

data.head()
data.drop(['Parch','SibSp'],axis=1,inplace=True)
data['Pclass'].unique()
data.dtypes

data['Sex'] = data['Sex'].map({'male':0,'female':1})
data['Embarked'] = data['Embarked'].map({'C':0,'S':1,'Q':2})
data.dtypes
# pclass = pd.get_dummies(data['Pclass'],drop_first=True)
# sex = pd.get_dummies(data['Sex'],drop_first=True)
# embarked = pd.get_dummies(data['Embarked'],drop_first=True)

# data = pd.concat([data,sex,pclass,embarked],axis=1)
# data['Embarked'] = data['Embarked'].astype('str')
# data[''] = data['Sex'].astype('str')
# from sklearn.preprocessing import LabelEncoder 
# le = LabelEncoder() 
# data['Embarked'] = le.fit_transform(data['Embarked'])
# data['Sex'] = le.fit_transform(data['Sex'])
data.isnull().sum()
data['Age'].fillna(value=data['Age'].mean(),inplace=True)

data['Embarked'].fillna(value=0,inplace=True)
data.isnull().sum()
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
ct = ColumnTransformer([
        ('scaler', StandardScaler(), ['Age','Fare'])
    ], remainder='passthrough')
data = ct.fit_transform(data)
data
data = pd.DataFrame(data)
data

data.columns = ['Age','Fare','Pclass','Sex','Embarked']
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data,target,test_size=0.3,random_state=0)

train_x
test_x
from sklearn.decomposition import PCA
pca = PCA(n_components=None)
train_x_new = pca.fit_transform(train_x)
test_x_new = pca.transform(test_x)
eval_score = pca.explained_variance_ratio_
np.array_str(eval_score, precision=2, suppress_small=True)
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# train_x = pca.fit_transform(train_x)
# test_x = pca.transform(test_x)
# eval_score = pca.explained_variance_ratio_
# np.array_str(eval_score, precision=2, suppress_small=True)
from sklearn.svm import SVC
svm_model = SVC()

svm_model.fit(train_x,train_y)
svm_model.score(test_x,test_y)
test = pd.read_csv("../input/titanic/test.csv")
test.head()
pid = test['PassengerId']
test.drop(['Cabin','Name','Ticket','PassengerId','SibSp','Parch'],axis=1,inplace=True)

test.dtypes
test['Sex'] = test['Sex'].map({'male':0,'female':1})
test['Embarked'] = test['Embarked'].map({'C':0,'S':1,'Q':2})
# test['Embarked'] = test['Embarked'].astype('str')
# test['Sex'] = test['Sex'].astype('str')
# from sklearn.preprocessing import LabelEncoder 
# le = LabelEncoder() 
# test['Embarked'] = le.fit_transform(test['Embarked'])
# test['Sex'] = le.fit_transform(test['Sex'])
test.isnull().sum()
test['Age'].fillna(value=test['Age'].mean(),inplace=True)
test['Fare'].fillna(value=test['Fare'].mean(),inplace=True)
test.isnull().sum()
test.head()
#test = test[['Age','Fare','Pclass','Sex','SibSp','Parch','Embarked']]#
test.head()
test_data = ct.fit_transform(test)
test_data
test = pd.DataFrame(test_data)
test

# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# test = pca.fit_transform(test)
from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=1000,max_features='sqrt')
# Fit on training data
model.fit(train_x, train_y)
sc = model.score(test_x,test_y)
sc
rf_preds = model.predict(test)
rf_preds
submission =  pd.DataFrame({'PassengerId':pid,'Survived':rf_preds})
submission.to_csv('submission.csv',index=False)
submission
