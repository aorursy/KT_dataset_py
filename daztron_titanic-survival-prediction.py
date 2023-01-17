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
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import (RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')

PassengerId=test['PassengerId']
train.head(3)
full_data=[train,test]

train['Has_cabin'] = train['Cabin'].apply(lambda x: 0 if type(x)==float else 1)
test['Has_cabin']=test['Cabin'].apply(lambda x: 0 if type(x)==float else 1)

for dataset in full_data:
    dataset['Family_size']=dataset['SibSp']+dataset['Parch']+1
for dataset in full_data:
    dataset['IsAlone']=0
    dataset.loc[dataset['Family_size']==1,'IsAlone']=1
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'],4)

for dataset in full_data:
    age_avg=dataset['Age'].median()
    age_std=dataset['Age'].std()
    age_null_count=dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg-age_std,age_avg+age_std,size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])]=age_null_random_list
    dataset['Age']=dataset['Age'].astype(int)
train['CategoricalAge']=pd.cut(train['Age'],5)

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1)
    return ""
for dataset in full_data:
    dataset['Title']=dataset['Name'].apply(get_title)
pd.crosstab(train['Title'],train['Sex'])
    
train['Title'].unique()

for dataset in full_data:
    dataset['Title']=dataset['Title'].replace(['Don', 'Rev', 'Dr','Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess','Jonkheer'],'Rare')
    dataset['Title']=dataset['Title'].replace('Mlle','Miss')
    dataset['Title']=dataset['Title'].replace('Mme','Mrs')
    dataset['Title']=dataset['Title'].replace('Ms','Miss')

print(train.info())
for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} )
    
    title_mapping={"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Title']=dataset['Title'].fillna(0)
    
    dataset["Embarked"]=dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']= 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    

    dataset.loc[ dataset['Age'] <= 16, 'Age']= 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']= 4

drop_elements = ['PassengerId','Name','SibSp','Cabin','Ticket']
train=train.drop(drop_elements,axis=1)
train=train.drop(['CategoricalFare','CategoricalAge'],axis=1)
test=test.drop(drop_elements,axis=1)

train.head()
train_y=train['Survived']
train_x=train.drop(['Survived'],axis=1)
from xgboost import XGBClassifier
my_model=XGBClassifier()
my_model.fit(train_x,train_y)
predicted=my_model.predict(test)

my_submission = pd.DataFrame({'PassengerId':PasengerId,'Survived': predicted})
my_submission.to_csv('submission.csv', index=False)