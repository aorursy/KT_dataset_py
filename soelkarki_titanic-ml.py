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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data=pd.read_csv('/kaggle/input/titanic/train.csv')
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.head()
test_data.isna().sum()
train_data.info()
train_data.drop('Name',axis=1,inplace=True)
test_data.drop('Name',axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
train_data['Sex']=number.fit_transform(train_data['Sex'].astype('str'))
test_data['Sex']=number.fit_transform(test_data['Sex'].astype('str'))
train_data['Sex']
train_data.info()
train_data['Age'].isna().sum()
train_data['Age'].value_counts()
train_data['Age'].median()
train_data['Age'].fillna(28,inplace=True)
test_data['Age'].fillna(28,inplace=True)
train_data['Age'].isna().sum()
train_data.info()
train_data['Ticket']=number.fit_transform(train_data['Ticket'].astype('str'))
test_data['Ticket']=number.fit_transform(test_data['Ticket'].astype('str'))
train_data['Ticket']
train_data['Fare'].isna().sum()
test_data['Fare'].fillna(test_data['Fare'].median(),inplace=True)
train_data.dropna(subset=['Embarked'],inplace=True)
test_data.dropna(subset=['Embarked'],inplace=True)
train_data
train_data.drop('Cabin',axis=1,inplace=True)
test_data.drop('Cabin',axis=1,inplace=True)

train_data['Embarked'].isna().sum()
train_data['Embarked'].value_counts()
embark_train=pd.get_dummies(train_data['Embarked'],prefix='Embarked')
embark_test=pd.get_dummies(test_data['Embarked'],prefix='Embarked')

train_data=pd.concat([train_data,embark_train],axis=1)
test_data=pd.concat([test_data,embark_test],axis=1)

train_data.drop('Embarked',axis=1,inplace=True)
test_data.drop('Embarked',axis=1,inplace=True)
train_data.isna().sum()
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import SGDClassifier
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.pipeline import Pipeline
# from sklearn.neighbors import KNeighborsClassifier

x=train_data.drop('Survived',axis=1).values
y=train_data['Survived'].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# clf=KNeighborsClassifier(n_neighbors=8)

# clf.fit(x_train,y_train)

from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(x_train,y_train)
gbc.score(x_test,y_test)
test_data.isna().sum()
predict=gbc.predict(test_data)
predicted={'PassengerId':test_data['PassengerId'],'Survived':predict}
predicted_df=pd.DataFrame(predicted)
predicted_df.shape
predicted_df.to_csv('titanic.csv',index=False)
