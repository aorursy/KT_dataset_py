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
train_df=pd.read_csv('/kaggle/input/titanic/train.csv')

test_df=pd.read_csv('/kaggle/input/titanic/test.csv')
train_df.head()

test_df.head()
final_df=pd.concat([train_df,test_df])
final_df
final_df.shape
final_df.corr()
final_df.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt

sns.heatmap(final_df.isnull())
def bar_charts(feature):

    survived=final_df[final_df['Survived']==1][feature].value_counts()

    dead=final_df[final_df['Survived']==0][feature].value_counts()

    new_df=pd.DataFrame([survived,dead])

    new_df.index=['Survived','Dead']

    new_df.plot(kind='bar',stacked=True,figsize=(10,5))
bar_charts('Sex')
bar_charts('Pclass')
bar_charts('Embarked')
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=final_df)
sns.countplot(x='Survived',hue='Sex',data=final_df)
sns.countplot(x='Survived',hue='Pclass',data=final_df)
sns.boxplot(x='Pclass',y='Age',data=final_df)
final_df['Age']=np.where(final_df['Pclass']==1,40,27)

test_df['Age']=np.where(test_df['Pclass']==1,40,27)
final_df
final_df.isnull().sum()
final_df=final_df.drop(['Cabin'],axis=1)

test_df=test_df.drop(['Cabin'],axis=1)
final_df
final_df=final_df.dropna()
final_df
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

final_df['Sex']=le.fit_transform(final_df['Sex'])

test_df['Sex']=le.fit_transform(test_df['Sex'])
final_df
final_df['Embarked'].unique()
embark=pd.get_dummies(final_df['Embarked'],drop_first=True)

embark_2=pd.get_dummies(test_df['Embarked'],drop_first=True)

final_df=final_df.drop(['Ticket','Name','Embarked'],axis=1)

test_df=test_df.drop(['Ticket','Name','Embarked'],axis=1)
test_df
test_df=pd.concat([test_df,embark_2],axis=1)

test_df
test_df.isnull().sum()
test_df=test_df.fillna(method='ffill')
test_df.isnull().sum()
final_df=pd.concat([final_df,embark],axis=1)
final_df
sns.heatmap(final_df.isnull())
x=final_df.drop(['Survived'],axis=1)

y=final_df['Survived']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

lr=LogisticRegression(C=3,max_iter=2000)

lr.fit(x_train,y_train)

pred_y=lr.predict(test_df)



submission=pd.DataFrame({'PassengerId':test_df['PassengerId'],

                        'Survived':pred_y})
submission.to_csv('submission.csv',index=False)



submission