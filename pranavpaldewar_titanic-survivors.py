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
import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling
df_train=pd.read_csv('../input/titanic/train.csv')

df_train.head()
plt.figure(figsize=(10,6))

sns.distplot(df_train[df_train['Survived']==1]['Age'],label='Survivors')

sns.distplot(df_train[df_train['Survived']==0]['Age'],label='non survivors')

plt.legend()

plt.xlabel('Age')

plt.show()
df_train.info()
df_train.Embarked.value_counts()
df_train.SibSp.value_counts()
df_train.Cabin.value_counts()
pandas_profiling.ProfileReport(df_train)
def conv_sex(sex):

    if sex=='male':

        sex=1

    else:

        sex=0

    return sex

df_train.Sex=df_train.Sex.apply(lambda x: conv_sex(x))
df_train.head()
from sklearn.model_selection import train_test_split 
df_train=df_train.drop('Cabin',axis=1)
sns.countplot(df_train.Embarked,hue='Survived',data=df_train)
%pip install autoviz
from autoviz.AutoViz_Class import AutoViz_Class

avz=AutoViz_Class()
report_2=avz.AutoViz('../input/titanic/train.csv')
df_train.Age=df_train.fillna(df_train.Age.median())
df_train=df_train.dropna()
lis=['PassengerId','Name','Ticket']

df_train=df_train.drop(lis,axis=1)
df_train.head()
con=pd.get_dummies(df_train.Embarked)
con.head()
df_train=df_train.drop('Embarked',axis=1)
df_1=df_train.merge(con,on=con.index)

df_1.head()
y=df_1.Survived

x=df_1.drop('Survived',axis=1)

train_x,valid_x,train_y,valid_y=train_test_split(x,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=200,max_depth=3)

rfc.fit(train_x,train_y)

predict=rfc.predict(valid_x)

from sklearn.metrics import mean_squared_error as mse

mse(predict,valid_y)
from sklearn.svm import SVC



svc=SVC(kernel='rbf')

svc.fit(train_x,train_y)

predict_1=svc.predict(valid_x)
mse(predict_1,valid_y)
df_test=pd.read_csv('../input/titanic/test.csv')
df_test.head()
con_1=pd.get_dummies(df_test.Embarked)
df_test=df_test.drop(['PassengerId','Name','Cabin','Ticket','Embarked'],axis=1)

df_2=df_test.merge(con_1,on=con_1.index)
df_2.Age=df_test.Age.fillna(df_test.Age.median())
df_2=df_2.fillna(0)
df_2.head()
df_2.Sex=df_2.Sex.apply(lambda x:conv_sex(x))
df_2.head()
df_2.shape
test_prediction=rfc.predict(df_2)
test_prediction[0:190]
test=pd.read_csv('../input/titanic/test.csv')

test=test.PassengerId

test.head()
submission=pd.DataFrame({'PassangerId':test,'Survived':test_prediction})
submission.to_csv()