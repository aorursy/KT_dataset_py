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
%matplotlib inline
import seaborn as sns
sns.set_style('whitegrid')
import statsmodels.api as sm
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

df=pd.read_csv("../input/titanic/train.csv")
d2f=pd.read_csv("../input/titanic/test.csv")

type(df)
df.describe()
df.info()
df.shape
df.isnull().sum()

df.head()

df['Sex'] = df['Sex'].astype('category')
df['Sex_cat'] = df['Sex'].cat.codes
d2f['Sex'] = d2f['Sex'].astype('category')
d2f['Sex_cat'] = d2f['Sex'].cat.codes

df['Embarked'] = df['Embarked'].astype('category')
df['Embarked'] = df['Embarked'].cat.codes
d2f['Embarked'] = d2f['Embarked'].astype('category')
d2f['Embarked'] = d2f['Embarked'].cat.codes

df
df=df.drop(['Name','Sex'],axis=1)
d2f=d2f.drop(['Name','Sex'],axis=1)

sns.catplot(x='Pclass',kind='count', data=df,hue='Survived')
#sns.catplot(x='Sex',kind='count', data=df,hue='Survived')
sns.catplot(x='Age',kind='count', data=df,hue='Survived')
sns.catplot(x='SibSp',kind='count', data=df,hue='Survived')
sns.catplot(x='Parch',kind='count', data=df,hue='Survived')


X_test=d2f.drop(['Ticket','Cabin','PassengerId'],axis=1)

X_train=df.drop(['Survived','Ticket','Cabin','PassengerId'],axis=1)
X_train


Y_train=df.Survived
Y_train=np.ravel(Y_train)
Y_train


X_train['Age']=X_train['Age'].fillna(X_train['Age'].mean())

X_test['Age']=X_test['Age'].fillna(X_train['Age'].mean())

X_test['Fare']=X_test['Fare'].fillna(X_train['Fare'].mean())
X_test.isnull().sum()

log_r=LogisticRegression()
log_r.fit(X_train,Y_train)
log_r.score(X_train,Y_train)

coeff_df=pd.DataFrame(zip(X_train.columns,np.transpose(log_r.coef_)))
coeff_df

class_predict=log_r.predict(X_test)

ndf=pd.DataFrame({'PassengerId':d2f['PassengerId'],'Survived':class_predict})

ndf.to_csv("submission2.csv",index=False)

import xgboost as xgb
mod=xgb.XGBClassifier()
mod.fit(X_train,Y_train)
mod.score(X_train,Y_train)

class_predict=log_r.predict(X_test)
ndf=pd.DataFrame({'PassengerId':d2f['PassengerId'],'Survived':class_predict})
ndf.to_csv("submission2.csv",index=False)
