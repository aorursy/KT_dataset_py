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
import pandas as pd


dft=pd.read_csv('../input/my-dada-test/test.csv')
df=pd.read_csv('../input/my-data-train/train.csv')
dft.head()
df.head()
df.shape
dft.shape
df.info()
dft.info()
df.describe()
dft.describe()
df.isnull().sum()
dft.isnull().sum()

import seaborn as sns
import matplotlib.pyplot as plt
#% matplotlib inlinesns.heatmap(df.isnull())
sns.heatmap(df.isnull())
df.drop(columns="Cabin",axis=1,inplace=True)#inplace=True
df.head()
dft.drop(columns="Cabin",axis=1,inplace=True)#inplace=True
dft.head()
df.corr()
sns.heatmap(df.corr(),annot=True)
df.Age.fillna(df.Age.mean(),inplace=True)
dft.Age.fillna(dft.Age.mean(),inplace=True)
df.Embarked.fillna(df.Embarked.mode()[0],inplace=True)
dft.Fare.fillna(dft.Fare.mean(),inplace=True)
df.isna().sum()
dft.isna().sum()
sns.countplot(df.Survived,hue=df.Sex)#hue
sns.countplot(df.Survived,hue=df.Sex)

sns.distplot(df.Age)
sns.violinplot(df.Survived,df.Sex,hue=df.Pclass)
'''count plot
distplot
pairplot
violin plot 
box plot 
hist 
#hue --- categ'''
x1=df.copy()
y1=dft.copy()
df.Sex=pd.get_dummies(df.Sex,drop_first=True)
df.head()
dft.Sex=pd.get_dummies(dft.Sex,drop_first=True)
dft.head()
df.Embarked=pd.get_dummies(df.Embarked,drop_first=True)
dft.Embarked=pd.get_dummies(dft.Embarked,drop_first=True)
df.head()
dft.head()
x1.Embarked.unique()
df.Embarked.unique()
df.columns
remove=['PassengerId','Name','Ticket', 'Embarked']
df.drop(columns=remove,inplace=True)
dft.drop(columns=remove,inplace=True)
df.head()
dft.head()
from sklearn.model_selection import train_test_split
target=df.Survived#.values
df.drop(columns="Survived",inplace=True)
df.head()
df_train,df_test,target_train,target_test=train_test_split(df,target,test_size=.30,random_state=0)
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
lr=LogisticRegression()
lrcv=LogisticRegressionCV(cv=5,random_state=0)
lr.fit(df_train,target_train)
target.shape
result=lr.predict(df_test)
from sklearn.metrics import accuracy_score

accuracy_score(target_test,result)
plt.figure(figsize=(15,8))
sns.scatterplot(df.Age,df.Fare,hue=x1.Survived,alpha=.6)
lrcv.fit(df_train,target_train)
pred_df=lrcv.predict(df_test)
accuracy_score(target_test,pred_df)
from sklearn.model_selection import GridSearchCV
gscv=GridSearchCV(lr,
    {"C":[1,2,3]},
    cv=5,
    return_train_score=False,
)
gscv.fit(df_train,target_train)
pred_gs=gscv.predict(df_test)
accuracy_score(target_test,pred_gs)
