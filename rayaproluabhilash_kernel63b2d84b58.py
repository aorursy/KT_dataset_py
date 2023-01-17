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
df=pd.read_csv('/kaggle/input/nba-all-star-game-20002016/NBA All Stars 2000-2016 - Sheet1.csv')
df.head()
df.tail()
df.isnull().sum()
df.info()
df.Player.value_counts()
df.Pos.value_counts()
df.HT.value_counts()
df.Team.value_counts()
df.Nationality.value_counts()
df=df.drop(['Selection Type','NBA Draft Status'],axis=1)
df.info()
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

df['Player']=label.fit_transform(df['Player'])

df['Pos']=label.fit_transform(df['Pos'])

df['HT']=label.fit_transform(df['HT'])

df['Team']=label.fit_transform(df['Team'])

df['Nationality']=label.fit_transform(df['Nationality'])
df.info()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot('Team', data=df)

plt.title('Position List', fontsize=14)

plt.ylabel("Team")

plt.show()
sns.lineplot(x=df.WT,y=df.HT)
sns.boxplot(x=df.Pos,y=df.Year)
sns.scatterplot(x=df.Year,y=df.Player)
from sklearn.model_selection import train_test_split

train,test=train_test_split(df,test_size=0.1,random_state=1)
def data_splitting(df):

    x=df.drop(['HT'],axis=1)

    y=df['HT']

    return x,y



x_train,y_train=data_splitting(train)

x_test,y_test=data_splitting(test)
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression



log_model=LogisticRegression()

log_model.fit(x_train, y_train)

prediction=log_model.predict(x_test)

score= accuracy_score(y_test, prediction)

print(score)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

smote = XGBClassifier()

smote.fit(x_train, y_train)



# Predict on test

smote_pred = smote.predict(x_test)

accuracy = accuracy_score(y_test, smote_pred)

print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))