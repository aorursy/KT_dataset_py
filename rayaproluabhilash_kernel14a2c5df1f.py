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
df=pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
df.head()
df.tail()
df.isnull().sum()
df.Year.value_counts()
df.Publisher.value_counts()
df.Year.fillna("1431",inplace=True)

df.Publisher.fillna("1351",inplace=True)



df.isnull().sum()
df=df.drop(['Year','EU_Sales','JP_Sales','Other_Sales'],axis=1)

df.info()
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

df['Name']=label.fit_transform(df['Name'])

df['Platform']=label.fit_transform(df['Platform'])

df['Genre']=label.fit_transform(df['Genre'])

df['Publisher']=label.fit_transform(df['Publisher'])
df.info()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot('Genre', data=df)

plt.title('Genre List', fontsize=14)

plt.ylabel("Genre")

plt.show()
sns.lineplot(x=df.Platform,y=df.Publisher)
sns.scatterplot(x=df.Genre,y=df.Publisher)
sns.boxplot(x=df.Genre,y=df.Platform)
sns.swarmplot(x=df.Genre,y=df.Publisher)
sns.stripplot(x=df.Platform,y=df.Publisher)
sns.jointplot(x=df.Publisher,y=df.Genre)
sns.distplot(df.Platform.dropna(),bins=8,color='green')
from sklearn.model_selection import train_test_split

train,test=train_test_split(df,test_size=0.1,random_state=1)
def data_splitting(df):

    x=df.drop(['Genre'],axis=1)

    y=df['Genre']

    return x,y



x_train,y_train=data_splitting(train)

x_test,y_test=data_splitting(test)
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression



log_model=LogisticRegression()

log_model.fit(x_train,y_train)

prediction=log_model.predict(x_test)

score=accuracy_score(y_test,prediction)

print(score)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

smote = XGBClassifier()

smote.fit(x_train, y_train)



# Predict on test

smote_pred = smote.predict(x_test)

accuracy = accuracy_score(y_test, smote_pred)

print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))