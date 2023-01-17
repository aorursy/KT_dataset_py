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
df=pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
df.head()
df.tail()
df.isnull().sum()
df=df.drop(['race/ethnicity','test preparation course'],axis=1)

df.info()
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

df['gender']=label.fit_transform(df['gender'])

df['parental level of education']=label.fit_transform(df['parental level of education'])

df['lunch']=label.fit_transform(df['lunch'])
df.info()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot('gender', data=df)

plt.title('Gender', fontsize=14)

plt.ylabel("gender")

plt.show()
sns.lineplot(x=df.gender,y=df.lunch)
sns.scatterplot(x=df.gender,y=df.lunch)
sns.boxplot(x=df.gender,y=df.lunch)
sns.swarmplot(x=df.gender,y=df.lunch)
sns.stripplot(x=df.gender,y=df.lunch)
sns.jointplot(x=df.gender,y=df.lunch)
sns.dogplot()
sns.distplot(df.lunch.dropna(),bins=8,color='pink')
sns.distplot(df.gender.dropna(),bins=8,color='pink')
from sklearn.model_selection import train_test_split

train,test=train_test_split(df,test_size=0.1,random_state=1)
def data_splitting(df):

    x=df.drop(['lunch'],axis=1)

    y=df['lunch']

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
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()

reg.fit(x_train , y_train)

reg_train = reg.score(x_train , y_train)

reg_test = reg.score(x_test , y_test)





print(reg_train)

print(reg_test)