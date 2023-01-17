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
df=pd.read_csv('/kaggle/input/yandextoloka-water-meters-dataset/WaterMeters/data.csv')
df.head()
df.tail()
df.isnull().sum()
df.info()
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

df['photo_name']=label.fit_transform(df['photo_name'])

df['location']=label.fit_transform(df['location'])



df.info()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot('location',data=df)

plt.title('Time Serie or Date')

plt.ylabel("location")

plt.show()
sns.boxplot(df.value,data=df)
sns.distplot(df.photo_name.dropna(),bins=8,color='red')
df['value']=df['value'].astype(int)

df.info()
from sklearn.model_selection import train_test_split

train,test=train_test_split(df,test_size=0.12,random_state=200)

def data_split(df):

    x=df.drop(['location'],axis=1)

    y=df['location']

    return x,y

x_train,y_train=data_split(train)

x_test,y_test=data_split(test)
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression



log_model=LogisticRegression()

log_model.fit(x_train,y_train)

prediction=log_model.predict(x_test)

score=accuracy_score(y_test,prediction)

print(score)
from sklearn.ensemble import RandomForestRegressor

regress=RandomForestRegressor()

regress.fit(x_train,y_train)

reg_train=regress.score(x_train,y_train)

reg_test=regress.score(x_test,y_test)

print(reg_train)

print(reg_test)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

smote = XGBClassifier()

smote.fit(x_train, y_train)



# Predict on test

smote_pred = smote.predict(x_test)

accuracy = accuracy_score(y_test, smote_pred)

print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))