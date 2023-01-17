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
df = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
df.info()
df.isnull().sum()
del df['Year']
df.isnull().sum()
df.Publisher.value_counts()
df.Publisher.fillna('Electronic Arts' , inplace = True)
df.isnull().sum()
df.info()
from sklearn.preprocessing import LabelEncoder

label=LabelEncoder()

df['Name']=label.fit_transform(df['Name'])

df['Platform']=label.fit_transform(df['Platform'])

df['Genre']=label.fit_transform(df['Genre'])

df['Publisher']=label.fit_transform(df['Publisher'])
df.info()
df['Global_Sales'] = df['Global_Sales'].astype(int)
df.info()
df =df.drop(['NA_Sales', 'EU_Sales', 'JP_Sales','Other_Sales'], axis=1)

df.info()
from sklearn.model_selection import train_test_split

train, test=train_test_split(df, test_size=0.1, random_state=1)



def data_splitting(df):

    x=df.drop(['Global_Sales'], axis=1)

    y=df['Global_Sales']

    return x, y



x_train, y_train=data_splitting(train)

x_test, y_test=data_splitting(test)
from sklearn.linear_model import LinearRegression

log = LinearRegression()

log.fit(x_train , y_train)

log_train = log.score(x_train , y_train)

log_test = log.score(x_test , y_test)



print("Training score :" , log_train)

print("Testing score :" , log_test)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

smote = XGBClassifier()

smote.fit(x_train, y_train)



# Predict on test

smote_pred = smote.predict(x_test)

accuracy = accuracy_score(y_test, smote_pred)

print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))