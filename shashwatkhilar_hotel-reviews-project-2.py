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
df=pd.read_csv('/kaggle/input/515k-hotel-reviews-data-in-europe/Hotel_Reviews.csv')

df.head()
import seaborn as sns

sns.set(rc={'figure.figsize':(20,20)})



sns.lineplot(x = df.Reviewer_Score, y = df.Average_Score)
import seaborn as sns

sns.set(style="darkgrid")

ax = sns.countplot(x="Average_Score", data=df)
import seaborn as sns

sns.scatterplot(x=df.Reviewer_Score,y=df.Average_Score)
df.info()
df.isnull().sum()
for col in df.columns:

    if(df[col].isnull().sum()!=0):

        if(df.dtypes[col] == 'object'):

            df[col].fillna(df[col].value_counts().idxmax(), inplace = True)

        if(df.dtypes[col] == 'int64'):

            df[col].fillna(df[col].mean(), inplace= True)

        if(df.dtypes[col] == 'float64'):

            df[col].fillna(df[col].value_counts().idxmax(), inplace= True)
for col in df.columns:

    if(df.dtypes[col] == 'object'):

        from sklearn.preprocessing import LabelEncoder

        label=LabelEncoder()

        df[col]=label.fit_transform(df[col])
for col in df.columns:

    if(df.dtypes[col] == 'float64'):

        df[col] = df[col].astype(int)
df.info()
from sklearn.model_selection import train_test_split

train, test=train_test_split(df, test_size=0.1, random_state=1)



def data_splitting(df):

    x=df.drop(['Reviewer_Score'], axis=1)

    y=df['Reviewer_Score']

    return x, y





x_train, y_train=data_splitting(train)

x_test, y_test=data_splitting(test)
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
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()

reg.fit(x_train , y_train)

reg_train = reg.score(x_train , y_train)

reg_test = reg.score(x_test , y_test)





print(reg_train)

print(reg_test)