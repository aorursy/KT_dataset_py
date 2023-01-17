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
df = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')

df.head(50)
df.iloc[3530:3539]
df.rating.value_counts()
df.listed_in.value_counts()
df['Genre']=df['listed_in']
df.info()
for val in df['Genre']:

    if(val.find(',')):

        val=val.split(",")

    

    
df.head()
value=int(input("Input index of movie less than 6234 \n"))

c=0

C=0

t=df['title']

g=df['Genre']

g=g[value]

r=df['rating']

R=df['rating']

r=r[value]

print("\n \n The movies/tv shows similar to \'"+t[value]+"\' are:\n \n \n")





if(C==0):

    for i in df['Genre']:

        c=c+1

        if((i==g) & (R[c]==r)):

            C=C+1

            print(t[c])      





if(C<=2):

    c=c+1

    for i in df['Genre']:

        c=c+1

        if g[0] in i:

            C=C+1

            print(t[c])
df.info()
for col in df.columns:

    if(df[col].isnull().sum()!=0):

        if(df.dtypes[col] == 'object'):

            df[col].fillna(df[col].value_counts().idxmax(), inplace = True)
for col in df.columns:

    if(df.dtypes[col] == 'object'):

        from sklearn.preprocessing import LabelEncoder

        label=LabelEncoder()

        df[col]=label.fit_transform(df[col])
df.info()
import seaborn as sns

sns.set(rc={'figure.figsize':(20,20)})



sns.lineplot(x = df.country, y = df.listed_in)
import seaborn as sns

sns.set(style="darkgrid")

ax = sns.countplot(x="listed_in", data=df)
from sklearn.model_selection import train_test_split

train, test=train_test_split(df, test_size=0.1, random_state=1)



def data_splitting(df):

    x=df.drop(['Genre'], axis=1)

    y=df['Genre']

    return x, y





x_train, y_train=data_splitting(train)

x_test, y_test=data_splitting(test)
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