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
df=pd.read_csv('/kaggle/input/reliance-industries-ril-share-price-19962020/Reliance Industries 1996 to 2020.csv')
df
df.isnull().sum()
df.info()
#fill the missing values in categorical features

for i in list(df.columns):

    if df[i].dtype=='object':

        df[i]=df[i].fillna(df[i].mode()[0])
#fill the missing values in numerical features

for i in list(df.columns):

    if df[i].dtype!='object':

        df[i]=df[i].fillna(df[i].mean())
df.isnull().sum()
#now preprocess the categorical features

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in list(df.columns):

    if df[i].dtype=='object':

        df[i]=le.fit_transform(df[i])
df=df.drop(['Date'],axis=1)
df
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

y=df['Turnover']

x=df.drop(['Turnover'],axis=1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
x_train=ss.fit_transform(x_train)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

lr=LinearRegression()

lr.fit(x_train,y_train)
pred_1=lr.predict(ss.transform(x_test))

score_1=r2_score(y_test,pred_1)
score_1
import matplotlib.pyplot as plt

import seaborn as sns

sns.scatterplot(x=y_test,y=pred_1)
#seems a decent fit line
from sklearn.linear_model import Lasso

lasso=Lasso(max_iter=1000)

lasso.fit(x_train,y_train)

pred_2=lasso.predict(ss.transform(x_test))

score_2=r2_score(y_test,pred_2)
score_2
from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor()

rfr.fit(x_train,y_train)

pred_3=rfr.predict(ss.transform(x_test))

score_3=r2_score(y_test,pred_3)
score_3
sns.scatterplot(x=y_test,y=pred_3)
#seems to be quite good fit line
from sklearn.ensemble import GradientBoostingRegressor

gbr=GradientBoostingRegressor()

gbr.fit(x_train,y_train)

pred_4=gbr.predict(ss.transform(x_test))

score_4=r2_score(y_test,pred_4)
score_4