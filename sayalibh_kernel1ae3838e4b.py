# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train=pd.read_csv(r"/kaggle/input/airpressure/Folds5x2_pp.csv")

df_train.describe()

df_train.info()
df_train.isnull().sum()
for i in range(len(df_train.columns)):

    sns.distplot(df_train.iloc[:,i])

    plt.show()
sns.pairplot(df_train)
#normalizing data

from sklearn import preprocessing

df_train_nor=preprocessing.normalize(df_train)

df_train_nor

#converting array into dataframe

df_train_nor=pd.DataFrame(df_train_nor)

sns.pairplot(df_train_nor)
cor_nor=df_train_nor.corr()

cor_nor
x=df_train_nor.iloc[:,0:4]

y=df_train_nor.iloc[:,4]

x.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn import linear_model

lm=linear_model.LinearRegression()

model=lm.fit(X_train,y_train)

pred=lm.predict(X_train)



print(pred)
from sklearn.metrics import r2_score

print(r2_score(pred,y_train))



pred_test=lm.predict(X_test)

print(r2_score(pred_test,y_test))
import xgboost

from sklearn.model_selection import cross_val_score

#XGBoost regressor

import xgboost as xgb



xgbr=xgb.XGBRegressor()



xgbr.fit(X_train,y_train)

score=xgbr.score(X_test,y_test)

print(score)