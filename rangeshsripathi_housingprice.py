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
df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df.describe()
import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
df.isnull().sum()
df.drop(['Alley'],axis=1,inplace=True)

df.drop(['GarageYrBlt'],axis=1,inplace=True)

df.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

categoricalcols=df.select_dtypes(exclude=['int64', 'float64']).columns

categoricalcols
df[categoricalcols]=df[categoricalcols].fillna(df.mode().iloc[0])

sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])

df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])

#Replace all Categorical values with Numerics make use of get_dummies 
traindf=pd.get_dummies(data=df, columns=categoricalcols)

traindf.drop(columns=['Id','SalePrice'],inplace=True,axis=1)

from sklearn.model_selection import train_test_split

X=traindf

y=df[['SalePrice']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression

LR =LogisticRegression() 
LR.fit(X_train,y_train)

LR.coef_
predict=LR.predict(X_test)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LR, X_train, 

         y_train, cv=5)

acc_score= np.mean(scores)

acc_score