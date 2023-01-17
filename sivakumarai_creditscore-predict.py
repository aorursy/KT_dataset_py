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
#import libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.linear_model import Ridge, Lasso, ElasticNet

from sklearn import tree

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb



CreditTrain = pd.read_csv("/kaggle/input/credit-score-prediction/CreditScore_train.csv")

CreditTest = pd.read_csv("/kaggle/input/credit-score-prediction/CreditScore_test.csv")

Credit_df = pd.concat((CreditTrain,CreditTest),axis=0)

Credit_df

    

    
print(Credit_df.shape)

Credit_null = Credit_df.isnull().sum() / len(Credit_df) * 100 

Credit_null_df = pd.DataFrame({"columns":Credit_df.columns,"Percentage": Credit_null})

Credit_null_df = Credit_null_df[(Credit_null_df['Percentage'] > 0) & (Credit_null_df['Percentage'] <=10)]

#Credit_df.drop(Credit_null_df['columns'],axis=1,inplace=True)

#Credit_df.shape

Credit_null_df
#Less than 10% of missing values

Credit_pre = Credit_df[['x005','x272']]

display(Credit_pre.describe())

#Credit_pre = Credit_pre[Credit_pre.isnull().any(axis=1)]

Credit_pre1 = Credit_pre.copy()

for col in Credit_pre1.columns:

    Credit_pre1.dropna(inplace=True)

    ax = sns.distplot(Credit_pre1[col],bins=15)

    plt.show(ax)

    ax1 = sns.boxplot(Credit_pre1[col])

    plt.show(ax1)

    q25,q75 = np.percentile(Credit_pre1[col],25),np.percentile(Credit_pre1[col],75)

    IQR = q75 - q25

    print("25th Quartile:",q25)

    print("75th Quartile:",q75)

    print("IQR:",IQR)

    upper, lower = (q75 + (1.5 *IQR)), (q25 - (1.5 * IQR))

    print("Upper Outlier:",upper)

    print("Lower Outlier:", lower)

    Credit_pre1 = Credit_pre1[(Credit_pre1[col] > lower) & (Credit_pre1[col] < upper)]

    display(Credit_pre1.describe())

    ax = sns.distplot(Credit_pre1[col],bins=10)

    plt.show(ax)

    Credit_pre[col].fillna(Credit_pre[col].mean(), inplace=True)

    display(Credit_pre.isnull().sum())

    display(Credit_pre.describe())


