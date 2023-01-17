# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv('../input/train.csv')

df_train.info()
df_train
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'])
def correlation(x,y):

    a=(x-x.mean())/x.std(ddof=0)

    b=(y-y.mean())/y.std(ddof=0)

    return (a*b).mean()
saleprice=df_train['SalePrice']

YearBuilt=df_train['YearBuilt']

LotFrontage=df_train['LotFrontage']   

LotArea=df_train['LotArea']

OverallQual=df_train['OverallQual']

YearRemodAdd=df_train['YearRemodAdd']

MasVnrArea=df_train['MasVnrArea']
correlation(saleprice,YearBuilt)
plt.scatter(df_train['YearBuilt'],df_train['SalePrice'])
correlation(saleprice,LotFrontage)
plt.scatter(df_train['LotFrontage'],df_train['SalePrice'])
correlation(saleprice,LotArea)
plt.scatter(df_train['LotArea'],df_train['SalePrice'])
correlation(saleprice,OverallQual)
plt.scatter(df_train['OverallQual'],df_train['SalePrice'])
correlation(saleprice,YearRemodAdd)
plt.scatter(df_train['YearRemodAdd'],df_train['SalePrice'])
correlation(saleprice,MasVnrArea)
plt.scatter(df_train['MasVnrArea'],df_train['SalePrice'])