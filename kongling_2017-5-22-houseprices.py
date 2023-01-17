# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
data.describe().T
data.head().T
test.isnull().sum()
data=data.drop(['Alley','MiscFeature','PoolQC'],axis=1)
test=test.drop(['Alley','MiscFeature','PoolQC'],axis=1)
data.isnull().sum()
data.columns
data.corr()['SalePrice']
import seaborn as sns

import matplotlib.pyplot as plt

corr=data.corr()

plt.figure(figsize=(30,30))



sns.heatmap(corr,vmax=1,linewidths=0.01,

           square=True,annot=True,cmap='YlGnBu',linecolor='blue')

plt.title('Correlation between features')
data=data.drop(['OverallCond','BsmtFinSF2','LowQualFinSF','BsmtHalfBath','3SsnPorch','YrSold'],axis=1)

test=test.drop(['OverallCond','BsmtFinSF2','LowQualFinSF','BsmtHalfBath','3SsnPorch','YrSold'],axis=1)
data.corr()['SalePrice'].abs()
data.describe(include='all').T