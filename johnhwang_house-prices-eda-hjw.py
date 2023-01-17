# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import missingno as msno

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
g = sns.jointplot(x = df_train['YrSold'], y = df_train['SalePrice'],kind="reg")
g = sns.jointplot(x = df_train['GarageYrBlt'], y = df_train['SalePrice'],kind="reg")
g = sns.jointplot(x = df_train['YearRemodAdd'], y = df_train['SalePrice'],kind="reg")
g = sns.jointplot(x = df_train['YearBuilt'], y = df_train['SalePrice'],kind="reg")
sns.pairplot(df_train, x_vars=["YrSold", "YearBuilt","YearRemodAdd","GarageYrBlt"], y_vars=["SalePrice"],

            size=5, aspect=.8, kind="reg");
from pandas.tools import plotting

plotting.scatter_matrix(df_train[['SalePrice',"YrSold", "YearBuilt","YearRemodAdd","GarageYrBlt"]]) 
time_df = df_train[["SalePrice","YrSold", "YearBuilt","YearRemodAdd","GarageYrBlt"]]
from statsmodels.formula.api import ols

model = ols("SalePrice ~ YrSold + YearBuilt + YearRemodAdd + GarageYrBlt", time_df).fit()
print(model.summary())