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
df=pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
df.head()
sum(df.duplicated())
df.info()
df=df.fillna(df.mean())
df.info()
df.describe()
df.nunique()
from sklearn import linear_model
reg = linear_model.LinearRegression()
df['intercept'] = 1
X=df.iloc[:,[2,4]].values
Y=df.iloc[:,12].values.reshape(-1,1)


reg = linear_model.LinearRegression()
reg.fit(Y,X)

r2 = reg.score(Y, X)
r2
X_new=df.iloc[:,[2,7]].values
Y_new=df.iloc[:,12].values.reshape(-1,1)
reg_new = linear_model.LinearRegression()
reg_new.fit(Y_new,X_new)
r2_new = reg.score(Y_new,X_new)
r2_new
X_n=df.iloc[:,[4,7]].values
Y_n=df.iloc[:,12].values.reshape(-1,1)
reg_n = linear_model.LinearRegression()
reg_n.fit(Y_n,X_n)
r2_n = reg.score(Y_n,X_n)
r2_n
print(X_n.shape)
print(Y_n.shape)
X_t=df.iloc[:,[2,4,7]].values
Y_t=df.iloc[:,12].values.reshape(-1,1)

print(X_t.shape)
print(Y_t.shape)
reg_t = linear_model.LinearRegression()
reg_t.fit(Y_t,X_t)
r2_t = reg.score(Y_t,X_t)
r2_t
from patsy import dmatrices
import os
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
y, x = dmatrices('mba_p ~ ssc_p + hsc_p', df, return_type='dataframe')

vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif['features'] = x.columns
print(vif)
import statsmodels.api as sm
ln = sm.OLS(Y, X)
result = ln.fit()
print(result.summary())