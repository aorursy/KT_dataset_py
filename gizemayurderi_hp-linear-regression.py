# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

hp=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
hp_test= pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
hp.head()
hp_test.head()
hp_test.info()
hp.describe()
hp.info()
fig = plt.figure(figsize = (25,15))
sns.heatmap(hp.corr(), annot =True, fmt='.1f')
df1= hp[["OverallQual","GrLivArea", "FullBath", "TotalBsmtSF", "1stFlrSF", "GarageCars", "GarageArea","YearBuilt", "YearRemodAdd", "MasVnrArea","TotRmsAbvGrd",

"Fireplaces", "GarageYrBlt", "SalePrice"]]
df1
fig = plt.figure(figsize = (15,15))
sns.heatmap(df1.corr(), annot =True, fmt='.1f')
df1
df2=df1.drop(["GrLivArea", "GarageYrBlt", "1stFlrSF", "GarageCars"], axis=1)
df2.dropna(inplace=True)
Y =df2["SalePrice"]
X =df2.drop(["SalePrice"],axis =1)
X.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
lr.coef_
coeff_df = pd.DataFrame(lr.coef_,x_train.columns,columns=['Coefficient'])
coeff_df
y_pred = lr.predict(x_test)
lr.intercept_
lr.score(x_test,y_test)
from sklearn import metrics
from math import sqrt
print('MAE: {}'.format(metrics.mean_absolute_error(y_test, y_pred)))
print('MSE: {}'.format(metrics.mean_squared_error(y_test, y_pred)))
print('RMSE: {}'.format(sqrt(metrics.mean_squared_error(y_test, y_pred))))
print("R2: {}".format(metrics.r2_score(y_test,y_pred)))