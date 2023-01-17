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
df=pd.read_csv('../input/ipl-data/IPL2013.csv')
df.head()
df.info()
df.describe()
data=df.loc[:5,df.columns[:10]]
data
import matplotlib.pyplot as plt

import seaborn as sns
drop = ["Sl.NO.","PLAYER NAME", "TEAM"]

df.drop(drop, axis=1, inplace=True)

df.isnull().sum()
features = list(df.columns)

cat_feat = ['COUNTRY', 'TEAM', 'PLAYING ROLE']

num_feat = []

for val in features:

    if val not in cat_feat:

        num_feat.append(val)

        

print(num_feat)
df1 = df[num_feat]

corr_mat = df1.corr()

plt.subplots(figsize=(12,10))

sns.heatmap(corr_mat, cmap="coolwarm", linewidths = 0.1)
df = pd.get_dummies(df)

df.columns
from statsmodels.stats.outliers_influence import variance_inflation_factor
def vif(x):

    x_matrix = x.as_matrix()

    vif = [variance_inflation_factor(x_matrix,i) for i in range(x_matrix.shape[1])]

    vif_factors = pd.DataFrame()

    vif_factors['column'] = x.columns

    vif_factors['vif'] = vif

    return vif_factors



predictors = df.copy()

predictors_copy = predictors[num_feat]

predictors.drop(["SOLD PRICE"], axis=1, inplace=True)

predictors_copy.drop(["SOLD PRICE"], axis=1, inplace=True)

vif_factors = vif(predictors_copy)

print(vif_factors.sort_values(by='vif'))
data=df[['CAPTAINCY EXP','ODI-SR-BL']]
sns.heatmap(data.corr())
data=df[['T-WKTS','ECON']]

sns.heatmap(data.corr(),cmap='coolwarm')
from statsmodels.regression.linear_model import OLS
predictors.columns
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
model=LinearRegression()

x_train,x_test,y_train,y_test=train_test_split(predictors.drop('BASE PRICE',axis=1),df['SOLD PRICE'],test_size=0.2)
model.fit(x_train,y_train)
preds=model.predict(x_test)
from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(y_test, preds)
print("error: ",error)