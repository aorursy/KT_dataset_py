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


import numpy as np # linear algebra

import pandas as pd 
dt_wine=pd.read_csv("/kaggle/input/prediction-wine/winequality-red.csv")
dt_wine
dt_wine.mean()
dt_wine.columns
import  matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,10))

plt.tight_layout()

sns.distplot(dt_wine['quality'])
quality = dt_wine["quality"].values

status = []

for num in quality:

    if num<5:

        status.append("worst")

    elif num>6:

        status.append("best")

    else:

        status.append("better")
status= pd.DataFrame(data=status, columns=["status"])

data_wine = pd.concat([dt_wine,status],axis=1)

data_wine.drop(columns="quality",axis=1,inplace=True)
data_wine.head()
data_wine['status'].value_counts()[:20]
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

data_wine['status'] = labelencoder.fit_transform(data_wine['status'])
plt.figure(figsize=(9,6))

sns.heatmap(dt_wine.corr(),annot=True)##annot is like array in same shape
from scipy import stats

from scipy.stats import norm



sns.distplot(dt_wine['quality'], fit = norm)

fig = plt.figure()
X = data_wine.loc[:, data_wine.columns != 'status']

y = data_wine.loc[:, data_wine.columns == 'status']
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2,random_state=0)
from sklearn import metrics

import math

import numpy as np
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
l_model=lm.fit(xtrain,ytrain)
l_model
lm.score(xtrain, ytrain)
predict_wine= lm.predict(xtest)
print('MSE:', metrics.mean_squared_error(ytest, predict_wine))

print('RMSE:', np.sqrt(metrics.mean_squared_error(ytest, predict_wine)))
"""RMSE is 0.34, which is slightly greater than 7%ofthemean value which is 5.63(QUALITY). 

This means that our algorithm was not very accurate but can still make reasonably good predictions."""
###crossvalidate

from sklearn.model_selection import cross_val_score





scores = cross_val_score(l_model, xtrain, ytrain, cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")



lin_rmse_scores = np.sqrt(-scores)



lin_rmse_scores.mean()
from sklearn.linear_model import Ridge



ridge_reg = Ridge(alpha=0.05, solver="cholesky")

ridge_reg.fit(xtrain, ytrain)



ridge_reg.score(xtrain,ytrain)


from sklearn.metrics import mean_squared_error



ypredict_ridge = ridge_reg.predict(xtrain)



ridge_mse = mean_squared_error(ytrain, ypredict_ridge)

ridge_rmse = np.sqrt(ridge_mse)

ridge_rmse


from sklearn.model_selection import cross_val_score





scores_ridge = cross_val_score(ridge_reg, xtrain, ytrain, cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")



ridge_rmse_scores = np.sqrt(-scores_ridge)

ridge_rmse_scores.mean()
from sklearn.linear_model import Lasso



lasso_reg = Lasso(alpha=0.05, random_state = 42)

lasso_reg.fit(xtrain, ytrain)



lasso_reg.score(xtrain, ytrain)


from sklearn.metrics import mean_squared_error



ypredict_lasso = lasso_reg.predict(xtrain)



lasso_mse = mean_squared_error(ytrain, ypredict_lasso)

lasso_rmse = np.sqrt(lasso_mse)

lasso_rmse


from sklearn.model_selection import cross_val_score





scores_lasso = cross_val_score(lasso_reg,xtrain, ytrain, cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")



lasso_rmse_scores = np.sqrt(-scores_lasso)

lasso_rmse_scores.mean()
predicted = pd.DataFrame(ypredict_lasso,columns=["probability"],index=xtrain.index)
def recode(probability):

    if probability <= 0.5:

        return 0

    elif probability <= 1.0:

        return 1

    else:

        return 2
predicted['predict']=predicted['probability'].apply(recode)
predicted
res = pd.DataFrame(predicted)

res.index = xtrain.index

res.columns = ["predict","probability"]

res.to_csv("result.csv")