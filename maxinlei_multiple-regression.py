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
# imports

import pandas as pd

import numpy as np

import seaborn as sns

import pylab

import math

import matplotlib.pyplot as plt



from scipy import stats

import statsmodels.api as sm

from statsmodels.stats import diagnostic as diag

from statsmodels.stats.outliers_influence import variance_inflation_factor



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.model_selection import train_test_split
df1 = pd.read_csv("../input/cs98xspotifyclassification/CS98XClassificationTrain.csv")

df2 = pd.read_csv("../input/cs98xspotifyclassification/CS98XClassificationTest.csv")
df1.head()
df2.head()
data = pd.concat([df1, df2])

display(data)
data.drop(columns=["title","artist","top genre","year","Id"],inplace=True)
display(data)


data.dropna(inplace=True)

# Check for Perfect Multicollinearity

corr = data.corr()

display(corr)
sns.heatmap(corr,xticklabels = corr.columns, yticklabels = corr.columns,cmap = 'RdBu')
A = data

B = sm.tools.add_constant(data)

series_before = pd.Series([variance_inflation_factor(B.values, i) for i in range(B.shape[1])], index = B.columns)

print('DATA BEFORR')

print('_'*100)

display(series_before)
pd.plotting.scatter_matrix(B, alpha = 1, figsize = (30,20))

plt.show
data = data.drop(columns = "spch", axis = 0)
display(data)
D = data.describe()

print(D)
D.loc['+3_std'] = D.loc['mean'] + (D.loc['std'] * 3)

D.loc['-3_std'] = D.loc['mean'] - (D.loc['std'] * 3)

print(D)
data_remove = data[(np.abs(stats.zscore(data)) < 3).all(axis = 1)]

data.index.difference(data_remove.index)
X = data_remove.drop('pop', axis = 1)

Y = data_remove[['pop']]
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

X_scaled = std_scaler.fit_transform(X)
# Split dataset into training and testing portion

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.20,random_state=1)

# Creat an instance of our model

regression_model = LinearRegression()

regression_model.fit(X_train, y_train)

# Exloring the Output

intercept = regression_model.intercept_[0]

print("The intercept for  model is {:.4}".format(intercept))

print('_'*100)
print(regression_model.coef_)
y_predict = regression_model.predict(X_test)



y_predict[:150]
# Evaluation

E = sm.add_constant(X)



model = sm.OLS(Y,E)



est = model.fit()
import pylab
sm.qqplot(est.resid, line = 's')

pylab.show()



mean_residuals = sum(est.resid) / len(est.resid)

mean_residuals
import math

model_mse = mean_squared_error(y_test, y_predict)



model_mae = mean_absolute_error(y_test, y_predict)



model_rmse = math.sqrt(model_mse)



print("MSE {:.3}".format(model_mse))

print("MAE {:.3}".format(model_mae))

print("RMSE {:.3}".format(model_rmse))
model_r2 = r2_score(y_test, y_predict)

model_r2
est.pvalues
print(est.summary())