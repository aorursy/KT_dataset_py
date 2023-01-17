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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import math as m

from sklearn import linear_model

from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import r2_score as rsq

from scipy import stats

from statistics import mean

import statsmodels.formula.api as sm
os.getcwd()
data = pd.read_csv("../input/country.csv")

data
X = np.array(data["Gini_Index"])

y = np.array(data["Corruption_Index"])



X = X.reshape((-1,1))

y = y.reshape((-1,1))



plt.scatter(X,y)

plt.show()
lr = linear_model.LinearRegression(normalize = True)
lr.fit(X,y)
pred = lr.predict(X)
plt.plot(X,pred,color = "r")

plt.scatter(X,y)

plt.show()
residuals = y - pred

residuals = residuals.reshape((-1,))

Residuals = list(residuals**2)

size = len(Residuals)

size
rmse = m.sqrt(mean(Residuals))

print(rmse)

r_sq = rsq(y, pred)

print(r_sq)
X_try = X.reshape((-1,))

y_try = y.reshape((-1,))
slope, intercept, r_value, p_value, std_err = stats.linregress(X_try, y_try)
std_err
r_value**2
formula_str = "Corruption_Index ~ Gini_Index"
model=sm.ols(formula_str,data)
fitted = model.fit()
print(fitted.summary())
plt.scatter(X,fitted.resid,color='blue')

xmin = min(X)

xmax = max(X)

plt.hlines(y=0,xmin=xmin*0.9,xmax=xmax*1.1,color='red',linestyle='--',lw=3)

plt.show()
plt.scatter(x=fitted.fittedvalues,y=fitted.resid,edgecolor='k')

xmin=min(fitted.fittedvalues)

xmax = max(fitted.fittedvalues)

plt.hlines(y=0,xmin=xmin*0.9,xmax=xmax*1.1,color='red',linestyle='--',lw=3)

plt.show()



# all line are horizontal thats why homoscedacity
plt.hist(fitted.resid_pearson,bins=20,edgecolor='k')

plt.show()
from statsmodels.graphics.gofplots import qqplot
plt.figure(figsize=(8,5))

fig=qqplot(fitted.resid_pearson,line='45',fit='True')

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.show()
from scipy.stats import shapiro
_,p=shapiro(fitted.resid)
if p<0.01:

    print("The residuals seem to come from Gaussian process")

else:

    print("The normality assumption may not hold")
from statsmodels.stats.outliers_influence import OLSInfluence as influence
inf=influence(fitted)
(c, p) = inf.cooks_distance

plt.figure(figsize=(12,12))

plt.stem(np.arange(len(c)), c, markerfmt=",")

plt.show()