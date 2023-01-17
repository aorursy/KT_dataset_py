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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

%matplotlib inline
import pandas as pd

dataset = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")

dataset.head(10)
dataset.dtypes
dataset = dataset.drop(["id","date"], axis = 1)
# plotting_context return a parameter dict to scale elements of the figure.

with sns.plotting_context("paper",font_scale=2.5):

    g = sns.pairplot(dataset[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], hue='bedrooms', palette='tab20',size=6)

    g.set(xticklabels=[]);
X = dataset.iloc[:,1:].values

y = dataset.iloc[:,0].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
import statsmodels.api as sm

def backwardElimination(x, SL):

    numVars = len(x[0])

    temp = np.zeros((21613,19)).astype(int)

    for i in range(0, numVars):

        regressor_OLS = sm.OLS(y, x).fit()

        maxVar = max(regressor_OLS.pvalues).astype(float)

        adjR_before = regressor_OLS.rsquared_adj.astype(float)

        if maxVar > SL:

            for j in range(0, numVars - i):

                if (regressor_OLS.pvalues[j].astype(float) == maxVar):

                    temp[:,j] = x[:, j]

                    x = np.delete(x, j, 1)

                    tmp_regressor = sm.OLS(y, x).fit()

                    adjR_after = tmp_regressor.rsquared_adj.astype(float)

                    if (adjR_before >= adjR_after):

                        x_rollback = np.hstack((x, temp[:,[0,j]]))

                        x_rollback = np.delete(x_rollback, j, 1)

                        print (regressor_OLS.summary())

                        return x_rollback

                    else:

                        continue

    regressor_OLS.summary()

    return x
SL = 0.05

X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17]]

X_Modeled = backwardElimination(X_opt, SL)