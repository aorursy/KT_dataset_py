# special IPython command to prepare the notebook for matplotlib and other libraries
%matplotlib inline 

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

import seaborn as sns

# special matplotlib argument for improved plots
from matplotlib import rcParams
sns.set_style("whitegrid")
sns.set_context("poster")

from sklearn.datasets import load_boston
import pandas as pd

boston = load_boston()
boston.keys()
boston.data.shape
# Print column names
print(boston.feature_names)
# Print description of Boston housing data set
print(boston.DESCR)
bos = pd.DataFrame(boston.data)
bos.head()
bos.columns = boston.feature_names
bos.head()
print(boston.target.shape)
bos['PRICE'] = boston.target
bos.head()
bos.describe()
plt.scatter(bos.CRIM, bos.PRICE)
plt.xlabel("Per capita crime rate by town (CRIM)")
plt.ylabel("Housing Price")
plt.title("Relationship between CRIM and Price")
# your turn: describe relationship

# your turn: scatter plot between *RM* and *PRICE*

# your turn: scatter plot between *PTRATIO* and *PRICE*

# your turn: create some other scatter plots

sns.regplot(y="PRICE", x="RM", data=bos, fit_reg = True)
plt.hist(np.log(bos.CRIM))
plt.title("CRIM")
plt.xlabel("Crime rate per capita")
plt.ylabel("Frequencey")
plt.show()
#your turn

# Import regression modules
import statsmodels.api as sm
from statsmodels.formula.api import ols
# statsmodels works nicely with pandas dataframes
# The thing inside the "quotes" is called a formula, a bit on that below
m = ols('PRICE ~ RM',bos).fit()
print(m.summary())
# your turn
from sklearn.linear_model import LinearRegression
X = bos.drop('PRICE', axis = 1)

# This creates a LinearRegression object
lm = LinearRegression()
lm
# Look inside lm object
# lm.<tab>
# Use all 13 predictors to fit linear regression model
lm.fit(X, bos.PRICE)
# your turn

print('Estimated intercept coefficient: {}'.format(lm.intercept_))
print('Number of coefficients: {}'.format(len(lm.coef_)))
# The coefficients
pd.DataFrame({'features': X.columns, 'estimatedCoefficients': lm.coef_})[['features', 'estimatedCoefficients']]
# first five predicted prices
lm.predict(X)[0:5]
# your turn

print(np.sum((bos.PRICE - lm.predict(X)) ** 2))
print(np.sum((lm.predict(X) - np.mean(bos.PRICE))**2))
# your turn

# your turn
# Your turn.
