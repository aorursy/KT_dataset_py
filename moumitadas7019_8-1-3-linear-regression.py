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

plt.scatter(bos.RM, bos.PRICE)
plt.xlabel("Average number of rooms per dwelling (RM)")
plt.ylabel("Housing Price")
plt.title("Relationship between RM and Price")
# your turn: scatter plot between *PTRATIO* and *PRICE*
plt.scatter(bos.PTRATIO, bos.PRICE)
plt.xlabel("Pupil-to-Teacher Ratio (PTRATIO)")
plt.ylabel("Housing Price")
plt.title("Relationship between PTRATIO and Price")
# your turn: create some other scatter plots

plt.scatter(bos.AGE, bos.TAX)
plt.xlabel("proportion of owner-occupied units built prior to 1940: AGE")
plt.ylabel("full-value property-tax rate per $10,000: TAX")
plt.title("Relationship between AGE and TAX")
sns.regplot(y="PRICE", x="RM", data=bos, fit_reg = True)
plt.hist(np.log(bos.CRIM))
plt.title("CRIM")
plt.xlabel("Crime rate per capita")
plt.ylabel("Frequencey")
plt.show()
#In the above histogram, we took the logarithm of the crime rate per capita. Repeat this histogram without taking the log.

plt.hist(bos.CRIM)
plt.title("CRIM")
plt.xlabel("Crime rate per capita")
plt.ylabel("Frequency")
plt.show()
x = [bos.RM, bos.PTRATIO, bos.AGE, bos.TAX]
plt.hist(x, bins = 10)
#plt.hist(bos.RM, bos.PTRATIO)
plt.title("RM vs PTRATIO")
plt.xlabel("RM")
plt.ylabel("PTRATIO")
plt.show()
# statsmodels works nicely with pandas dataframes
# The thing inside the "quotes" is called a formula, a bit on that below
# Import regression modules
# ols - stands for Ordinary least squares, we'll use this
import statsmodels.api as sm
from statsmodels.formula.api import ols
m = ols('PRICE ~ RM',bos).fit()
print(m.summary())
# your turn
plt.scatter(bos['PRICE'], m.fittedvalues)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted Prices: $Y_i$ vs $\hat{Y}_i$")
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
#Exercise: How would you change the model to not fit an intercept term? Would you recommend not having an intercept? Why or why not? For more information on why to include or exclude an intercept, look [here](https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faq-what-is-regression-through-the-origin/).

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
