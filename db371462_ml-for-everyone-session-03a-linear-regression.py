%matplotlib inline
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import seaborn as sns
np.random.seed(10)



n_samples = 30



def true_fun(X):

    return np.cos(1.5 * np.pi * X)



X = np.sort(np.random.rand(n_samples))

noise_size = 0.1

y = true_fun(X) + np.random.randn(n_samples) * noise_size
np.random.rand(n_samples)
X.shape
plt.scatter(X, y)
linear_regression = LinearRegression()

linear_regression.fit(X.reshape((30, 1)), y)
print(linear_regression.intercept_)

print(linear_regression.coef_)
# equally spaced array of 100 values between 0 and 1, like the seq function in R

X_to_pred = np.linspace(0, 1, 100).reshape(100, 1)



preds = linear_regression.predict(X_to_pred)



plt.scatter(X, y)

plt.plot(X_to_pred, preds)

plt.show()
X**2
X2 = np.column_stack((X, X**2))

X2
linear_regression.fit(X2, y)
print(linear_regression.intercept_)

print(linear_regression.coef_)
# equally spaced array of 100 values between 0 and 1, like the seq function in R

X_p = np.linspace(0, 1, 100).reshape(100, 1)

X_to_pred = np.column_stack((X_p, X_p**2))



preds = linear_regression.predict(X_to_pred)



plt.scatter(X, y)

plt.plot(X_p, preds)

plt.show()
import statsmodels.api as sm

import statsmodels.formula.api as smf



np.random.seed(9876789)
# We load a datase compiled by A.M. Guerry in the 1830's looking at social factors like crime and literacy

# http://vincentarelbundock.github.io/Rdatasets/doc/HistData/Guerry.html

# In general, statsmodels can download any of the toy datasets provided in R, and provides

# the same documentation from within Python

dta = sm.datasets.get_rdataset("Guerry", "HistData", cache=True)

print(dta.__doc__)
original_df = dta.data

original_df.head()
# Now, let's select a subset of columns

subsetted_df = original_df[['Lottery', 'Literacy', 'Wealth', 'Region']]

subsetted_df.head(100)
df = dta.data[['Lottery', 'Literacy', 'Wealth', 'Region']].dropna()

df.head(100)
# Next, let's fit the model by using a formula, just as we can in R, then running .fit()

# We regress the amount of money bet on the lottery on literacy, wealth, region, and

# and interaction between literacy and wealth.

mod = smf.ols(formula='Lottery ~ Literacy + Wealth + Region + Literacy:Wealth', data=df)

res = mod.fit()

print(res.summary())
# Next, we add polynomial terms for wealth, i.e., wealth^2 and wealth^3

mod = smf.ols(formula='Lottery ~ Literacy + Wealth + I(Wealth ** 2.0) + I(Wealth ** 3.0) + Region + Literacy:Wealth', data=df)

res = mod.fit()

print(res.summary())
res = smf.ols(formula='Lottery ~ Literacy + Wealth + C(Region)', data=df).fit()

print(res.params) # Print estimated parameter values
print(res.bse) # Print standard errors for the estimated parameters
print(res.predict()) # Print fitted values