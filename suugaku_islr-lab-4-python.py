import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from sklearn.linear_model import LinearRegression

from sklearn import metrics



# Note that if we wish to use R-style formulas, then we use statsmodels.formula.api

import statsmodels.api as sm

import statsmodels.formula.api as smf
auto_filepath = "../input/ISLR-Auto/Auto.csv"

Auto = pd.read_csv(auto_filepath, na_values = ["?"]).dropna()
np.random.seed(1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Auto["horsepower"], Auto["mpg"], test_size = 0.5)
reg = LinearRegression()

reg.fit(X_train.values.reshape(-1, 1), y_train)
metrics.mean_squared_error(y_test, reg.predict(X_test.values.reshape(-1, 1)))
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import PolynomialFeatures
quad_pipe = make_pipeline(PolynomialFeatures(degree = 2), LinearRegression())

quad_pipe.fit(X_train.values.reshape(-1, 1), y_train)

pred = quad_pipe.predict(X_test.values.reshape(-1, 1))

metrics.mean_squared_error(y_test, pred)
cube_pipe = make_pipeline(PolynomialFeatures(degree = 3), LinearRegression())

cube_pipe.fit(X_train.values.reshape(-1, 1), y_train)

pred = cube_pipe.predict(X_test.values.reshape(-1, 1))

metrics.mean_squared_error(y_test, pred)
np.random.seed(2)

X_train, X_test, y_train, y_test = train_test_split(Auto["horsepower"], Auto["mpg"], test_size = 0.5)
pipe = make_pipeline(PolynomialFeatures(degree = 1), LinearRegression())

pipe.fit(X_train.values.reshape(-1, 1), y_train)

pred = pipe.predict(X_test.values.reshape(-1, 1))

metrics.mean_squared_error(y_test, pred)
pipe = make_pipeline(PolynomialFeatures(degree = 2), LinearRegression())

pipe.fit(X_train.values.reshape(-1, 1), y_train)

pred = pipe.predict(X_test.values.reshape(-1, 1))

metrics.mean_squared_error(y_test, pred)
pipe = make_pipeline(PolynomialFeatures(degree = 3), LinearRegression())

pipe.fit(X_train.values.reshape(-1, 1), y_train)

pred = pipe.predict(X_test.values.reshape(-1, 1))

metrics.mean_squared_error(y_test, pred)
from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold
# Using LeaveOneOut cross-validation splitter explicitly

X = Auto["horsepower"].values.reshape(-1, 1)

y = Auto["mpg"]

reg = LinearRegression()

loo = LeaveOneOut()

cv_scores = cross_val_score(reg, X, y, scoring = "neg_mean_squared_error", cv = loo)

# Since cv_scores is an array of scores, need to compute the mean afterward

cv_scores.mean()
# Performing leave-one-out cross-validation by passing on the number of observations

# as the argument cv

cv_scores = cross_val_score(reg, X, y, scoring = "neg_mean_absolute_error", cv = X.shape[0])

cv_scores.mean()
X = Auto["horsepower"].values.reshape(-1, 1)

y = Auto["mpg"]

loo = LeaveOneOut()

cv_error = []

for i in range(1, 11):

    pipe = make_pipeline(PolynomialFeatures(degree = i), LinearRegression())

    cv_scores = cross_val_score(pipe, X, y, scoring = "neg_mean_squared_error", cv = loo)

    cv_error.append(abs(cv_scores.mean()))

cv_error
# Using 10-fold cross-validation by passing the argument cv = 10 to cross_val_score()

X = Auto["horsepower"].values.reshape(-1, 1)

y = Auto["mpg"]

cv_error = []

for i in range(1, 11):

    pipe = make_pipeline(PolynomialFeatures(degree = i), LinearRegression())

    cv_scores = cross_val_score(pipe, X, y, scoring = "neg_mean_squared_error", cv = 10)

    cv_error.append(abs(cv_scores.mean()))

cv_error
# Using 10-fold cross-validation by passing an instance of KFold with shuffle = True

# In this situation the value of random_state matters

X = Auto["horsepower"].values.reshape(-1, 1)

y = Auto["mpg"]

kfolds = KFold(n_splits = 10, shuffle = True, random_state = 1)

cv_error = []

for i in range(1, 11):

    pipe = make_pipeline(PolynomialFeatures(degree = i), LinearRegression())

    cv_scores = cross_val_score(pipe, X, y, scoring = "neg_mean_squared_error", cv = kfolds)

    cv_error.append(abs(cv_scores.mean()))

cv_error
portfolio_filepath = "../input/islr-lab-4/Portfolio.csv"

Portfolio = pd.read_csv(portfolio_filepath)
from sklearn.utils import resample
def alpha(data):

    X = data.X

    Y = data.Y

    return ((Y.var() - np.cov(X, Y)[0, 1])/(X.var() + Y.var() - 2*np.cov(X, Y)[0, 1]))
alpha(Portfolio)
sample = resample(Portfolio, n_samples = 100, random_state = 1)

alpha(sample)
np.random.seed(1)

boot_estimates = np.empty(1000)

for i in range(1000):

    sample = resample(Portfolio)

    boot_estimates[i] = alpha(sample)

print("Bootstrap estimated alpha:", boot_estimates.mean(), 

      "\nBootstrap estimated std. err.:", boot_estimates.std())
def fit_coefs(X, y, estimator):

    reg = estimator.fit(X, y)

    coefs = reg.coef_

    intercept = reg.intercept_

    return np.append(coefs, intercept)
X = Auto["horsepower"].values.reshape(-1, 1)

y = Auto["mpg"]

reg = LinearRegression()

pd.Series(fit_coefs(X, y, reg), index = ["horsepower", "intercept"])
np.random.seed(1)

sample = resample(Auto)

X = sample["horsepower"].values.reshape(-1, 1)

y = sample["mpg"]

reg = LinearRegression()

pd.Series(fit_coefs(X, y, reg), index = ["horsepower", "intercept"])
sample = resample(Auto)

X = sample["horsepower"].values.reshape(-1, 1)

y = sample["mpg"]

reg = LinearRegression()

pd.Series(fit_coefs(X, y, reg), index = ["horsepower", "intercept"])
np.random.seed(17)

reg = LinearRegression()

bootstrap_estimates = pd.DataFrame()

for i in range(1000):

    sample = resample(Auto)

    X = sample["horsepower"].values.reshape(-1, 1)

    y = sample["mpg"]

    coefs = pd.Series(fit_coefs(X, y, reg), index = ["horsepower", "intercept"], name = i)

    bootstrap_estimates = bootstrap_estimates.join(coefs, how = "right")

pd.DataFrame({"original": bootstrap_estimates.mean(axis = 1), "std. error": bootstrap_estimates.std(axis = 1)})
exog = sm.add_constant(Auto["horsepower"])

endog = Auto["mpg"]

mod = sm.OLS(endog, exog)

res = mod.fit()

print(res.summary())
res.bse
np.random.seed(17)

poly = PolynomialFeatures(degree = 2, include_bias = False)

reg = LinearRegression()

bootstrap_estimates = pd.DataFrame()

for i in range(1000):

    sample = resample(Auto)

    X = poly.fit_transform(sample["horsepower"].values.reshape(-1, 1))

    y = sample["mpg"]

    coefs = pd.Series(fit_coefs(X, y, reg), index = ["horsepower", "horsepower^2", "intercept"], name = i)

    bootstrap_estimates = bootstrap_estimates.join(coefs, how = "right")

pd.DataFrame({"original": bootstrap_estimates.mean(axis = 1), "std. error": bootstrap_estimates.std(axis = 1)})
poly = PolynomialFeatures(degree = 2)

exog = poly.fit_transform(Auto["horsepower"].values.reshape(-1, 1))

endog = Auto["mpg"]

mod = sm.OLS(endog, exog)

res = mod.fit()

print(res.summary())
res.bse