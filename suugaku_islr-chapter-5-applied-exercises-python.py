# Import standard Python data science libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Import classes from scikit-learn for logistic regression, least squares linear regression

# Import OneHotEncoder and PolynomialFeatures for data pre-processing

# Import Pipeline, ColumnTransformer to encapsulate pre-processing heterogenous data and fitting

# into a single estimator

# Import train_test_split, cross_val_score, LeaveOneOut, KFold for model validation

# Import resample for bootstrapping

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, KFold

from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.utils import resample



# Load StatsModels API

# Note that if we wish to use R-style formulas, then we would use the StatsModels Formula API

import statsmodels.api as sm

import statsmodels.formula.api as smf
default_filepath = "../input/default-credit-card/attachment_default.csv"

Default = pd.read_csv(default_filepath)

Default.head()
X = Default[["balance", "income"]]

y = Default["default"]

clf = LogisticRegression(penalty = "none", solver = "lbfgs")

clf.fit(X, y)

clf.coef_
X_train, X_test, y_train, y_test = train_test_split(Default[["balance", "income"]], Default["default"],

                                                   test_size = 0.25, random_state = 312)

clf = LogisticRegression(penalty = "none", solver = "lbfgs")

clf.fit(X_train, y_train)

1 - clf.score(X_test, y_test)
X_train, X_test, y_train, y_test = train_test_split(Default[["balance", "income"]], Default["default"],

                                                   test_size = 0.25, random_state = 456)

clf = LogisticRegression(penalty = "none", solver = "lbfgs")

clf.fit(X_train, y_train)

1 - clf.score(X_test, y_test)
X_train, X_test, y_train, y_test = train_test_split(Default[["balance", "income"]], Default["default"],

                                                   test_size = 0.25, random_state = 789)

clf = LogisticRegression(penalty = "none", solver = "lbfgs")

clf.fit(X_train, y_train)

1 - clf.score(X_test, y_test)
X_train, X_test, y_train, y_test = train_test_split(Default[["balance", "income"]], Default["default"],

                                                   test_size = 0.25, random_state = 314159)

clf = LogisticRegression(penalty = "none", solver = "lbfgs")

clf.fit(X_train, y_train)

1 - clf.score(X_test, y_test)
(Default["default"] != "No").mean()
np.random.seed(312)

with_student = {}

without_student = {}



# Create two classifier pipelines

# with_student takes the student variable and encodes it using one hot encoding, passes through income, balance

# without_student drops the student variable and only passes through income and balance

categorical_features = ["student"]

categorical_transformer = Pipeline([("onehot", OneHotEncoder(drop = "first"))])

numerical_features = ["income", "balance"]

with_student_preprocessor = ColumnTransformer([("cat", categorical_transformer, categorical_features),

                                 ("num", "passthrough", numerical_features)])

with_student_clf = Pipeline([("preprocessor", with_student_preprocessor), 

                ("classifier", LogisticRegression(penalty = "none", solver = "lbfgs"))])

without_student_preprocessor = ColumnTransformer([("num", "passthrough", numerical_features)])

without_student_clf = Pipeline([("preprocessor", without_student_preprocessor), 

                ("classifier", LogisticRegression(penalty = "none", solver = "lbfgs"))])



# Loop through 50 train-test splits to compute average difference in error rate

for i in range(50):

    # Split the data in to training and test sets

    X_train, X_test, y_train, y_test = train_test_split(Default, Default["default"], test_size = 0.25)

    # Fit classifier which includes student variable and compute validation set error

    with_student_clf.fit(X_train, y_train)

    with_student[i] = 1 - with_student_clf.score(X_test, y_test)

    # Fit classifier which excludes student variable and compute validation set error

    without_student_clf.fit(X_train, y_train)

    without_student[i] = 1 - without_student_clf.score(X_test, y_test)

errors = pd.DataFrame({"with_student": with_student, "without_student": without_student})

errors["difference"] = errors["with_student"] - errors["without_student"]

errors["difference"].mean()
# Using the Logit class from StatsModels

# First encode response numerically

endog = (Default["default"] == "Yes").astype(int)

exog = sm.add_constant(Default[["income", "balance"]])

mod = sm.Logit(endog, exog)

res = mod.fit()
res.bse["income"]
res.bse["balance"]
def boot_fn(data, clf):

    clf.fit(data[["income", "balance"]], data["default"])

    return clf.coef_
np.random.seed(17)

num_estimates = 10000

boot_estimates = np.empty((num_estimates, 2))

for i in range(num_estimates):

    clf = LogisticRegression(penalty = "none", solver = "lbfgs")

    sample = resample(Default)

    coefs = boot_fn(sample, clf)

    boot_estimates[i, 0] = coefs[0, 0]

    boot_estimates[i, 1] = coefs[0, 1]
boot_df = pd.DataFrame(boot_estimates, columns = ["income", "balance"])
boot_df["income"].std()
boot_df["balance"].std()
np.random.seed(17)

num_estimates = 10000

boot_estimates = np.empty((num_estimates, 2))

for i in range(num_estimates):

    sample = resample(Default)

    endog = (sample["default"] == "Yes").astype(int)

    exog = sm.add_constant(sample[["income", "balance"]])

    mod = sm.Logit(endog, exog)

    res = mod.fit(disp = False)

    boot_estimates[i, 0] = res.params["income"]

    boot_estimates[i, 1] = res.params["balance"]

sm_df = pd.DataFrame(boot_estimates, columns = ["income", "balance"])
sm_df["income"].std()
sm_df["balance"].std()
coef_comparison = boot_df.join(sm_df, rsuffix = "_sm")

coef_comparison.head()
coef_comparison[np.isclose(coef_comparison["income"], coef_comparison["income_sm"])].shape
close_coefs = np.isclose(coef_comparison["income"], coef_comparison["income_sm"])
coef_comparison[~close_coefs].mean()
coef_comparison.loc[~close_coefs, "income"].std()
coef_comparison.loc[~close_coefs, "income_sm"].std()
coef_comparison.loc[~close_coefs, "balance"].std()
coef_comparison.loc[~close_coefs, "balance_sm"].std()
coef_comparison.loc[close_coefs, "income"].std()
coef_comparison.loc[close_coefs, "income_sm"].std()
coef_comparison.loc[close_coefs, "balance"].std()
coef_comparison.loc[close_coefs, "balance_sm"].std()
weekly_filepath = "../input/islr-weekly/Weekly.csv"

weekly = pd.read_csv(weekly_filepath)

weekly.head()
X = weekly[["Lag1", "Lag2"]]

y = weekly["Direction"]

clf = LogisticRegression(penalty = "none", solver = "lbfgs")

clf.fit(X, y)

print(clf.coef_, clf.intercept_)
X_loo = weekly.loc[weekly.index != 0, ["Lag1", "Lag2"]]

y_loo = weekly.loc[weekly.index != 0, "Direction"]

clf = LogisticRegression(penalty = "none", solver = "lbfgs")

clf.fit(X_loo, y_loo)

print(clf.coef_, clf.intercept_)
clf.predict_proba(weekly.loc[0, ["Lag1", "Lag2"]].to_frame().T)
clf.predict(weekly.loc[0, ["Lag1", "Lag2"]].to_frame().T)
weekly.iloc[0]
n = weekly.shape[0]

scores = np.empty(n)

clf = LogisticRegression(penalty = "none", solver = "lbfgs")

for i in range(n):

    X_loo = weekly.loc[weekly.index != i, ["Lag1", "Lag2"]]

    y_loo = weekly.loc[weekly.index != i, "Direction"]

    clf.fit(X_loo, y_loo)

    scores[i] = clf.score(weekly.loc[i, ["Lag1", "Lag2"]].to_frame().T, pd.Series(weekly.loc[i, "Direction"]))
scores.mean()
errors = 1 - scores

errors.mean()
# Using cross_val_score to compare results with LeaveOneOut splitter

X = weekly[["Lag1", "Lag2"]]

y = weekly["Direction"]

clf = LogisticRegression(penalty = "none", solver = "lbfgs")

loo = LeaveOneOut()

cv_scores = cross_val_score(clf, X, y, cv = loo)

cv_scores.mean()
1 - cv_scores.mean()
(weekly["Direction"] != "Up").mean()
np.random.seed(312)

x = np.random.normal(size = 100)

y = x - 2*x**2 + np.random.normal(size = 100)
fig, ax = plt.subplots(figsize = (10, 8))

ax.scatter(x, y)



#Plot parabola used to model the data set

x_range = np.linspace(start = -2.5, stop = 2.5)

parabola = x_range - 2*x_range**2

ax.plot(x_range, parabola, color = "orange", linestyle = "--");
np.random.seed(312)

loocv_scores = pd.Series()

for i in range(1, 5):

    poly_reg = Pipeline([("poly", PolynomialFeatures(degree = i)), ("reg", LinearRegression())])

    loo = LeaveOneOut()

    cv_scores = cross_val_score(poly_reg, x.reshape(-1, 1), y, scoring = "neg_mean_squared_error", cv = loo)

    loocv_scores.loc[i] = abs(cv_scores.mean())

loocv_scores
np.random.seed(42)

loocv_scores = pd.Series()

for i in range(1, 5):

    poly_reg = Pipeline([("poly", PolynomialFeatures(degree = i)), ("reg", LinearRegression())])

    loo = LeaveOneOut()

    cv_scores = cross_val_score(poly_reg, x.reshape(-1, 1), y, scoring = "neg_mean_squared_error", cv = loo)

    loocv_scores.loc[i] = abs(cv_scores.mean())

loocv_scores
poly = PolynomialFeatures(degree = 1, include_bias = True)

# Orthogonalize the powers of the predictor using QR decomposition

ortho_X = np.linalg.qr(poly.fit_transform(x.reshape(-1, 1)))[0][:, 1:]

exog = sm.add_constant(ortho_X)

endog = y

mod = sm.OLS(endog, exog)

res = mod.fit()

print(res.summary())

print("\np-values: ", res.pvalues)
poly = PolynomialFeatures(degree = 2, include_bias = True)

# Orthogonalize the powers of the predictor using QR decomposition

ortho_X = np.linalg.qr(poly.fit_transform(x.reshape(-1, 1)))[0][:, 1:]

exog = sm.add_constant(ortho_X)

endog = y

mod = sm.OLS(endog, exog)

res = mod.fit()

print(res.summary())

print("\np-values: ", res.pvalues)
poly = PolynomialFeatures(degree = 3, include_bias = True)

# Orthogonalize the powers of the predictor using QR decomposition

ortho_X = np.linalg.qr(poly.fit_transform(x.reshape(-1, 1)))[0][:, 1:]

exog = sm.add_constant(ortho_X)

endog = y

mod = sm.OLS(endog, exog)

res = mod.fit()

print(res.summary())

print("\np-values: ", res.pvalues)
poly = PolynomialFeatures(degree = 4, include_bias = True)

# Orthogonalize the powers of the predictor using QR decomposition

ortho_X = np.linalg.qr(poly.fit_transform(x.reshape(-1, 1)))[0][:, 1:]

exog = sm.add_constant(ortho_X)

endog = y

mod = sm.OLS(endog, exog)

res = mod.fit()

print(res.summary())

print("\np-values: ", res.pvalues)
from scipy import stats
boston_filepath = "../input/corrected-boston-housing/boston_corrected.csv"

index_cols = ["TOWN", "TRACT"]

data_cols = ["TOWN", "TRACT", "CMEDV", "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX",

            "PTRATIO", "B", "LSTAT"]

boston = pd.read_csv(boston_filepath, index_col = index_cols, usecols = data_cols)

boston.head()
boston["CMEDV"].mean()
sample_sd = boston["CMEDV"].std()

sample_sd / boston.shape[0]**0.5
boston["CMEDV"].sem()
np.random.seed(312)

n_bootstraps = 10000

means = np.empty(n_bootstraps)

for i in range(n_bootstraps):

    sample = resample(boston["CMEDV"])

    means[i] = sample.mean()

print("Bootstrap estimate of mean: ", means.mean())

print("Bootstrap estimate of std. error: ", means.std(ddof = 1))
stats.t.interval(0.95, boston.shape[0] - 1, loc = boston["CMEDV"].mean(), scale = boston["CMEDV"].sem())
boston["CMEDV"].median()
np.random.seed(312)

n_bootstraps = 10000

medians = np.empty(n_bootstraps)

for i in range(n_bootstraps):

    sample = resample(boston["CMEDV"])

    medians[i] = sample.median()

print("Bootstrap estimate of median: ", medians.mean())

print("Bootstrap estimate of std. error: ", medians.std(ddof = 1))
boston["CMEDV"].quantile(0.1)
np.random.seed(312)

n_bootstraps = 10000

tenth_percs = np.empty(n_bootstraps)

for i in range(n_bootstraps):

    sample = resample(boston["CMEDV"])

    tenth_percs[i] = sample.quantile(0.1)

print("Bootstrap estimate of tenth percentile: ", tenth_percs.mean())

print("Bootstrap estimate of std. error: ", tenth_percs.std(ddof = 1))