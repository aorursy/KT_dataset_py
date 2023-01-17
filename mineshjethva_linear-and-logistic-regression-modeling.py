%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import statsmodels.api as sm

import numpy as np
# Read the 2015-2016 wave of NHANES data

da = pd.read_csv("../input/nhanes-2015-2016/NHANES.csv")



# Drop unused columns, and drop rows with any missing values.

vars = ["BPXSY1", "RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2", "BMXBMI", "SMQ020"]

da = da[vars].dropna()
model = sm.OLS.from_formula("BPXSY1 ~ RIDAGEYR", data=da)

result = model.fit()

result.summary()
da.BPXSY1.std()
cc = da[["BPXSY1", "RIDAGEYR"]].corr()

print(cc.BPXSY1.RIDAGEYR**2)
cc = np.corrcoef(da.BPXSY1, result.fittedvalues)

print(cc[0, 1]**2)
# Create a labeled version of the gender variable

da["RIAGENDRx"] = da.RIAGENDR.replace({1: "Male", 2: "Female"})
model = sm.OLS.from_formula("BPXSY1 ~ RIDAGEYR + RIAGENDRx", data=da)

result = model.fit()

result.summary()
# We need to use the original, numerical version of the gender

# variable to calculate the correlation coefficient.

da[["RIDAGEYR", "RIAGENDR"]].corr()
cc = np.corrcoef(da.BPXSY1, result.fittedvalues)

print(cc[0, 1]**2)
model = sm.OLS.from_formula("BPXSY1 ~ RIDAGEYR + BMXBMI + RIAGENDRx", data=da)

result = model.fit()

result.summary()
da[["RIDAGEYR", "RIAGENDR", "BMXBMI"]].corr()
from statsmodels.sandbox.predict_functional import predict_functional



# Fix certain variables at reference values.  Not all of these

# variables are used here, but we provide them with a value anyway

# to prevent a warning message from appearing.

values = {"RIAGENDRx": "Female", "RIAGENDR": 1, "BMXBMI": 25,

          "DMDEDUC2": 1, "RIDRETH1": 1, "SMQ020": 1}



# The returned values are the predicted values (pr), the confidence bands (cb),

# and the function values (fv).

pr, cb, fv = predict_functional(result, "RIDAGEYR",

                values=values, ci_method="simultaneous")



ax = sns.lineplot(fv, pr, lw=4)

ax.fill_between(fv, cb[:, 0], cb[:, 1], color='grey', alpha=0.4)

ax.set_xlabel("Age")

_ = ax.set_ylabel("SBP")
del values["BMXBMI"] # Delete this as it is now the focus variable

values["RIDAGEYR"] = 50

pr, cb, fv = predict_functional(result, "BMXBMI",

                values=values, ci_method="simultaneous")



ax = sns.lineplot(fv, pr, lw=4)

ax.fill_between(fv, cb[:, 0], cb[:, 1], color='grey', alpha=0.4)

ax.set_xlabel("BMI")

_ = ax.set_ylabel("SBP")
pp = sns.scatterplot(result.fittedvalues, result.resid)

pp.set_xlabel("Fitted values")

_ = pp.set_ylabel("Residuals")
# This is not part of the main Statsmodels API, so needs to be imported separately

from statsmodels.graphics.regressionplots import plot_ccpr



ax = plt.axes()

plot_ccpr(result, "RIDAGEYR", ax)

ax.lines[0].set_alpha(0.2) # Reduce overplotting with transparency

_ = ax.lines[1].set_color('orange')
ax = plt.axes()

plot_ccpr(result, "BMXBMI", ax)

ax.lines[0].set_alpha(0.2)

ax.lines[1].set_color("orange")
# This is not part of the main Statsmodels API, so needs to be imported separately

from statsmodels.graphics.regressionplots import add_lowess



# This is an equivalent way to fit a linear regression model, it needs to be

# done this way to be able to make the added variable plot

model = sm.GLM.from_formula("BPXSY1 ~ RIDAGEYR + BMXBMI + RIAGENDRx", data=da)

result = model.fit()

result.summary()



fig = result.plot_added_variable("RIDAGEYR")

ax = fig.get_axes()[0]

ax.lines[0].set_alpha(0.2)

_ = add_lowess(ax)
da["smq"] = da.SMQ020.replace({2: 0, 7: np.nan, 9: np.nan})
c = pd.crosstab(da.RIAGENDRx, da.smq).apply(lambda x: x/x.sum(), axis=1)

c["odds"] = c.loc[:, 1] / c.loc[:, 0]

c
c.odds.Male / c.odds.Female
c["logodds"] = np.log(c.odds)

c
model = sm.GLM.from_formula("smq ~ RIAGENDRx", family=sm.families.Binomial(), data=da)

result = model.fit()

result.summary()
c.logodds.Male - c.logodds.Female
model = sm.GLM.from_formula("smq ~ RIDAGEYR + RIAGENDRx", family=sm.families.Binomial(), data=da)

result = model.fit()

result.summary()
# Create a labeled version of the educational attainment variable

da["DMDEDUC2x"] = da.DMDEDUC2.replace({1: "lt9", 2: "x9_11", 3: "HS", 4: "SomeCollege",

                                       5: "College", 7: np.nan, 9: np.nan})



model = sm.GLM.from_formula("smq ~ RIDAGEYR + RIAGENDRx + DMDEDUC2x", family=sm.families.Binomial(), data=da)

result = model.fit()

result.summary()
values = {"RIAGENDRx": "Female", "RIAGENDR": 1, "BMXBMI": 25,

          "DMDEDUC2": 1, "RIDRETH1": 1, "SMQ020": 1,

          "DMDEDUC2x": "College", "BPXSY1": 120}



pr, cb, fv = predict_functional(result, "RIDAGEYR",

                values=values, ci_method="simultaneous")



ax = sns.lineplot(fv, pr, lw=4)

ax.fill_between(fv, cb[:, 0], cb[:, 1], color='grey', alpha=0.4)

ax.set_xlabel("Age")

ax.set_ylabel("Smoking")
pr1 = 1 / (1 + np.exp(-pr))

cb1 = 1 / (1 + np.exp(-cb))

ax = sns.lineplot(fv, pr1, lw=4)

ax.fill_between(fv, cb1[:, 0], cb1[:, 1], color='grey', alpha=0.4)

ax.set_xlabel("Age", size=15)

ax.set_ylabel("Smoking", size=15)
fig = result.plot_partial_residuals("RIDAGEYR")

ax = fig.get_axes()[0]

ax.lines[0].set_alpha(0.2)



_ = add_lowess(ax)
fig = result.plot_added_variable("RIDAGEYR")

ax = fig.get_axes()[0]

ax.lines[0].set_alpha(0.2)

_ = add_lowess(ax)
fig = result.plot_ceres_residuals("RIDAGEYR")

ax = fig.get_axes()[0]

ax.lines[0].set_alpha(0.2)

_ = add_lowess(ax)