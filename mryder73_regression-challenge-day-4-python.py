# import libraries we'll need

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from statsmodels.graphics.gofplots import ProbPlot



%matplotlib inline
# read in data, using only those features we'll need

df = pd.read_csv('../input/ehresp_2014.csv', usecols=['erbmi', 'euexfreq', 'euwgt', 'euhgt', 'ertpreat'])

df.head()



# feature references:

# erbmi = body mass index 

# euexfreq = how many times in the past week the person exercised (outside of their job)

# euwgt = weight, in pounds

# euhgt = height, in inches

# ertpreat = amount of time spent eating and drinking (in minutes) over the past week
# remove rows where bmi is less than 0

df = df[df['erbmi'] > 0]
# set input and output variables to use in regression model

x = df[['euexfreq', 'euwgt', 'euhgt', 'ertpreat']]

y = df['erbmi']



# add intercept to input variable

x = sm.add_constant(x)



# fit regression model, using statsmodels GLM uses a different method but gives the same results

#model = sm.GLM(y, x, family=sm.families.Gaussian()).fit()

model = sm.OLS(y, x).fit()
# seaborn residual plot

sns.residplot(model.fittedvalues, df['erbmi'], lowess=True, line_kws={'color':'r', 'lw':1})

plt.title('Residual plot')

plt.xlabel('Predicted values')

plt.ylabel('Residuals');
# statsmodels Q-Q plot on model residuals

QQ = ProbPlot(model.resid)

fig = QQ.qqplot(alpha=0.5, markersize=5, line='s')

plt.title('QQ plot');
# normalised residuals

model_norm_resid = model.get_influence().resid_studentized_internal



# absolute squared normalised residuals

model_norm_resid_abs_sqrt = np.sqrt(np.abs(model_norm_resid))



# plot scale-location

sns.regplot(model.fittedvalues, model_norm_resid_abs_sqrt, lowess=True, line_kws={'color':'r', 'lw':1})

plt.xlabel('Fitted values')

plt.ylabel('Sqrt abs standardized residuals')

plt.title('Scale-location');
# get data relating to high leverage points using statsmodels



# leverage, from statsmodels

model_leverage = model.get_influence().hat_matrix_diag



# plot residuals vs high leverage points

sns.regplot(model_leverage, model.resid_pearson, fit_reg=False)

plt.xlim(xmin=-0.0005, xmax=0.013)

plt.xlabel('Leverage')

plt.ylabel("Pearson residuals")

plt.title("Residuals vs leverage");
# examine our model

model.summary()
# generate GLM model for accessible deviance output

model2 = sm.GLM(y, x, family=sm.families.Gaussian()).fit()



# deviance output

print('Null deviance: {:.1f}'.format(model2.null_deviance))

print('Residual deviance: {:.1f}'.format(model2.deviance))
# added variable plots

fig = plt.figure(figsize=(12,8))

fig = sm.graphics.plot_partregress_grid(model, fig=fig)