# import libraries we'll need

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from statsmodels.graphics.gofplots import ProbPlot

from sklearn.linear_model import ElasticNetCV



%matplotlib inline
# read in the data, inluding only the features we'll need

df = pd.read_csv('../input/2016-FCC-New-Coders-Survey-Data.csv', 

                 usecols=['Gender', 'Age', 'CommuteTime', 'HasChildren', 'AttendedBootcamp', 'HasDebt', 

                          'HoursLearning', 'MonthsProgramming', 'Income'])
# rename 'Gender' feature, and change to a boolean

df.rename(columns={'Gender':'IsWoman'}, inplace=True)

df['IsWoman'] = df['IsWoman'] == 'female'



# remove null values

df.dropna(inplace=True)

df.head()
# create input and output from dataframe

x = df[['Age', 'CommuteTime', 'HasChildren', 'AttendedBootcamp', 'IsWoman', 'HasDebt', 'HoursLearning', 'MonthsProgramming']]

y = df['Income']



# how many examples do we have?

print('Number of data points: ', len(y))

print("Mean of our predicted value: ", y.mean())
# use 10-fold cross-validation to fit a bunch of models using elastic net

# default parameters used

elnet = ElasticNetCV(cv=10)

elnet.fit(x, y)
# get coefficents for the best model

print('Intercept: ', elnet.intercept_)

coefs = list(zip(x.columns, elnet.coef_))

coefs
# get the variables with a coefficent that's not 0 

nonzero_coefs = [i[0] for i in coefs if i[1] != 0]

nonzero_coefs
# set new input variables to use in regression model

x = df[nonzero_coefs]



# add intercept to input variable

x = sm.add_constant(x)



# fit regression model, using statsmodels GLM uses a different method but gives the same results

#model = sm.GLM(y, x, family=sm.families.Gaussian()).fit()

model = sm.OLS(y, x).fit()
# seaborn residual plot

sns.residplot(model.fittedvalues, df['Income'], lowess=True, line_kws={'color':'r', 'lw':1})

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

plt.xlim(xmin=0, xmax=0.037)

plt.xlabel('Leverage')

plt.ylabel("Pearson residuals")

plt.title("Residuals vs leverage");
# take a closer look at our model

model.summary()
# generate GLM model for accessible deviance output

model2 = sm.GLM(y, x, family=sm.families.Gaussian()).fit()



# deviance output

print('Null deviance: {:.1f}'.format(model2.null_deviance))

print('Residual deviance: {:.1f}'.format(model2.deviance))
# added-variable plots for our model

fig = plt.figure(figsize=(12,8))

fig = sm.graphics.plot_partregress_grid(model, fig=fig)