# read in libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm



%matplotlib inline
# read in data - due to the size of the data I'm only using the two features needed

# as in the tutorial, only the first 100000 records are used

df = pd.read_csv('../input/harddrive.csv', usecols=['failure', 'smart_1_normalized'], nrows=100000)

df.head()
# set input and output variables to use in regression model

x = df['smart_1_normalized']

y = df['failure']



# add intercept to input variable

x = sm.add_constant(x)



# fit binomial regression model - statsmodels logit uses a different method but generates the same coefficients 

model = sm.GLM(y, x, family=sm.families.Binomial()).fit()
# show model summary

model.summary()
# show null vs model deviance

model.null_deviance, model.deviance
# plot & add a regression line

sns.regplot(df['smart_1_normalized'], df['failure'], line_kws={'color':'k', 'lw':1});