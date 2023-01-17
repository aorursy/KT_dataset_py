# import libraries we'll need



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from statsmodels.graphics.gofplots import ProbPlot



%matplotlib inline
# read in data

df = pd.read_csv('../input/multipleChoiceResponses.csv', encoding='ISO-8859-1')

df.head()
# take a quick look for dtype and any null values in the features we're interested in

df[['CompensationAmount', 'Age']].info()
# select features of interest

df = df[['CompensationAmount', 'Age']]
# do some data cleaning



# remove punctuation with regex

df['CompensationAmount'] = df['CompensationAmount'].str.replace('[^\w\s]','')



# fill null values with a zero

df['CompensationAmount'].fillna(0, inplace=True)

df['Age'].fillna(0, inplace=True)



# change dtype of cleaned series to numeric

df['CompensationAmount'] = pd.to_numeric(df['CompensationAmount'])



# remove rows where CleanedCompensationAmount is less than 0

df = df[df['CompensationAmount'] > 0]
# set input and output variables to use in regression model

y = df['CompensationAmount']

x = df['Age']



# add intercept to input variable

x = sm.add_constant(x)



# fit poisson regression model 

model = sm.GLM(y, x, family=sm.families.Poisson()).fit()
# seaborn residual plot

sns.regplot(df['Age'], model.resid_deviance, fit_reg=False)

plt.title('Residual plot')

plt.xlabel('Age')

plt.ylabel('Residuals');
# statsmodels Q-Q plot on model residuals

QQ = ProbPlot(model.resid_deviance)

fig = QQ.qqplot(alpha=0.5, markersize=5);
# get data relating to high leverage points using statsmodels

# leverage info doesn't appear to be available for a poisson distribution

# gaussian model used for illustration



# fit OLS regression model 

model_g = sm.OLS(y, x).fit()



# leverage, from statsmodels

model_leverage = model_g.get_influence().hat_matrix_diag

# cook's distance, from statsmodels

model_cooks = model_g.get_influence().cooks_distance[0]



# plot cook's distance vs high leverage points

sns.regplot(model_leverage, model_cooks, fit_reg=False)

plt.xlim(xmin=-0.005, xmax=0.02)

plt.xlabel('Leverage')

plt.ylabel("Cook's distance")

plt.title("Cook's vs Leverage");
# remove compensation values above 150,000

df = df[df['CompensationAmount'] <= 150000]
# linear model to predict salary by age



# set input and output variables to use in regression model

y = df['CompensationAmount']

x = df['Age']



# add intercept to input variable

x = sm.add_constant(x)



# fit poisson regression model 

model = sm.GLM(y, x, family=sm.families.Poisson()).fit()



# fit OLS regression model 

model_g = sm.OLS(y, x).fit()

# seaborn residual plot

sns.regplot(df['Age'], model.resid_deviance, fit_reg=False)

plt.title('Residual plot')

plt.xlabel('Age')

plt.ylabel('Residuals');
# statsmodels Q-Q plot on model residuals

QQ = ProbPlot(model.resid_deviance)

fig = QQ.qqplot(alpha=0.5, markersize=5);
# get data relating to high leverage points using statsmodels



# leverage, from statsmodels

model_leverage = model_g.get_influence().hat_matrix_diag

# cook's distance, from statsmodels

model_cooks = model_g.get_influence().cooks_distance[0]



# plot cook's distance vs high leverage points

sns.regplot(model_leverage, model_cooks, fit_reg=False)

plt.xlim(xmin=-0.005, xmax=0.02)

plt.xlabel('Leverage')

plt.ylabel("Cook's distance")

plt.title("Cook's vs Leverage");
# add poisson fitted values to dataframe

df['reg_fit'] = model.fittedvalues



# sort dataframe by 'Age'

df.sort_values('Age', inplace=True)
# plot & add a regression line

sns.regplot(df['Age'], df['CompensationAmount'], fit_reg=False)

plt.plot(df['Age'], df['reg_fit']);