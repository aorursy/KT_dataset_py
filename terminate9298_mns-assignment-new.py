import os

import pandas as pd

import seaborn as sns 

import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression 

from sklearn.metrics import mean_squared_error

from math import sqrt

from scipy.stats import ttest_ind ,linregress , ttest_rel

import statsmodels.api as sm

# import statsmodels.regression.linear_model as sm

from scipy.stats import probplot

from scipy.stats import zscore

import numpy as np

import scipy

from statsmodels.graphics.regressionplots import influence_plot
df = pd.read_csv('/kaggle/input/Machine Learning (Codes and Data Files)/Data/country.csv')[['Corruption_Index','Gini_Index']]
print('Creating Linear Regression Model ... ')

lr = LinearRegression()

lr.fit(df['Gini_Index'].values.reshape(-1, 1) , df['Corruption_Index'].values)

print('Fitting Done on Model ... ')

r2_score = lr.score(df['Gini_Index'].values.reshape(-1, 1) , df['Corruption_Index'])

print('R2 Score is ',r2_score)

print('Since the Model R2 Score is ',r2_score , ', the model explains ',round(r2_score*100,2) , ' % of the variation in GI')

print('Coefficients for the linear regression problem is ',lr.coef_)

print('Intersect Value is ',lr.intercept_)

y_pred = lr.predict(df['Gini_Index'].values.reshape(-1, 1))

rms = sqrt(mean_squared_error(df['Corruption_Index'].values.reshape(-1,1), y_pred))

print('Root Mean Squared Is ',rms)

print('The p-Value is ',linregress(df['Gini_Index'],df['Corruption_Index'])[3])
linreg = linregress(df['Gini_Index'],df['Corruption_Index'])

slope = linreg[0]

se = linreg[4]

print('The Confidence Intervel for Regression Coefficient b1 is ')

print('Upper Bound is ' ,slope - se* scipy.stats.t.ppf(0.05/2.,18))

print('Lower Bound is ' ,slope + se* scipy.stats.t.ppf(0.05/2.,18))
sm1 = sm.OLS(df['Corruption_Index'], df['Gini_Index'] ).fit()
corruption_resid = sm1.resid

probplot(corruption_resid, plot=plt)

print('\n')
def get_standardized_values( vals ):

        return (vals - vals.mean())/vals.std()

plt.scatter( get_standardized_values( sm1.fittedvalues ),get_standardized_values(corruption_resid))
zscore(df['Gini_Index'].values)
corr_influence = sm1.get_influence()

(c, p) = corr_influence.cooks_distance

plt.stem(np.arange( 20),np.round( c, 3 ),markerfmt=',')