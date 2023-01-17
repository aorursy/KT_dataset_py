# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import math

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats

from datetime import datetime

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

%matplotlib inline

import statsmodels.api as sm

import statsmodels.formula.api as sm

from statsmodels.formula.api import ols

from statsmodels.sandbox.regression.predstd import wls_prediction_std

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import plotly.plotly as py

import plotly.graph_objs as go



# MatPlotlib

from matplotlib import pylab



# Scientific libraries

from scipy.optimize import curve_fit



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/aggregates/Aggregates/naep_states_base.csv')
df.tail()
df = df[df['STATE'].str.contains("CALIFORNIA")] 

df.TEST_SUBJECT.unique()
df.TEST_YEAR.unique()
dfread = df[df['TEST_SUBJECT'].str.contains("Reading")] 

dfmath = df[df['TEST_SUBJECT'].str.contains("Mathematics")] 
dfread4 = dfread.loc[dfread['TEST_YEAR'] == 4]

dfmath4 = dfmath.loc[dfmath['TEST_YEAR'] == 4] 

dfread8 = dfread.loc[dfread['TEST_YEAR'] == 8]

dfmath8 = dfmath.loc[dfmath['TEST_YEAR'] == 8] 
ax1 = dfread4.plot.scatter(x='YEAR', y='AVG_SCORE', c='DarkBlue')

ax1.grid()

plt.title('Scatterplot of Reading Scores of 4th graders vs Time for CA Education NAEP Scores')
fig,ax= plt.subplots()

for n, group in dfread.groupby('TEST_YEAR'):

    group.plot(x='YEAR',y='AVG_SCORE', ax=ax,label=n)

    ax.grid()

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',

           ncol=4, mode="expand", title="Test Year/Grade",borderaxespad=2.)

plt.title('Plot of Reading Scores vs Time for CA Education NAEP Scores')
# ========================

# Model for Original Data

# ========================



# Get the linear models

lm_original = np.polyfit(dfread4.YEAR, dfread4.AVG_SCORE, 1)



# calculate the y values based on the co-efficients from the model

r_x, r_y = zip(*((i, i*lm_original[0] + lm_original[1]) for i in dfread4.YEAR))



# Put in to a data frame, to keep is all nice

lm_original_plot = pd.DataFrame({

'YEAR' : r_x,

'AVG_SCORE' : r_y

})

fig, ax = plt.subplots()

 

# Plot the original data and model

dfread4.plot(kind='scatter', color='Blue', x='YEAR', y='AVG_SCORE', ax=ax, title='Linear Regression of CA 4th Graders Reading Scores')

lm_original_plot.plot(kind='line', color='Red', x='YEAR', y='AVG_SCORE', ax=ax)

ax.grid()

 

plt.show()

	

# Needed to show the plots inline

%matplotlib inline


from sklearn.linear_model import LinearRegression

X = dfread4.YEAR.values.reshape(-1, 1)

y = dfread4.AVG_SCORE.values.reshape(-1, 1) 



model = LinearRegression()

model.fit(X, y)

xs = np.array([2018, 2019, 2020, 2021]).reshape((-1, 1))



y_predict = model.predict(xs)



print('Projected Scores:', y_predict, sep='\n')



fig,ax= plt.subplots()

plt.scatter(xs, y_predict)

ax.grid()

plt.title('Scatterplot of Projected Reading Scores of 4th graders vs Time')
# ========================

# Model for Original Data

# ========================



# Get the linear models

lm_original = np.polyfit(dfread8.YEAR, dfread8.AVG_SCORE, 1)



# calculate the y values based on the co-efficients from the model

r_x, r_y = zip(*((i, i*lm_original[0] + lm_original[1]) for i in dfread8.YEAR))



# Put in to a data frame, to keep is all nice

lm_original_plot = pd.DataFrame({

'YEAR' : r_x,

'AVG_SCORE' : r_y

})



fig, ax = plt.subplots()

 

# Plot the original data and model

dfread8.plot(kind='scatter', color='Blue', x='YEAR', y='AVG_SCORE', ax=ax, title='Linear Regression of CA 8th Graders Reading Scores')

lm_original_plot.plot(kind='line', color='Red', x='YEAR', y='AVG_SCORE', ax=ax)

ax.grid()

 

plt.show()

	

# Needed to show the plots inline

%matplotlib inline


x = dfread8.YEAR 

y= dfread8.AVG_SCORE

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures

import operator



# transforming the data to include another axis

x = x[:, np.newaxis]

y = y[:, np.newaxis]



polynomial_features= PolynomialFeatures(degree=3)

x_poly = polynomial_features.fit_transform(x)



model = LinearRegression()

model.fit(x_poly, y)

y_poly_pred = model.predict(x_poly)



rmse = np.sqrt(mean_squared_error(y,y_poly_pred))

r2 = r2_score(y,y_poly_pred)

print("RMSE=", rmse)

print("R^2=", r2)



plt.scatter(x, y, s=10)

# sort the values of x before line plot



sort_axis = operator.itemgetter(0)

sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)

x, y_poly_pred = zip(*sorted_zip)

plt.plot(x, y_poly_pred, color='m')

plt.xlabel('Year')

plt.ylabel('Average Score')

plt.title('Polynomial Regression of degree 3 of CA 8th Reading Scores')

plt.grid(True)

plt.show()





xs = np.array([2018, 2019, 2020, 2021]).reshape((-1, 1))

x_ = polynomial_features.fit_transform(xs)

y_predict = model.predict(x_)



print('Projected Scores:', y_predict, sep='\n')



fig,ax= plt.subplots()

plt.scatter(xs, y_predict)

ax.grid()

plt.title('Scatterplot of Projected Reading Scores of 8th graders vs Time')
fig,ax= plt.subplots()

for n, group in dfmath.groupby('TEST_YEAR'):

    group.plot(x='YEAR',y='AVG_SCORE', ax=ax,label=n)

    ax.grid()

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',

           ncol=4, mode="expand", title="Test Year/Grade", borderaxespad=2.)

plt.title('Plot of Math Scores vs Time for CA Education NAEP Scores')
lm_original = np.polyfit(dfmath8.YEAR, dfmath8.AVG_SCORE, 1)



# calculate the y values based on the co-efficients from the model

r_x, r_y = zip(*((i, i*lm_original[0] + lm_original[1]) for i in dfmath8.YEAR))



# Put in to a data frame, to keep is all nice

lm_original_plot = pd.DataFrame({

'YEAR' : r_x,

'AVG_SCORE' : r_y

})



fig, ax = plt.subplots()

 

# Plot the original data and model

dfmath8.plot(kind='scatter', color='Blue', x='YEAR', y='AVG_SCORE', ax=ax, title='Linear Regression of CA 8th Graders Math Scores')

lm_original_plot.plot(kind='line', color='Red', x='YEAR', y='AVG_SCORE', ax=ax)

ax.grid()

 

plt.show()

	

# Needed to show the plots inline

%matplotlib inline

X = dfmath8.YEAR.values.reshape(-1, 1)

y = dfmath8.AVG_SCORE.values.reshape(-1, 1) 



model = LinearRegression()

model.fit(X, y)

xs = np.array([2018, 2019, 2020, 2021]).reshape((-1, 1))



y_predict = model.predict(xs)



print('Projected Scores:', y_predict, sep='\n')



fig,ax= plt.subplots()

plt.scatter(xs, y_predict)

ax.grid()

plt.title('Scatterplot of Projected Math Scores of 8th graders vs Time')
lm_original = np.polyfit(dfmath4.YEAR, dfmath4.AVG_SCORE, 1)



# calculate the y values based on the co-efficients from the model

r_x, r_y = zip(*((i, i*lm_original[0] + lm_original[1]) for i in dfmath4.YEAR))



# Put in to a data frame, to keep is all nice

lm_original_plot = pd.DataFrame({

'YEAR' : r_x,

'AVG_SCORE' : r_y

})



fig, ax = plt.subplots()

 

# Plot the original data and model

dfmath4.plot(kind='scatter', color='Blue', x='YEAR', y='AVG_SCORE', ax=ax, title='Linear Regression of CA 4th Graders Math Scores')

lm_original_plot.plot(kind='line', color='Red', x='YEAR', y='AVG_SCORE', ax=ax)

ax.grid()

 

plt.show()

	

# Needed to show the plots inline

%matplotlib inline



x = dfmath4.YEAR 

y= dfmath4.AVG_SCORE





# transforming the data to include another axis

x = x[:, np.newaxis]

y = y[:, np.newaxis]



polynomial_features= PolynomialFeatures(degree=3)

x_poly = polynomial_features.fit_transform(x)



model = LinearRegression()

model.fit(x_poly, y)

y_poly_pred = model.predict(x_poly)



rmse = np.sqrt(mean_squared_error(y,y_poly_pred))

r2 = r2_score(y,y_poly_pred)

print("RMSE=", rmse)

print("R^2=", r2)



plt.scatter(x, y, s=10)

# sort the values of x before line plot



sort_axis = operator.itemgetter(0)

sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)

x, y_poly_pred = zip(*sorted_zip)

plt.plot(x, y_poly_pred, color='m')

plt.xlabel('Year')

plt.ylabel('Average Score')

plt.title('Polynomial Regression of degree 3 of CA 4th Graders Math Scores')

plt.grid(True)



plt.show()

xs = np.array([2018, 2019, 2020, 2021]).reshape((-1, 1))

x_ = polynomial_features.fit_transform(xs)

y_predict = model.predict(x_)



print('Projected Scores:', y_predict, sep='\n')



fig,ax= plt.subplots()

plt.scatter(xs, y_predict)

ax.grid()

plt.title('Scatterplot of Projected Math Scores of 4th graders vs Time')
print("Years contained in 8th grade math scores")

dfmath8.YEAR.unique()
print("Years contained in 4th grade math scores")

dfmath4.YEAR.unique()


growthCAmath=[]

yr = [2017, 2015, 2013, 2011, 2009, 2007, 2005, 2003, 2000, 1996, 1992,1990]

for y in yr:

    b= y-4

    dfm8 = dfmath8['AVG_SCORE'][(dfmath8['YEAR'] == y)].values

    dfm4 = dfmath4['AVG_SCORE'][(dfmath4['YEAR'] == b)].values

    diff = np.subtract(dfm8,dfm4)

    growthCAmath.append(diff)

    cols=['Growth']

    g = pd.DataFrame(growthCAmath, columns=cols)



g.fillna(0, inplace=True)

year=['Year']

dfy = pd.DataFrame(yr, columns=year) 

CAmath8g = pd.concat([dfy, g], axis=1)

print(CAmath8g)

ax1 = CAmath8g.plot.scatter(x='Year', y='Growth', c='DarkBlue')

ax1.grid()

plt.title('Scatterplot of Growth of Math Scores vs Time for CA Education NAEP Scores')
print("Years contained in 8th grade reading scores")

dfread8.YEAR.unique()
print("Years contained in 4th grade reading scores")

dfread4.YEAR.unique()
growthCAread=[]

yr = [2017, 2015, 2013, 2011, 2009, 2007, 2005, 2003, 2002, 1998]

for y in yr:

    b= y-4

    dfr8 = dfread8['AVG_SCORE'][(dfread8['YEAR'] == y)].values

    dfr4 = dfread4['AVG_SCORE'][(dfread4['YEAR'] == b)].values

    diff = np.subtract(dfr8,dfr4)

    growthCAread.append(diff)

    cols=['Growth']

    gr = pd.DataFrame(growthCAread, columns=cols)



gr.fillna(0, inplace=True)

year=['Year']

dfyr = pd.DataFrame(yr, columns=year) 

CAread8g = pd.concat([dfyr, gr], axis=1)

print(CAread8g)

ax1 = CAread8g.plot.scatter(x='Year', y='Growth', c='DarkBlue')

ax1.grid()

plt.title('Scatterplot of Growth of Reading Scores vs Time for CA Education NAEP Scores')
dffs = pd.read_csv('../input/states_all.csv')

dffs.head()
dffs = dffs[dffs['STATE'].str.contains("CALIFORNIA")] 

dffs.head()




x = dffs['YEAR']

y1 = dffs['ENROLL']

y2 = dffs['TOTAL_EXPENDITURE']



plt.show()

fig, axs = plt.subplots(2, 1, constrained_layout=True)

axs[0].plot(x, y1, 'x')

axs[0].set_title('Enrollment vs Year')

axs[0].set_xlabel('Year')

axs[0].set_ylabel('Enrollment')

axs[0].grid()

fig.suptitle('Comparison of Enrollment and Total Revenue over the Years', fontsize=16)



axs[1].plot(x, y2, '--')

axs[1].set_xlabel('Year')

axs[1].set_title('Total Expenditure vs Year')

axs[1].set_ylabel('Total Expenditure')

axs[1].grid()

plt.show()
