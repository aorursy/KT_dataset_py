import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb

import os



from pandas import Series, DataFrame

from pylab import rcParams

from sklearn import preprocessing



from sklearn import metrics

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_score



from sklearn.linear_model import LinearRegression

met_df = pd.read_csv('/kaggle/input/did-it-rain-in-seattle-19482017/seattleWeather_1948-2017.csv')

print(met_df.head()); print(); print()

met_df.info()
met_df.describe(include = 'all')
met_df.isna().sum()
P_median = met_df.PRCP.median()

met_df.PRCP.fillna(P_median, inplace = True)

met_df.isna().sum()
met_df.drop('RAIN', axis=1)
date = pd.to_datetime(met_df.DATE, format="%Y-%m-%d")

met_df['DATE'] = date

met_df.head()
## daily TMAX

from datetime import datetime

DoY_str = met_df.DATE.dt.strftime('%j')

Day_of_Year = [int(a) for a in DoY_str]

#group data based on "day of year"

groupD      = met_df.groupby(Day_of_Year)

DoY_TMAX   = groupD['TMAX'].mean()  # daily climatological mean TMAX

daily_TMAX = [DoY_TMAX[a] for a in Day_of_Year]



## daily TMIN

DoY_TMIN   = groupD['TMIN'].mean()  # daily climatological mean TMIN

daily_TMIN = [DoY_TMIN[a] for a in Day_of_Year]





## daily PRCP

#group data based on "day of year"

DoY_PRCP   = groupD['PRCP'].mean()  # daily climatological mean PRCP

daily_PRCP = [DoY_PRCP[a] for a in Day_of_Year]
# Add all these new variables to DataFrame:

daily_PRCP_df  = pd.DataFrame(daily_PRCP,  columns = ['daily_PRCP'])

daily_TMAX_df  = pd.DataFrame(daily_TMAX,  columns = ['daily_TMAX'])

daily_TMIN_df  = pd.DataFrame(daily_TMIN,  columns = ['daily_TMIN'])



met_new_df = pd.concat([met_df['DATE'],met_df['PRCP'],met_df['TMAX'],met_df['TMIN'],

                        daily_TMAX_df,daily_TMIN_df,daily_PRCP_df], axis = 1)



met_new_df.describe(include = 'all')
%matplotlib inline

rcParams['figure.figsize'] = 10, 8

sb.set_style('whitegrid')



sb.pairplot(met_new_df, palette = 'husl')

plt.show()
rcParams['figure.figsize'] = 8, 6

sb.heatmap(met_new_df.corr(), vmin=-1, vmax=1, annot=True, cmap = 'RdBu_r')

plt.show()
fig, axis = plt.subplots(1, 3,figsize=(16,5))

sb.scatterplot(x = 'daily_TMAX', y ='daily_PRCP', data = met_new_df, ax = axis[0])

sb.scatterplot(x = 'daily_TMIN', y ='daily_PRCP', data = met_new_df, ax = axis[1])

sb.scatterplot(x = 'daily_TMAX', y ='daily_TMIN', data = met_new_df, ax = axis[2])

plt.show()
fig, ax1 = plt.subplots(figsize=(8,6))

ax2 = ax1.twinx()

ax1.scatter(Day_of_Year, daily_TMAX, c='purple')

ax1.scatter(Day_of_Year, daily_TMIN, c='orange')

ax2.scatter(Day_of_Year, daily_PRCP, c='g')

plt.grid(False)

ax1.set_xlabel('Day of year', fontsize=14)

ax1.set_ylabel('daily_TMAX & daily_TMIN', fontsize=14)

ax2.set_ylabel('daily_PRCP', fontsize=16, c='g')



plt.show()
met_new_df.drop(['DATE','PRCP','TMIN','TMAX','daily_TMIN'], inplace = True, axis=1)

met_new_df.head()
X_train, X_test, Y_train, Y_test = train_test_split(met_new_df['daily_TMAX'].values.reshape(-1, 1),

                                                   met_new_df['daily_PRCP'], test_size=0.2, random_state=10)                             

Lin_Reg = LinearRegression()

Lin_Reg.fit(X_train, Y_train)

R_squared_train = Lin_Reg.score(X_train, Y_train)

print(R_squared_train)
Lin_Reg.coef_[0], Lin_Reg.intercept_
T = np.array(range(42,79))

P = Lin_Reg.coef_[0] *  T + Lin_Reg.intercept_



fig, ax1 = plt.subplots(figsize=(8, 6))

plt.plot(T, P, color = 'purple', lw = 3)

plt.scatter(X_train, Y_train, c = 'orange')



ax1.set_xlabel('daily_TMAX', fontsize=14)

ax1.set_ylabel('daily_PRCP', fontsize=14)



TITLE1 = 'Train dataset: daily_PRCP = %.4f'% Lin_Reg.coef_

TITLE2 = ' * daily_TMAX + %.4f, ' %Lin_Reg.intercept_

TITLE3 = 'R-squared = %.3f'% R_squared_train

TITLE = TITLE1 + TITLE2 + TITLE3

ax1.set_title(TITLE, fontsize=14)

plt.show()
R_squared_test = Lin_Reg.score(X_test, Y_test)

print(R_squared_test)
Y_pred = Lin_Reg.predict(X_test)



T = [0, 0.1, 0.2, 0.28]



fig, ax1 = plt.subplots(figsize=(7, 7))

plt.plot(T, T, color = 'b', lw = 3)

plt.scatter(Y_test, Y_pred, c = 'g')



ax1.set_xlabel('test daily_PRCP', fontsize=14)

ax1.set_ylabel('predicted daily_PRCP', fontsize=14)



TITLE = 'test R-squared = %.3f'% R_squared_test

ax1.set_title(TITLE, fontsize=14)

plt.show()
CV_scores = cross_val_score(Lin_Reg.fit(X_train, Y_train), X_test, Y_test, cv=5)

print ('5-fold cross-validation: scores = ')

print(CV_scores)
CV_prediction = cross_val_predict(Lin_Reg.fit(X_train, Y_train), X_test, Y_test, cv=5)

CV_R_squared = metrics.r2_score(Y_test, CV_prediction)

CV_R_squared