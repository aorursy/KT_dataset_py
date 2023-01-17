# importing packages

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import numpy as np

import pandas as pd



# import ML packages

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression

from sklearn import datasets, linear_model

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from scipy import stats

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error
# save filepath to variable for easier access

bike_file_path = '../input/london-bike-sharing-dataset/london_merged.csv' 



# read the data and store data in DataFrame titled bike_data

bike_data = pd.read_csv(bike_file_path)



# inspect data

bike_data.head()
# inspect variables

bike_data.info()
# check the sum of null records

bike_data.isnull().sum()
# plot distribution of cnt target variable

sns.distplot(bike_data['cnt'])

plt.show()
# inspect description of variables

bike_data.describe()
# create correlation matrix displaying pearson correlation coefficients for all variables

corr_matrix = bike_data.corr()

corr_matrix
# plot pair grid with histograms and scatterplots

bike_data_sample = bike_data.sample(1000)

p = sns.PairGrid(data=bike_data_sample, vars=['t1', 't2', 'hum', 'wind_speed', 'cnt'])

p.map_diag(plt.hist)

p.map_offdiag(plt.scatter)
# plot pair grid with histograms and scatterplots using season as hue

bike_data_sample = bike_data.sample(1000)

p = sns.PairGrid(data=bike_data_sample, vars=['t1', 't2', 'hum', 'wind_speed', 'cnt'], hue='season')

p.map_diag(plt.hist)

p.map_offdiag(plt.scatter)

plt.legend(title='Season', loc='center right', bbox_to_anchor=(1.65, 0.5), ncol=1, labels=['Spring', 'Summer', 'Fall', 'Winter'])
# convert timestamp string to datetime format for entire timestamp column

bike_data['timestamp'] = pd.to_datetime(bike_data['timestamp']) 

# retrieving timestamp column by iloc method

type(bike_data['timestamp'].iloc[0]) 



# create hour, month, and day of week variables from timestamp data

bike_data['hour']=bike_data['timestamp'].apply(lambda time: time.hour) 

bike_data['month']=bike_data['timestamp'].apply(lambda time: time.month)

bike_data['day_of_week']=bike_data['timestamp'].apply(lambda time: time.dayofweek)



# creating mapping variable for day of week labels

date_names = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'} 

bike_data['day_of_week'] = bike_data['day_of_week'].map(date_names)



bike_data.head()
# create box plots for time related variables

figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

figure.set_size_inches(24, 8)



sns.boxplot(data=bike_data, x='month', y='cnt', ax=ax1)

sns.boxplot(data=bike_data, x='hour', y='cnt', ax=ax2)

sns.boxplot(data=bike_data, x='day_of_week', y='cnt', ax=ax3)
# create point plot comparing cnt by hour for is_holiday variable

fig,(ax1)= plt.subplots(nrows=1)

fig.set_size_inches(18,5)

sns.pointplot(data=bike_data, x='hour', y='cnt', ci="sd", hue='is_holiday', ax=ax1, palette='YlGnBu')
# create point plot comparing cnt by hour for is_weekend variable

fig,(ax1)= plt.subplots(nrows=1)

fig.set_size_inches(18,5)

sns.pointplot(data=bike_data, x='hour', y='cnt', ci="sd", hue='is_weekend', ax=ax1, palette='YlGnBu')
# creating mapping variable for season labels

season_names = {0:'Spring',1:'Summer',2:'Fall',3:'Winter'} 

bike_data['season'] = bike_data['season'].map(season_names) 



# create point plot comparing cnt by hour for season variable

fig,(ax1)= plt.subplots(nrows=1)

fig.set_size_inches(18,5)

sns.pointplot(data=bike_data, x='hour', y='cnt', ci="sd", hue='season', ax=ax1, palette='YlGnBu')

plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
# creating mapping variable for weather labels

weather_names = {1:'Clear',2:'Scattered Clouds',3:'Broken Clouds',4:'Cloudy',7:'Light Rain',10:'Thunderstorm',26:'Snowing',94:'Freezing Fog'}

bike_data['weather_code'] = bike_data['weather_code'].map(weather_names)



# create point plot comparing cnt by hour for weather variable

fig,(ax1)= plt.subplots(nrows=1)

fig.set_size_inches(18,5)

sns.pointplot(data=bike_data, x='hour', y='cnt', ci="sd", hue='weather_code',ax=ax1, palette='YlGnBu')

plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
# reset data to prepare for building regression model

bike_data = pd.read_csv(bike_file_path)



# convert float variables to int

bike_data.weather_code = bike_data.weather_code.astype(int)

bike_data.is_holiday = bike_data.is_holiday.astype(int)

bike_data.is_weekend = bike_data.is_weekend.astype(int)

bike_data.season = bike_data.season.astype(int)



# convert timestamp string to datetime format for entire timestamp column

bike_data['timestamp'] = pd.to_datetime(bike_data['timestamp']) 



# retrieving timestamp column by iloc method

type(bike_data['timestamp'].iloc[0]) 



# create new variables from timestamp data

bike_data['hour']=bike_data['timestamp'].apply(lambda time: time.hour) 

bike_data['month']=bike_data['timestamp'].apply(lambda time: time.month)



# inspect data

bike_data.head()
# create binary dummy variables from categorical variables and drop first column to avoid multicollinearity

bike_data = pd.get_dummies(bike_data, columns = ['weather_code', 'season','hour','month'],drop_first = True)



# drop timestamp

bike_data.drop('timestamp', axis=1, inplace=True)



# inspect bike_data df with added dummy variables

bike_data.head(5)
# inspect variables

bike_data.info()
# set limit for correlation coefficient

drop_corr = .95



# select only upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# drop any variables with correlation coefficient greater than drop_corr value

to_drop = [column for column in upper.columns if any(upper[column] > drop_corr)]

bike_data = bike_data.drop(to_drop,axis=1)

print("Dropping: " + str(to_drop) + " variable(s) for exceeding correlation of " + str(drop_corr))

      

# display remaining variables represented as dataframe columns

bike_data.columns
# set the target variable

y = bike_data['cnt']



# set the independent predictor variables

X = bike_data.drop('cnt', axis=1)



# split data into training and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

X_train = sm.add_constant(X_train)

X_train.head()
# fit data to linear regression

mlr1 = sm.OLS(y_train, X_train).fit()



# view OLS regression results

print(mlr1.summary())
X_train2 = X_train.drop(['month_2','month_3','month_4','month_5','month_6','month_7','month_8','month_9','month_10','month_11','month_12'], axis=1)

mlr2 = sm.OLS(y_train, X_train2).fit()

print(mlr2.summary())
X_train3 = X_train2.drop(['season_1','season_2','season_3'], axis=1)

mlr3 = sm.OLS(y_train, X_train3).fit()

print(mlr3.summary())
X_train4 = X_train3.drop(['weather_code_2','weather_code_3','weather_code_4','weather_code_7','weather_code_10','weather_code_26'], axis=1)

mlr4 = sm.OLS(y_train, X_train4).fit()

print(mlr4.summary())
# create dataframe to calculate and display VIF for each variable

vif = pd.DataFrame()

vif['Features'] = X_train4.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train4.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# drop predictive variables to prepare final model

X_test = sm.add_constant(X_test)

X_test_1 = X_test[X_train4.columns] 



# fit data to linear regression for final model using test data

mlr_test = sm.OLS(y_test, X_test_1).fit()



# inspect X_test data

X_test_1.head()
# Making predictions using the final model

y_pred = mlr_test.predict(X_test_1)
# distribution plot of predicted y values vs test y values

sns.distplot((y_test - y_pred), bins=50);
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)   

plt.xlabel('y_test ', fontsize=18)                       

plt.ylabel('y_pred', fontsize=16) 
# create R2 score

r2 = r2_score(y_test, y_pred)



# create Adjusted R2 Score

p = len(X_test_1.columns)

n = y_test.shape[0]

adj_r2 = 1 - (1 - r2) * ((n - 1)/(n-p-1))



# create RMSE score

rmse = mean_squared_error(y_test, y_pred, squared = False)





# print final model performance stats

print(str(p) + " Predictors in Test Set")

print(str(n) + " Records in Test Set")

print("R2: " + str(r2))

print("Adj R2: " + str(adj_r2))

print("RMSE: " + str(rmse))
