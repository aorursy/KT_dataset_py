# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

filePath = '/kaggle/input/bikes.csv'

bikesData = pd.read_csv(filePath)

print(bikesData.info())
bikesData.describe()

columnsToDrop = ['instant','casual','registered','atemp','dteday']
bikesData = bikesData.drop(columnsToDrop,1)
#Scaling some 

columnsToScale = ['temp','hum','windspeed']

scaler = StandardScaler()
# Task 4
bikesData[columnsToScale] = scaler.fit_transform(bikesData[columnsToScale])
bikesData[columnsToScale].describe()
bikesData.head()
#Transforming the Data
#dayCnt:** count of the days from the beginning of the dataset

bikesData['dayCount'] = pd.Series(range(bikesData.shape[0]))/24
bikesData.sort_values('dayCount', axis= 0, inplace=True)
nrow = bikesData.shape[0]
X = bikesData.dayCount.values.reshape(nrow,1)
Y = bikesData.cnt
clf = linear_model.LinearRegression()
bike_lm = clf.fit(X, Y)

# Task
bikesData['cntDeTrended'] = bikesData.cnt - bike_lm.predict(X)
### Analyzing and visualizing the dataset

%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(bikesData.loc[:,'cnt'])
plt.plot(bikesData.loc[:,'cntDeTrended'])
plt.plot(bike_lm.predict(X))
plt.legend(['With trend','Detrended','Trend'])
plt.show()
bikesData.hist(bins=50, figsize=(20,15))
plt.show()
corr_matrix = bikesData.corr()
corr_matrix
# Monthly Distribution of counts

import seaborn as sns

fig,ax=plt.subplots(figsize=(15,8))
sns.set_style('white')
#Bar plot for seasonwise monthly distribution of counts
sns.barplot(x='mnth',y='cnt',data=bikesData[['mnth','cnt','season']],hue='season',ax=ax)
ax.set_title('Seasonwise monthly distribution of counts')
plt.show()
#Bar plot for weekday wise monthly distribution of counts
fig,ax1=plt.subplots(figsize=(15,8))
sns.barplot(x='mnth',y='cnt',data=bikesData[['mnth','cnt','weekday']],hue='weekday',ax=ax1)
ax1.set_title('Weekday wise monthly distribution of counts')
plt.show()
#Bar Plot for hour wise dictribution of counts
fig,ax1=plt.subplots(figsize=(15,8))
sns.barplot(x='weekday',y='cnt',data=bikesData[['weekday','cnt','hr']],hue='hr',ax=ax1)
ax1.set_title('Weekday wise hourly distribution of counts')
plt.show()
columnToPlotScatter = ['temp','hum','windspeed','hr','cntDeTrended']
from pandas.plotting import scatter_matrix
scatter_matrix(bikesData[columnToPlotScatter], figsize=(12,8), alpha=0.05)
plt.show()
# Dividing the Dataset into Train and Test Dataset

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(bikesData, test_size=0.3, random_state=42)
train_set.sort_values('dayCount', axis= 0, inplace=True)
test_set.sort_values('dayCount', axis= 0, inplace=True)
print(len(train_set), "train +", len(test_set), "test")
# Training on Linear Regression

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

lin_reg = LinearRegression()
display_scores(-cross_val_score(lin_reg, train_set, train_set['cnt'], cv=10, scoring="neg_mean_absolute_error"))
display_scores(np.sqrt(-cross_val_score(lin_reg, train_set, train_set['cnt'], cv=10, scoring="neg_mean_squared_error")))
train_set_lin = train_set.copy()
train_set_lin['predictedCounts'] = cross_val_predict(lin_reg, train_set, train_set['cnt'], cv=10)
train_set_lin['resids'] = train_set_lin['predictedCounts'] -  train_set['cnt']
# From the Above RMSE we find that Linear Regression with all attributes is overfitting the model, lets train on only few features

# From the correlation Matrix above we can see that features yr,hr,weekday,temp and hum have considerable correlation with cnt hence we run the model on these features


trainingCols= train_set[['yr','hr','weekday','temp','hum', 'dayCount']]
trainingLabels = train_set['cnt']

lin_reg = LinearRegression()
display_scores(-cross_val_score(lin_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_absolute_error"))
display_scores(np.sqrt(-cross_val_score(lin_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_squared_error")))
train_set_lin = train_set.copy()
train_set_lin['predictedCounts'] = cross_val_predict(lin_reg, trainingCols, trainingLabels, cv=10)
train_set_lin['resids'] = train_set_lin['predictedCounts'] - trainingLabels
# Lets try Decision Tree Regressor 

rfc_clf = DecisionTreeRegressor(random_state = 42)
display_scores(-cross_val_score(rfc_clf, trainingCols, trainingLabels, cv=10, scoring="neg_mean_absolute_error"))
display_scores(np.sqrt(-cross_val_score(rfc_clf, trainingCols, trainingLabels, cv=10, scoring="neg_mean_squared_error")))
train_set_dtr = train_set.copy()
train_set_dtr['predictedCounts'] = cross_val_predict(rfc_clf, trainingCols, trainingLabels, cv=10)
train_set_dtr['resids'] = train_set_dtr['predictedCounts'] - trainingLabels
# We see that RMSE has refuced to a great Extent, Let us try a better algorithim Random Forest Regressor 
# with different min_sample_leaf, max_depth and min_samples_split 

import math
min_samples_leaf = range(5,20,5)
max_depth = range(5,50,5)
min_samples_split = range(5,20,5)

mse_rf = {}
current_mse = math.inf

for msl in min_samples_leaf:
    for md in max_depth:
        for mss in min_samples_split:

            dt = RandomForestRegressor(n_estimators=40, min_samples_leaf=msl, max_depth=md, min_samples_split=mss)

            # Train (update y_train)
            dt.fit(trainingCols, trainingLabels)

            # Predict
            predictions = dt.predict(trainingCols)
             
            # Update MSE 
            mse = mean_squared_error(trainingLabels, predictions)

            if mse <= current_mse:
                mse_rf['value'] = mse
                mse_rf['min_samples_split'] = mss
                mse_rf['max_depth'] = md
                mse_rf['min_samples_leaf'] = msl

                current_mse = mse

print("-----------------\nRandom Forests\n-----------------")
print("MSE: ", mse_rf)

forest_reg = RandomForestRegressor(n_estimators=40, random_state=42)


display_scores(-cross_val_score(forest_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_absolute_error"))
display_scores(np.sqrt(-cross_val_score(forest_reg, trainingCols, trainingLabels, cv=10, scoring="neg_mean_squared_error")))
train_set_freg = train_set.copy()
train_set_freg['predictedCounts'] = cross_val_predict(forest_reg, trainingCols, trainingLabels, cv=10)
train_set_freg['resids'] = train_set_freg['predictedCounts'] - trainingLabels
print(train_set_freg)
# plotting predicted counts and resids

import seaborn as sns

plt.figure(figsize = (8,4))
sns.scatterplot(x = trainingLabels, y = train_set_freg['predictedCounts'])
plt.xlabel('counts')
plt.ylabel('Predictions')
plt.show()
times = [8,17]
for time in times:
    fig = plt.figure(figsize=(8, 6))
    fig.clf()
    ax = fig.gca()
    train_set_freg_time = train_set_freg[train_set.hr == time]
    train_set_freg_time.plot(kind = 'line', x = 'dayCount', y = 'cnt', ax = ax)
    train_set_freg_time.plot(kind = 'line', x = 'dayCount', y = 'predictedCounts', ax =ax)
    plt.show()
    
#plot to show in difference of count and predictedCount in terms of weekdays      
weekdays = [0,6]
for wd in weekdays:
    fig = plt.figure(figsize=(8, 6))
    fig.clf()
    ax = fig.gca()
    train_set_freg_time = train_set_freg[train_set.weekday == wd]
    train_set_freg_time.plot(kind = 'line', x = 'dayCount', y = 'cnt', ax = ax)
    train_set_freg_time.plot(kind = 'line', x = 'dayCount', y = 'predictedCounts', ax =ax)
    plt.show()
#Fine-Tuning the Model

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'max_depth': [28, 30, 32, 34, 36], 'min_samples_leaf': [5, 10, 15, 12],'min_samples_split': [120, 128, 136]},
]

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')


grid_search.fit(trainingCols, trainingLabels)
print(grid_search.best_params_)


feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

#We find that features Hour and Humidity is the most important followed by dayCount and WeekDay
# lets try Ensemble Gradient Boost Method

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor();

param_grid = [
    {'n_estimators': [2000], 'alpha': [0.1], 'min_samples_leaf': [50,],}]

grid_search = GridSearchCV(gbm, param_grid, cv=5, scoring='neg_mean_squared_error')


grid_search.fit(trainingCols, trainingLabels)
print(grid_search.best_params_)


feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
print(grid_search.best_score_)
final_mse=-grid_search.best_score_
print(np.sqrt(final_mse))

# Hence we find that Random Forest is giving the Best Results so far
# lets try Ensemble Gradient Boost Method

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor();

param_grid = [
    {'n_estimators': [2000], 'alpha': [0.1], 'min_samples_leaf': [50,],}]

grid_search = GridSearchCV(gbm, param_grid, cv=5, scoring='neg_mean_squared_error')


grid_search.fit(trainingCols, trainingLabels)
print(grid_search.best_params_)


feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
print(grid_search.best_score_)
final_mse=-grid_search.best_score_
print(np.sqrt(final_mse))

# Hence we find that Random Forest is giving the Best Results so far