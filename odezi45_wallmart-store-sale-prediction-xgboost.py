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
# We will need the following libraries and modules
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import *
import seaborn as sns
import datetime 
import time
from time import gmtime, strftime
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import StandardScaler   
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
sns.set()
# Load the data from a .csv in the same folder
train_df = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
test_df = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip')
stores_df = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')
features_df = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip')
# Let's explore the top 5 rows of the data
print(train_df.head())
print(test_df.head())
print(stores_df.head())
print(features_df.head())
trainall = pd.merge(train_df, stores_df, how='left', on=['Store'])
trainall = pd.merge(trainall, features_df, how='left', on=['Store','Date'])

testall = pd.merge(test_df, stores_df, how='left', on=['Store'])
testall = pd.merge(testall, features_df, how='left', on=['Store','Date'])

trainall.drop(columns=['IsHoliday_y'], axis=1, inplace=True)
testall.drop(columns=['IsHoliday_y'], axis=1, inplace=True)

trainall.rename(columns={'IsHoliday_x': 'IsHoliday'}, inplace=True)
testall.rename(columns={'IsHoliday_x': 'IsHoliday'}, inplace=True)


# Create Submission dataframe
submission = testall[['Store', 'Dept', 'Date']].copy()
submission['Id'] = submission['Store'].map(str) + '_' + submission['Dept'].map(str) + '_' + submission['Date'].map(str)
submission.drop(['Store', 'Dept', 'Date'], axis=1, inplace=True)
#check data type
pd.DataFrame(trainall.dtypes, columns=['Type'])
# Convert the Date column attribute from object to datetime column
trainall.Date = pd.to_datetime(trainall.Date)
testall.Date = pd.to_datetime(testall.Date)
#trainall.head()
pd.DataFrame(trainall.dtypes, columns=['Type'])
# data.isnull() # shows a df with the information whether a data point is null 
# Since True = the data point is missing, while False = the data point is not missing, we can sum them
# This will give us the total number of missing values feature-wise
trainall.isnull().sum()
#Lets convert all NaN value to '0' as simply droping all missing values is not always recommended
# Also make all value absolute because I am under the assumption that markdown are not negative values, as that will suggest 
#that the price of goods at walmart has been increased rather down marked down(reduction) e.g (+) values means reduction while
#(-) values will be addition in tis case.
trainall[['MarkDown1','MarkDown2','MarkDown3','MarkDown4', 'MarkDown5']] = trainall[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].fillna(0)
testall[['MarkDown1','MarkDown2','MarkDown3','MarkDown4', 'MarkDown5']] = testall[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].fillna(0)
trainall[['MarkDown1','MarkDown2','MarkDown3','MarkDown4', 'MarkDown5']] = abs(trainall[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']])
testall[['MarkDown1','MarkDown2','MarkDown3','MarkDown4', 'MarkDown5']] = abs(testall[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']])

# lets view if we have any null value in the dataset
trainall.isnull().sum()
# Let's check the descriptives without 'Model'
trainall.describe(include='all')
# A great step in the data exploration is to display the probability distribution function (PDF) of a variable
# The PDF will show us how that variable is distributed 
# This makes it very easy to spot anomalies, such as outliers
# The PDF is often the basis on which we decide whether we want to transform a feature
sns.distplot(trainall['Weekly_Sales'])
# Obviously there are some outliers present especially the negative value

# Outliers are a great issue for OLS, thus we must deal with them in some way
# It may be a useful exercise to try training a model without removing the outliers
# but in this case i will just simply remove the negative value as it will make little or no difference

# Then we can create a new df, with the condition that all weekly sales must be above or equal to 1 of 'weekly sales'
train_ = trainall[trainall['Weekly_Sales']>= 1]
# In this way we have essentially removed the less 0.5% of the data about 'weekly ales'
train_.describe(include='all')
# We can check the PDF once again to ensure that the result is still distributed in the same way overall
# however, there are much fewer outliers
sns.distplot(train_['Weekly_Sales'])
sns.set(style="white")
corr = train_.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(20, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.title('Correlation Matrix for Train dataset', fontsize=18)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.show()
#white, dark, whitegrid, darkgrid, ticks
train_.columns.values
#Scatter plot of numeric columns vs weekly sales

fig, axs = plt.subplots(4, 3,figsize=(16,14))

axs[0, 0].scatter(train_.Weekly_Sales, train_.Size,)
axs[0, 0].set_title('Size')

axs[0, 1].scatter(train_.Weekly_Sales, train_.Temperature,)
axs[0, 1].set_title('Temperature')

axs[0, 2].scatter(train_.Weekly_Sales, train_.Fuel_Price,)
axs[0, 2].set_title('Fuel_Price')

axs[1, 0].scatter(train_.Weekly_Sales, train_.MarkDown1,)
axs[1, 0].set_title('MarkDown1')

axs[1, 1].scatter(train_.Weekly_Sales, train_.MarkDown2,)
axs[1, 1].set_title('MarkDown2')

axs[1, 2].scatter(train_.Weekly_Sales, train_.MarkDown3,)
axs[1, 2].set_title('MarkDown3')

axs[2, 0].scatter(train_.Weekly_Sales, train_.MarkDown4,)
axs[2, 0].set_title('MarkDown4')

axs[2, 1].scatter(train_.Weekly_Sales, train_.MarkDown5,)
axs[2, 1].set_title('MarkDown5')

axs[2, 2].scatter(train_.Weekly_Sales, train_.CPI,)
axs[2, 2].set_title('CPI')

axs[3, 0].scatter(train_.Weekly_Sales, train_.Unemployment,)
axs[3, 0].set_title('Unemployment')

for ax in axs.flat:
    ax.set(xlabel='Label', ylabel='Weekly Sales')
    

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
#Get week from date for both train and test data
train_['Week'] = train_['Date'].dt.week
testall['Week'] = testall['Date'].dt.week
#convert Store and Department to string by adding a string value (S_, D_) to each variable 
train_['Store'] = 'S_' + train_['Store'].map(str)
train_['Dept'] = 'D_' + train_['Dept'].map(str)
testall['Store'] = 'S_' + testall['Store'].map(str)
testall['Dept'] = 'D_' + testall['Dept'].map(str)
# group the average departmental weekly sales value for each store

#select columns week,store,dept and weekly sales
trainallstore = train_[['Week','Store','Dept','Weekly_Sales']]
# group store and dept by average weekly sales
trainallstore = trainallstore.groupby(['Week','Store','Dept']).mean()

#remove index
trainallstore.reset_index(level=0, inplace=True)
trainallstore.reset_index(level=0, inplace=True)
trainallstore.reset_index(level=0, inplace=True)
trainallstore.rename(columns={'Weekly_Sales': 'SD_Sales'}, inplace=True)
#Now we merge the classK with the train and test data
trainall_N = train_.merge(trainallstore, how='left')
testall_N = testall.merge(trainallstore, how='left')
# add a check point to the data by copying
trainall = trainall_N.copy()
testall = testall_N.copy()
# Finally, once we reset the index, a new column will be created containing the old index (just in case)
# We won't be needing it, thus 'drop=True' to completely forget about it
train_ = trainall.reset_index(drop=True)
test_ = testall.reset_index(drop=True)
# dropping irrelevant columns since we will be creating dummies
train_some = train_.drop(columns=['Store','Dept','Date','Type','IsHoliday'])
#creating dummies for 5 column
dept_dummies = pd.get_dummies(train_['Dept'])
type_dummies = pd.get_dummies(train_['Type'])
store_dummies = pd.get_dummies(train_['Store'])
#n_dummies = pd.get_dummies(train_['storedept'])
holiday_dummies = pd.get_dummies(train_['IsHoliday'])
#merging all 5 dummies to main train data
#train_unscale = pd.concat([train_some,type_dummies,store_dummies,dept_dummies, holiday_dummies,n_dummies],axis=1)
train_unscale = pd.concat([train_some,type_dummies,store_dummies,dept_dummies, holiday_dummies],axis=1)


#Repeat the same for test data
test_some = test_.drop(columns=['Type','IsHoliday'])
dept_dummies2 = pd.get_dummies(test_['Dept'])
type_dummies2 = pd.get_dummies(test_['Type'])
store_dummies2 = pd.get_dummies(test_['Store'])
#n_dummies2 = pd.get_dummies(test_['storedept'])
holiday_dummies2 = pd.get_dummies(test_['IsHoliday'])

#test_ = pd.concat([test_some,type_dummies2,n_dummies2,store_dummies2,dept_dummies2,holiday_dummies2],axis=1)
test_ = pd.concat([test_some,type_dummies2,store_dummies2,dept_dummies2,holiday_dummies2],axis=1)
test_unscale = test_.drop(columns=['Store','Dept','Date'])
#fill NaN values with previous
test_unscale.fillna(method='ffill', inplace=True)
# sorted values of correlation
train_unscale.corr()['Weekly_Sales'].sort_values()
#train_unscale = train_unscale.reset_index()
#test_unscale = test_unscale.reset_index()

# Let's transform 'weekly_sales' with a log transformation
train_unscale['log_price'] = np.log(train_unscale['Weekly_Sales'])
#train_unscale['log_price'] = train_unscale['Weekly_Sales']
# Then we add it to our data frame
train_logprice = train_unscale.drop(['Weekly_Sales'],axis=1)
# Let's quickly see the columns of our data frame
train_logprice.columns.values
train_unscale.isnull().sum()
# The target(s) (dependent variable) is 'log price'
targets = train_logprice['log_price']

# The inputs are everything BUT the dependent variable, so we can simply drop it
inputs = train_logprice.drop(['log_price'],axis=1)
# Create a scaler object
scaler = StandardScaler()
# Fit the inputs (calculate the mean and standard deviation feature-wise)
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
#This is for the initial testing, once a model is selected, the xtrain will be inputs_scaled and the ytrain will be targets
# Split the variables with an 75-25 split and some random state
# To have the same split as mine, use random_state = 365
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.25, random_state=365)
# Lets try linear regression for fun
# Create a linear regression object
reg = LinearRegression()
# Fit the regression with the scaled TRAIN inputs and targets
reg.fit(x_train,y_train)
# Let's check the outputs of the regression
# I'll store them in y_hat
y_hat = reg.predict(x_train)
# The simplest way to compare the targets (y_train) and the predictions (y_hat) is to plot them on a scatter plot
# The closer the points to the 45-degree line, the better the prediction
plt.scatter(y_train, y_hat)
# Let's also name the axes
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
# Sometimes the plot will have different scales of the x-axis and the y-axis
# This is an issue as we won't be able to interpret the '45-degree line'
# We want the x-axis and the y-axis to be the same
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()
# Find the R-squared of the model
reg.score(x_train,y_train)

# Note that this is NOT the adjusted R-squared

# Obtain the bias (intercept) of the regression
reg.intercept_
# We can obtain the weights (coefficients) of the regression, but there is no need as I are not going to use thi model
# to check coefficient use reg.coef_
y_hat_test = reg.predict(x_test)
# To Create a scatter plot with the test targets and the test predictions
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()
# Finally, let's manually check these predictions
# To obtain the actual prices, we take the exponential of the log_price
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
df_pf.head()
# After displaying y_test, we find what the issue is
# The old indexes are preserved (recall earlier in that code we made a note on that)
# The code was: data_cleaned = data_4.reset_index(drop=True)

# Therefore, to get a proper result, we must reset the index and drop the old indexing
y_test = y_test.reset_index(drop=True)

# Let's overwrite the 'Target' column with the appropriate values
# Again, we need the exponential of the test log price
df_pf['Target'] = np.exp(y_test)
df_pf
# Find the R-squared of the model

''''#This will take too long, so dont run
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 4],
    'n_estimators': [50,80, 150]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(x_train, y_train)
grid_search.best_params_
'''
#Random forest model specification

regr = RandomForestRegressor(n_estimators=20, criterion='mse', max_depth=None, 
                      min_samples_split=2, min_samples_leaf=1, 
                      min_weight_fraction_leaf=0.0, max_features='auto', 
                      max_leaf_nodes=None, min_impurity_decrease=0.0, 
                      min_impurity_split=None, bootstrap=True, 
                      oob_score=False, n_jobs=1, random_state=None, 
                      verbose=2, warm_start=False)

#Train on data
regr.fit(x_train, y_train.ravel())
##Model evaluation
##To evaluate the model, we will look at MAE and accuracy in terms of the number of times it
##correctly estimated an upward or downward deviation from the median.
y_pred = regr.predict(x_train)
trainzz = y_train.to_frame()
trainzz['Predicted'] = y_pred
#quick plot of targets and prediction
plt.scatter(trainzz.log_price, trainzz.Predicted)
# Let's also name the axes
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_pred)',size=18)
# We want the x-axis and the y-axis to be the same
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()
y_predt = regr.predict(x_test)
testzz = y_test.to_frame()
testzz['Predicted'] = y_predt
plt.scatter(testzz.log_price, testzz.Predicted)
# Let's also name the axes
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_pred)',size=18)
# Sometimes the plot will have different scales of the x-axis and the y-axis
# This is an issue as we won't be able to interpret the '45-degree line'
# We want the x-axis and the y-axis to be the same
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()
trainr2 = r2_score(y_train , y_pred)
trainmae = mean_squared_error(y_train , y_pred)
testmae = mean_squared_error(y_test , y_predt)
testr2 = r2_score(y_test, y_predt)
print('The train score r2= ',trainr2, 'and mae =', trainmae)
print('The test score r2 = ',testr2,'and mae = ', testmae)
'''
xgb_reg = xgb.XGBRegressor(n_estimators=50)

# Parameters to Grid search
tuned_parameters = [{'learning_rate':[0.1],'max_depth':[3,4,5,10]}]

# Grid search

xgb_gs = GridSearchCV(estimator=xgb_reg, param_grid=tuned_parameters, scoring= 'r2', cv=5, n_jobs=10)
xgb_gs.fit(x_train,y_train)
print('The best parameters for XGBoost Regression is: ',xgb_gs.best_params_)
'''

xgbr = xgb.XGBRegressor(base_score=None, booster=None, colsample_bylevel=None,
             colsample_bynode=None, colsample_bytree=None, gamma=None,
             gpu_id=None, importance_type='gain', interaction_constraints=None,
             learning_rate=0.1, max_delta_step=None, max_depth=10,
             min_child_weight=None, monotone_constraints=None,
             n_estimators=200, n_jobs=None, num_parallel_tree=None,
             objective='reg:squarederror', random_state=None, reg_alpha=None,
             reg_lambda=None, scale_pos_weight=None, subsample=None,
             tree_method=None, validate_parameters=None, verbosity=0)

xgbr.fit(x_train, y_train)

score = xgbr.score(x_train, y_train)   
print("Training score: ", score) 
y_predxgb = xgbr.predict(x_train)
y_predxgbt = xgbr.predict(x_test)

xtrainr2 = r2_score(y_train , y_predxgb)
xtrainmae = mean_squared_error(y_train , y_predxgb)
xtestmae = mean_squared_error(y_test , y_predxgbt)
xtestr2 = r2_score(y_test, y_predxgbt)
print('The train score r2 = ',xtrainr2,'and mae = ', xtrainmae)
print('The test score r2 = ',xtestr2, ' and mae = ',xtestmae)

x_ax = range(len(y_test))
plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
plt.plot(x_ax, y_predxgbt, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()

trainzz['Predictedxgb'] = y_predxgb
testzz['Predictedxgb'] = y_predxgbt

fig, axs = plt.subplots(2, 2,figsize=(16,16))

axs[0, 0].scatter(trainzz.log_price, trainzz.Predicted,)
axs[0, 0].set_title('RF train')

axs[0, 1].scatter(testzz.log_price, testzz.Predicted,)
axs[0, 1].set_title('RF test')

axs[1, 0].scatter(trainzz.log_price, trainzz.Predictedxgb,)
axs[1, 0].set_title('XGB train')

axs[1, 1].scatter(testzz.log_price, testzz.Predictedxgb,)
axs[1, 1].set_title('XGB test')


for ax in axs.flat:
    ax.set(xlabel='Predicted', ylabel='Real')
    

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

xgbfinal = xgb.XGBRegressor(base_score=None, booster=None, colsample_bylevel=None,
             colsample_bynode=None, colsample_bytree=None, gamma=None,
             gpu_id=None, importance_type='gain', interaction_constraints=None,
             learning_rate=0.1, max_delta_step=None, max_depth=10,
             min_child_weight=None, monotone_constraints=None,
             n_estimators=200, n_jobs=None, num_parallel_tree=None,
             objective='reg:squarederror', random_state=None, reg_alpha=None,
             reg_lambda=None, scale_pos_weight=None, subsample=None,
             tree_method=None, validate_parameters=None, verbosity=0)
xgbfinal.fit(inputs_scaled, targets)

score = xgbfinal.score(inputs_scaled, targets)  
print("Training score: ", score) 
RF = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=None, max_features='auto', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=20, n_jobs=-1, oob_score=False,
                      random_state=None, verbose=2, warm_start=False)
RF.fit(inputs_scaled, targets)
#process test data by scaling it using the same scaler defined earlier
scaler = StandardScaler()
scaler.fit(test_unscale)

# The target(s) (dependent variable) is 'log price'
test_scaled = scaler.transform(test_unscale)
#At final, we can predict the test!
y_pred = xgbfinal.predict(test_scaled)

y_pred2 = RF.predict(test_scaled)
reg.fit(inputs_scaled, targets)
y_pred3 = reg.predict(test_scaled)

# Menaging final_df to have competition rules standard
testdataframe = test_[['Store','Dept','Date']]
testdataframe['Id'] = testdataframe['Store'].astype(str) + '_' + testdataframe['Dept'].astype(str) + '_' + testdataframe['Date'].astype(str)
testdataframe.drop(columns=['Store','Dept','Date'],inplace=True)

# Therefore, to get a proper result, we must reset the index and drop the old indexing
testdataframe['Id'].reset_index(drop=True)
testdataframe['Weekly_Sales'] = np.exp(y_pred)


testdataframe['Id']  = testdataframe.replace(regex=['S_'], value='')
testdataframe['Id']  = testdataframe.replace(regex=['D_'], value='')
testdataframe.head(10)
testdataframe.to_csv('submission.csv', index=False)

