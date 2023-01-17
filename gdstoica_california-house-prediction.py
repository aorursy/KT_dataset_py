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
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
df = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
df.info() #target is median_house_value
df['ocean_proximity'].value_counts()
df['total_bedrooms'].isna().sum()
df[df['total_bedrooms'].isna()]
df[df['total_bedrooms'].notna()]
bins = np.linspace(min(df['total_bedrooms']), max(df['total_bedrooms']), num=20)
ax1 = sns.distplot(df['total_bedrooms'], bins=bins, color='gold', kde=True, hist_kws=dict(edgecolor='k', lw=1)) 
ax2 = sns.distplot(df['total_bedrooms'].fillna(value=df['total_bedrooms'].mean()), bins=bins, color='green', kde=True, hist_kws=dict(edgecolor='k', lw=1))
bins = np.linspace(min(df['total_bedrooms']), max(df['total_bedrooms']), num=20)
ax1 = sns.distplot(df['total_bedrooms'], bins=bins, color='gold', kde=True, hist_kws=dict(edgecolor='k', lw=1)) 
ax2 = sns.distplot(df['total_bedrooms'].fillna(value=df['total_bedrooms'].median()), bins=bins, color='red', kde=True, hist_kws=dict(edgecolor='k', lw=1)) 
x = df.drop(columns='median_house_value')
y = df['median_house_value']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)
trainset = pd.concat([xtrain, ytrain], axis=1)
testset = pd.concat([xtest, ytest], axis=1)
trainset['median_house_value'].describe()
trainset['median_house_value'].hist(bins=20)
trainset[trainset['median_house_value']>=475000]
sns.distplot(trainset['median_house_value'])
print("Skewness: %f" % trainset['median_house_value'].skew()) # normal distribution: 0
print("Kurtosis: %f" % trainset['median_house_value'].kurt()) # normal distribution: 3
attributes = ["median_house_value", "median_income", "total_rooms", "total_bedrooms",
              "housing_median_age", "population"]
scatter_matrix(trainset[attributes], figsize=(20, 15))
plt.figure(figsize=(12, 10))
sns.scatterplot(data=trainset, x="longitude", y="latitude", hue="median_house_value", alpha=0.1, s=80)
plt.figure(figsize=(12, 10))
sns.scatterplot(data=trainset, x="longitude", y="latitude", hue="median_income", alpha=0.1, s=80)
trainset['housing_median_age'].describe()
sns.scatterplot(data=trainset, x="housing_median_age", y="median_income", alpha=0.5, s=80)
sns.scatterplot(data=trainset, x="housing_median_age", y="median_house_value", alpha=0.5, s=80)
trainset['housing_median_age'].hist(bins=20)
plt.figure(figsize=(12, 10))
sns.scatterplot(data=trainset[trainset['housing_median_age']>=52], x="longitude", y="latitude", hue="median_house_value", alpha=0.5, s=80)  # two main clusters (San Francisco & Los Angeles)
median = trainset['total_bedrooms'].median()
trainset['total_bedrooms'].fillna(value=median, inplace=True)
testset['total_bedrooms'].fillna(value=median, inplace=True)
trainset['bedrooms_per_household'] = trainset['total_bedrooms']/trainset['households']
trainset['bedrooms_per_room'] = trainset['total_bedrooms']/trainset['total_rooms']

testset['bedrooms_per_household'] = testset['total_bedrooms']/testset['households']
testset['bedrooms_per_room'] = testset['total_bedrooms']/testset['total_rooms']
trainset.drop(trainset[trainset['ocean_proximity']=='ISLAND'].index, inplace=True)
testset.drop(testset[testset['ocean_proximity']=='ISLAND'].index, inplace=True)
trainset.info()
trainset = pd.get_dummies(trainset, columns=['ocean_proximity'], sparse=False, 
                              drop_first=True)
testset = pd.get_dummies(testset, columns=['ocean_proximity'], sparse=False, 
                              drop_first=True)
trainset.info()
trainset['population_per_households'] = trainset['population'] / trainset['households']    #new feature
testset['population_per_households'] = testset['population'] / testset['households']   
import seaborn as sns
sns.heatmap(trainset.corr(), annot=True)
corr_matrix = trainset.corr().abs()
high_corr_var=np.where(corr_matrix>0.5)
high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]
high_corr_var
high_corr_var = ['total_rooms', 'total_bedrooms', 'population', 'median_income', 'households',  'median_house_value', 'bedrooms_per_room',
                'bedrooms_per_household']
plt.figure(figsize=(14,14))
sns.heatmap(trainset[high_corr_var].corr(), annot=True)
trainset.drop(columns=['total_bedrooms', 'population', 'bedrooms_per_household', 'households'], inplace=True)
testset.drop(columns=['total_bedrooms', 'population', 'bedrooms_per_household', 'households'], inplace=True)
trainset.info()
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age", "bedrooms_per_room", "population_per_households"]
scatter_matrix(trainset[attributes], figsize=(20, 15))
trainset.describe()
def get_iqr_results(num_series, k=1.5):
    # calculate percentiles and IQR
    q25 = np.percentile(num_series, 25)
    q75 = np.percentile(num_series, 75)
    iqr = q75 - q25
    
    # calculate normal and extreme upper and lower cut off
    cutoff = iqr * k
    lower = q25 - cutoff 
    upper = q75 + cutoff
    
    result = {
        'lower': lower,
        'upper': upper}
    
    return result
trainset.info()
numerical_columns = ['housing_median_age', 'total_rooms', 'median_income', 'bedrooms_per_room', 'population_per_households']
column_limits = {}
for column in numerical_columns:
    column_limits[column] = get_iqr_results(trainset[column])
column_limits
trainset.shape
for column in numerical_columns:
    trainset.loc[trainset[column]<column_limits[column]['lower'], column] = column_limits[column]['lower']
    trainset.loc[trainset[column]>column_limits[column]['upper'], column] = column_limits[column]['upper']
    testset.loc[testset[column]<column_limits[column]['lower'], column] = column_limits[column]['lower']
    testset.loc[testset[column]>column_limits[column]['upper'], column] = column_limits[column]['upper']
trainset.describe()
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age", "bedrooms_per_room", "population_per_households"]
scatter_matrix(trainset[attributes], figsize=(20, 15))
xtrain = trainset.drop(columns=["median_house_value"])
ytrain = trainset["median_house_value"]

xtest = testset.drop(columns=["median_house_value"])
ytest = testset["median_house_value"]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(xtrain)
xtrain_scaled = scaler.transform(xtrain)
xtest_scaled = scaler.transform(xtest)
xtrain_scaled = pd.DataFrame(xtrain_scaled, index=xtrain.index, columns=xtrain.columns)
xtest_scaled = pd.DataFrame(xtest_scaled, index=xtest.index, columns=xtest.columns)
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
lin_reg=LinearRegression()
lin_reg.fit(xtrain_scaled,ytrain)
ypred = lin_reg.predict(xtest_scaled)
mse_test = mean_squared_error(ytest,ypred)
rmse_test  = np.sqrt(mse_test)
rmse_test
ytpred = lin_reg.predict(xtrain_scaled)
mse_train = mean_squared_error(ytrain,ytpred)
rmse_train = np.sqrt(mse_train)
rmse_train
forest_reg = RandomForestRegressor()
forest_reg.fit(xtrain_scaled, ytrain)
ypred = forest_reg.predict(xtest_scaled)
ytpred = forest_reg.predict(xtrain_scaled)
rf_rmse_train = np.sqrt(mean_squared_error(ytrain,ytpred))
rf_rmse_test = np.sqrt(mean_squared_error(ytest,ypred))
rf_rmse_train
rf_rmse_test #overfitting
svm_reg = SVR(kernel="linear")
svm_reg.fit(xtrain_scaled, ytrain)
ytpred = svm_reg.predict(xtrain_scaled)
ypred = svm_reg.predict(xtest_scaled)
svm_mse_train = mean_squared_error(ytrain, ytpred)
svm_mse_test = mean_squared_error(ytest, ypred)
svm_rmse_train = np.sqrt(svm_mse_train)
svm_rmse_test = np.sqrt(svm_mse_test)
svm_rmse_train
svm_rmse_test
xtrain_scaled.shape
forest_reg = RandomForestRegressor(n_estimators=15, max_features=4, max_depth=8, random_state=42)
forest_reg.fit(xtrain_scaled, ytrain)
ypred = forest_reg.predict(xtest_scaled)
ytpred = forest_reg.predict(xtrain_scaled)
rf_rmse_train = np.sqrt(mean_squared_error(ytrain,ytpred))
rf_rmse_test = np.sqrt(mean_squared_error(ytest,ypred))
rf_rmse_train
rf_rmse_test
params = [
    {'n_estimators': [10, 15, 20, 30], 'max_features': [2, 4, 6, 8], 'min_samples_split':[2,4],
     'min_samples_leaf':[1,2,3], 'bootstrap':[True, False]}
  ]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, params, cv=5, scoring='neg_root_mean_squared_error', return_train_score=True)
grid_search.fit(xtrain_scaled, ytrain)
grid_search.best_params_
grid_search.cv_results_
forest_reg = RandomForestRegressor(n_estimators=30, max_features=2, bootstrap=False, min_samples_leaf=1, 
                                   min_samples_split=2, random_state=42)
forest_reg.fit(xtrain_scaled, ytrain)
ypred = forest_reg.predict(xtest_scaled)
ytpred = forest_reg.predict(xtrain_scaled)
rf_rmse_train = np.sqrt(mean_squared_error(ytrain,ytpred))
rf_rmse_test = np.sqrt(mean_squared_error(ytest,ypred))
rf_rmse_train #overfitting a lot
rf_rmse_test
list(zip(xtrain_scaled.columns, list(forest_reg.feature_importances_)))
params = [
    {'n_estimators': [15, 20, 30], 'max_features': [2, 4], 'min_samples_split':[3,4],
     'min_samples_leaf':[2,3], 'bootstrap':[True, False], 'max_depth': [3, 4, 6]}
  ]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, params, cv=5, scoring='neg_root_mean_squared_error', return_train_score=True)
grid_search.fit(xtrain_scaled, ytrain)
grid_search.best_params_
forest_reg = RandomForestRegressor(n_estimators=10, max_features=4, max_depth=14, bootstrap=True, random_state=42, max_samples=0.7,  max_leaf_nodes=35)
# min_samples_split=10, min_samples_leaf=14, 
forest_reg.fit(xtrain_scaled, ytrain)
ypred = forest_reg.predict(xtest_scaled)
ytpred = forest_reg.predict(xtrain_scaled)
rf_rmse_train = np.sqrt(mean_squared_error(ytrain,ytpred))
rf_rmse_test = np.sqrt(mean_squared_error(ytest,ypred))
rf_rmse_train
rf_rmse_test
params = [
    {'n_estimators': [10, 15, 20], 'max_features': [4, 6], 'min_samples_split':[8, 10, 12],
     'min_samples_leaf':[2, 4, 8], 'bootstrap':[True, False], 'max_depth': [10, 12, 14], 'max_leaf_nodes':[20, 25, 30, 35]}
  ]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, params, cv=3, scoring='neg_root_mean_squared_error', return_train_score=True)
grid_search.fit(xtrain_scaled, ytrain)
grid_search.best_params_
forest_reg = RandomForestRegressor(n_estimators=20, max_features=6, max_depth=12, min_samples_leaf=8, min_samples_split=8,
                                   bootstrap=True, random_state=42, max_leaf_nodes=35)
# min_samples_split=10, min_samples_leaf=14, 
forest_reg.fit(xtrain_scaled, ytrain)
ypred = forest_reg.predict(xtest_scaled)
ytpred = forest_reg.predict(xtrain_scaled)
rf_rmse_train = np.sqrt(mean_squared_error(ytrain,ytpred))
rf_rmse_test = np.sqrt(mean_squared_error(ytest,ypred))
rf_rmse_train
rf_rmse_test
from sklearn.metrics import r2_score
r2_score(ytrain,ytpred)
r2_score(ytest,ypred)
forest_reg_basic = RandomForestRegressor(random_state=42)
forest_reg_basic.fit(xtrain_scaled, ytrain)
ypredb = forest_reg_basic.predict(xtest_scaled)
ytpredb = forest_reg_basic.predict(xtrain_scaled)
rf_rmse_trainb = np.sqrt(mean_squared_error(ytrain,ytpredb))
rf_rmse_testb = np.sqrt(mean_squared_error(ytest,ypredb))
rf_rmse_trainb
rf_rmse_testb
r2_score(ytrain,ytpredb)
r2_score(ytest,ypredb)
from sklearn.linear_model import Ridge
rr = Ridge(alpha=0.1) 
rr.fit(xtrain_scaled, ytrain)
ypred_ridge = rr.predict(xtest_scaled)
ypredt_ridge = rr.predict(xtrain_scaled)
rr_rmse_trainb = np.sqrt(mean_squared_error(ytrain,ypredt_ridge))
rr_rmse_testb = np.sqrt(mean_squared_error(ytest, ypred_ridge))
rr_rmse_trainb
rr_rmse_testb
r2_score(ytrain,ypredt_ridge)
r2_score(ytest,ypred_ridge)