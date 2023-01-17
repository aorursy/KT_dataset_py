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
import matplotlib.pyplot as plt #visualization library in Python for 2D plots of arrays
import seaborn as sns #interactive visualization library
import xgboost as xgb #uses a gradient boosting framework
from sklearn.model_selection import GridSearchCV #implements a “fit” and a “score” method
from sklearn.model_selection import cross_val_score #cross-validation
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import GradientBoostingRegressor#it allows for the optimization of arbitrary differentiable loss functions
from sklearn.preprocessing import StandardScaler#for scalling
from sklearn.metrics import r2_score,mean_squared_error# for scoring

train= pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print(train.shape)
print(test.shape)

#print(train.info())
print("number of Non_catgorical feature",train.dtypes[train.dtypes != "object"].count())
print("number of catgorical feature",train.dtypes[train.dtypes == "object"].count())
train.describe()
plt.figure(figsize=(18,12))
sns.heatmap(train.corr(),annot=True, )
# Features most correlated with SalePrice
corr_matrix_pearson=train.corr()
top_correlated = corr_matrix_pearson.nlargest(10, 'SalePrice')[
    'SalePrice'].index
corr_matrix_pearson_top_correlated = corr_matrix_pearson.loc[top_correlated, top_correlated]
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_pearson_top_correlated, cbar=True,
            annot=True, square=True, fmt='.2f', annot_kws={'size': 12})
plt.show()
sns.set_style("whitegrid")
cols = corr_matrix_pearson_top_correlated['SalePrice'][corr_matrix_pearson_top_correlated['SalePrice']>0.6].index
plt.figure(figsize=(8,6))
sns.pairplot(train[cols], size = 2)
plt.figure()
sns.pairplot(train,x_vars=["GrLivArea"], y_vars='SalePrice', size=5)#,kind='scatter' or 'reg')
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.title("GrLivArea vs SalePrice")
plt.show()
plt.figure(figsize=(16,8))
sns.distplot(train.SalePrice)
plt.xlabel("SalePrice")
plt.ylabel("count")
plt.title("Sale_Price_distribution")
plt.figure()
sns.pairplot(train,x_vars=["GrLivArea"], y_vars=['SalePrice'], hue="SaleCondition",size=5)#,kind='scatter' or 'reg')
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.title("GrLivArea vs SalePrice")
plt.show()
plt.figure()
sns.pairplot(train,x_vars=["GrLivArea"], y_vars=['SalePrice'],size=5)#,kind='scatter' or 'reg')
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.title("GrLivArea vs SalePrice")
plt.show()
train.drop(train[(train["GrLivArea"]>4000) & (train['SalePrice']<300000)].index, inplace=True)
plt.figure()
sns.pairplot(train,x_vars=["GrLivArea"], y_vars=['SalePrice'],size=5)#,kind='scatter' or 'reg')
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.title("GrLivArea vs SalePrice")
plt.show()
features_all = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']))
features_all.shape
print("train dataset have missing values:",np.sum(train.isnull().sum()))
print("test dataset have missing values:",np.sum(test.isnull().sum()))
print("feature_all have missing values:",np.sum(features_all.isnull().sum()))

def count_missing_data(df):
    """ Counts the missing values for each features
        and display the Total and the Percentage
    """
    # Calculates the total number of missing values for each feature
    total = df.isnull().sum().sort_values(ascending=False)
    
    # Calculates the percentage of missing values for each features
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    
    # Combines the percentage and totals
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    # Only returns data that have missing values
    return missing_data[missing_data['Percent']>0]

print("Number of features with missing values: {}".format(len(count_missing_data(features_all))))

count_missing_data(features_all)
def fillna_numerica_average(df):
    """ Fills the numeric features that contain 
        NaN with the average in those columns
    """
    # Get features that contain missing values
    features_with_nan = count_missing_data(df).index 
    
    # Get numeric features that contain missing values
    numeric_missing_features = df[features_with_nan].dtypes[df.dtypes != "object"].index 
    
    # Fill each missing values for the numeric features with the corresponding median
    for feature in numeric_missing_features:
        df[feature].fillna(df[feature].median(), inplace=True)
        
fillna_numerica_average(features_all)
# Shows the missing values again, but seeing as we have filled all numerical features, the categorical features are left
list_missing_data = count_missing_data(features_all)
print("Number of features with missing values: {}".format(len(list_missing_data)))
print("Categorical features")
list_missing_data
# #log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])
plt.figure(figsize=(16,8))
sns.distplot(train.SalePrice)
plt.xlabel("SalePrice")
plt.ylabel("count")
plt.title("Sale_Price_distribution")
features_all = pd.get_dummies(features_all)
# Top 20 features with largest positive skew
import scipy.stats as stats
skewness = train.select_dtypes(include=np.number).apply(lambda x: stats.skew(x, nan_policy='propagate'))
skewness = pd.DataFrame(skewness.sort_values(ascending=False))
skewness.columns = ['Skew']
skewness.head(20)
numeric_feature_names = features_all.dtypes[features_all.dtypes != "object"].index

# # Selecting only the numeric features
from scipy.stats import skew

# # Calculates the skewedness of the features and then gets the features with a skewedness above a certain threshold
skewed_features = features_all[numeric_feature_names].apply(lambda x: skew(x, nan_policy='propagate'))
skewed_features = skewed_features[skewed_features > 0.70].index

features_all[skewed_features] = np.log1p(features_all[skewed_features])
scaler = StandardScaler()
scaler.fit(features_all[numeric_feature_names])
scaled = scaler.transform(features_all[numeric_feature_names])
from sklearn.feature_selection import VarianceThreshold
variance = VarianceThreshold(0.01)
features_all = variance.fit_transform(features_all)
target = train['SalePrice']
X_train = features_all[:train.shape[0]]
test_data = features_all[train.shape[0]:]
# Gradient Boosting
gradient = GradientBoostingRegressor(n_estimators = 3000, min_samples_leaf = 15, learning_rate = 0.05, max_features = 'sqrt',
                                      max_depth = 3, min_samples_split = 10, loss = 'huber')
gradient.fit(X_train, target)

# XGBoost
xgb = xgb.XGBRegressor(n_estimators = 30000, colsample_bytree = 0.2, gamma = 0.0,  reg_lambda = 0.6,  min_child_weight = 1.5,
                  reg_alpha = 0.9, learning_rate = 0.01, max_depth = 4, subsample = 0.2, seed = 42, silent = 1)
xgb.fit(X_train, target)
print('XGBoost CV r2: {}'.format(cross_val_score(xgb, X_train,target,cv=5,scoring='r2')))
print('Gradient CV r2: {}'.format(cross_val_score(gradient, X_train,target,cv=5,scoring='r2')))
rmse_xgb=np.sqrt(-cross_val_score(xgb, X_train,target,cv=5,scoring='neg_mean_squared_error'))
rmse_gradient=np.sqrt(-cross_val_score(gradient, X_train,target,cv=5,scoring='neg_mean_squared_error'))

print('XGBoost CV RMSE: {}'.format(rmse_xgb))
print('rmse_gradient CV RMSE: {}'.format(rmse_gradient))
# Gradient Boosting
y_pred_gradient = gradient.predict(X_train)

# XGBoost
y_pred_xgb = xgb.predict(X_train)
# Averaging scores using weights
y_pred = (0.70*y_pred_xgb + 0.30*y_pred_gradient)
# XGBoost
import xgboost as xgb

xgb = xgb.XGBRegressor(n_estimators = 30000, colsample_bytree = 0.2, gamma = 0.0,  reg_lambda = 0.6,  min_child_weight = 1.5,
                       reg_alpha = 0.9, learning_rate = 0.01, max_depth = 4, subsample = 0.2, seed = 42, silent = 1)
xgb.fit(X_train, target)
y_pred_xgb = xgb.predict(test_data)

# Gradient boosting
gradient = GradientBoostingRegressor(n_estimators = 3000, min_samples_leaf = 15, learning_rate = 0.05, max_features = 'sqrt',
                                      max_depth = 3, min_samples_split = 10, loss = 'huber', random_state = 42)
gradient.fit(X_train, target)
y_pred_gradient = gradient.predict(test_data)
y_pred = (.80*y_pred_xgb +.20*y_pred_gradient)
solution = pd.DataFrame({"id":test.Id, "SalePrice": np.expm1(y_pred)})
solution.to_csv("submission2.csv", index = False)
