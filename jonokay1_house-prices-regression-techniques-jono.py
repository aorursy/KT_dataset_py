# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

print("Training data size:{}".format(train_df.shape))
print("Validation data size:{}".format(test_df.shape))
# Any results you write to the current directory are saved as output.

# Hide warnings
import warnings
warnings.filterwarnings('ignore')

# Pick columns that have numerics
# data = train_df.select_dtypes(include=[np.number]).interpolate().dropna()
# data.columns

# Return everything as a Numeric
# pd.get_dummies(train_df)

# Overview of the data
# Using the describe method for the overview
train_df.describe()
# Visualising the data
# Using the correlation heatmap matrix to understand how the features are related
corr_matrix = train_df.corr() 
f,ax = plt.subplots(figsize= (12,9))
sns.heatmap(corr_matrix, vmax = 1, square = True, cmap = "RdYlGn_r")
# Visulaising with Scatter plots
sns.set()
columns = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'TotRmsAbvGrd', 'FullBath', 'YearBuilt']
sns.pairplot(train_df[columns], height = 2.5)
plt.show()
# Checking for missing data
missing = train_df.isnull().sum()
missing = missing[missing > 0]
percent = ((train_df.isnull().sum())*100/train_df.isnull().count())
percent = percent[percent>0]
missing.sort_values(inplace=True)
percent.sort_values(inplace=True)
missing
print(percent)
percent.plot.bar()

# Dropping columns missing more than 15% of the data.
train_df_missing = train_df.copy()
train_df_missing = train_df_missing.drop((percent[percent> 15]).index,1)
print(train_df_missing.shape)
train_df_missing.columns
dummy_data = pd.get_dummies(train_df_missing)
dummy_data = dummy_data.select_dtypes(include=[np.number]).interpolate().dropna()
dummy_data_= dummy_data.copy()
data_ = dummy_data_.drop(['SalePrice'], axis=1) # Drop SalePrice so it wont appear in the X_train later
# By droping SalePrice from X_train we ensure that the model is only seeing the target for the first time under y_train
# This ensures there is no over fitting, leading to poor results

# Transformation of data to improve performance and predicability
# Log transformation of data
dummy_data_['SalePrice'] = np.log(dummy_data_['SalePrice'])
dummy_data_['GrLivArea'] = np.log(dummy_data_['GrLivArea'])
data_['GrLivArea'] = np.log(data_['GrLivArea'])

print("Shape of data without Sale Price:{}".format(data_.shape))
print("Shape of data with Sale Price:{}".format(dummy_data_.shape))
# # Filling columns that have less than 15% data missing using the sklearn simpleimputer
# imputed_train = data_.copy()

# cols_with_missing = (col for col in imputed_train.columns 
#                                  if imputed_train[col].isnull().any())
# for col in cols_with_missing:
#     imputed_train[col + '_was_missing'] = imputed_train[col].isnull()

# # Imputation
# my_imputer = SimpleImputer()
# imputed_train = my_imputer.fit_transform(imputed_train)

# imputed_train
# # train_df_imputed = SimpleImputer().fit_transform(data)
# # train_df_imputed
#histogram and normal probability plot
from scipy import stats
from scipy.stats import norm
print("Plot of SalePrice data before transformation")
sns.distplot(dummy_data['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(dummy_data['SalePrice'], plot=plt)
print ("Plot of SalePrice data after transformation")
sns.distplot(dummy_data_['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(dummy_data_['SalePrice'], plot=plt)
# Preprocessing the test data, removing missing elements, and predicting the results
test_df_ = test_df.copy()
# Checking for missing data
tmissing = test_df_.isnull().sum()
tmissing = tmissing[tmissing > 0]
tpercent = ((test_df_.isnull().sum())*100/test_df_.isnull().count())
tpercent = tpercent[tpercent>0]
tmissing.sort_values(inplace=True)
tpercent.sort_values(inplace=True)
tmissing
print(tpercent)
tpercent.plot.bar()

# Dropping columns missing more than 15% of the data.

test_df_missing = test_df_.drop((tpercent[tpercent> 15]).index,1)
print("Size of training dataset after removing columns with more than 15% data missing is {}".format(train_df_missing.shape))
print("Size of validation dataset after removing columns with more than 15% data missing is {}".format(test_df_missing.shape))
tdummy_data = pd.get_dummies(test_df_missing)
# tdummy_data.columns
# Return only columns with a dtype
tdata = tdummy_data.select_dtypes(include=[np.number]).interpolate().dropna()
tdata.shape
# Data transformation to remove the positive skewness in the data
#applying log transformation
tdata['GrLivArea'] = np.log(tdata['GrLivArea'])

#Plotting the transformed data
sns.distplot(tdata['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(tdata['GrLivArea'], plot=plt)
# Columns that will be used for both the training and validation data

train_col = list(data_.columns)
test_col = list(tdata.columns)
# col=[]
# for i in train_col:
#     if i in test_col:
#         col.append(i)

# col

tcol = [i for i in train_col if i in test_col]

TrainData = data_[tcol]
ValidationData = tdata[tcol]
print("Processed Training data size:{}".format(TrainData.shape))
print("Processed Validation data size:{}".format(ValidationData.shape))
from sklearn.model_selection import train_test_split
tsize = 0.3
X_train, X_test, y_train, y_test = train_test_split(TrainData, dummy_data_.SalePrice, test_size=tsize, random_state=42)
# Number of training features
n_samples, n_features = X_train.shape

# Print out `n_samples`
print("Total Training Samples {} of data are: {}".format((1 - tsize), n_samples))

# Print out `n_features`
print("Total features are: {}".format(n_features))

# # Inspect training targets `y_train`
# print("Total training targets {} of data are: {}".format((1 - tsize), len(y_train)))

# Size of test 
print("Total test samples {} of data: {}".format((tsize),X_test.shape[0]))

# Inspect testing targets `y_test`
print("Total testing targets {} of data are: {}".format((tsize), len(y_test)))
# Selecting the best regression model
# Using sklearn pipeline to facilitate selecting of the best performance model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

pipeline = Pipeline([('normalizer', StandardScaler()), # Step 1 - Normalise the data
                     ('clf', LinearRegression(normalize=True)) # Step 2 - Classifier
                    ])
clfs = []
clfs.append(LinearRegression(normalize=True))
clfs.append(SGDRegressor(alpha = 3))
clfs.append(SVR())
clfs.append(DecisionTreeRegressor())
clfs.append(RandomForestRegressor())
clfs.append(GradientBoostingRegressor())
clfs.append(AdaBoostRegressor())


for classifier in clfs:
    pipeline.set_params(clf = classifier)
    scores = cross_validate(pipeline, X_train, y_train, cv=3, return_train_score = True)
    print('---------------------------------')
    print(str(classifier))
    print('-----------------------------------')
    for key, values in scores.items():
            print(key,' mean ', values.mean())
            print(key,' std ', values.std())
linearmodel = LinearRegression().fit(X_train,y_train)
y_pred = linearmodel.predict(X_test)
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))
# Plot outputs
plt.scatter(y_test, y_pred,  color='black')
plt.plot(y_test, y_pred, color='blue', linewidth=1)
gbr = GradientBoostingRegressor().fit(X_train,y_train)
gbr_pred = gbr.predict(X_test)
print('Variance score: %.2f' % r2_score(y_test, gbr_pred))
print("Mean squared error: %.2f"% mean_squared_error(y_test, gbr_pred))
# Plot outputs
plt.scatter(y_test, gbr_pred,  color='black')
plt.plot(y_test, gbr_pred, color='blue', linewidth=1)
# Dataframe of Model 1 results
result = X_test.copy()
result['Log SalePrice Prediction'] = y_pred
result['SalePrice Prediction'] = np.exp(y_pred)
result['Log SalePrice'] = y_test
result['Real SalePrice'] = np.exp(y_test)
result_show = result[['Id','Log SalePrice','Log SalePrice Prediction','Real SalePrice','SalePrice Prediction']]
result_show.head()
x = result.Id
j = plt.scatter(x, y_pred, color='blue')
k = plt.scatter(x, y_test, color ='red')
plt.legend((j,k),('Predicted SalePrice','Real SalePrice'), loc = 1)
plt.show()
x = result.Id
j = plt.scatter(x, np.exp(y_pred), color='blue')
k = plt.scatter(x, np.exp(y_test), color ='red')
plt.legend((j,k),('Predicted SalePrice','Real SalePrice'), loc = 1)
plt.show()
# Based on the Scatter plot above, the model still needs tweaking to improve its performance
# Dataframe of Model 2 results
result2 = X_test.copy()
result2['Log SalePrice Prediction'] = gbr_pred
result2['SalePrice Prediction'] = np.exp(gbr_pred)
result2['Log SalePrice'] = y_test
result2['Real SalePrice'] = np.exp(y_test)
result2_show = result[['Id','Log SalePrice','Log SalePrice Prediction','Real SalePrice','SalePrice Prediction']]
result2_show.head()
x = result2.Id
j = plt.scatter(x, gbr_pred, color='blue')
k = plt.scatter(x, y_test, color ='red')
plt.legend((j,k),('Predicted SalePrice','Real SalePrice'), loc = 1)
plt.show()
x = result2.Id
j = plt.scatter(x, np.exp(gbr_pred), color='blue')
k = plt.scatter(x, np.exp(y_test), color ='red')
plt.legend((j,k),('Predicted SalePrice','Real SalePrice'), loc = 1)
plt.show()
# Using the model to predict the sale price of the 
test_y_pred = gbr.predict(ValidationData)
test_y_pred
# Dataframe of Model 2 results
prove = ValidationData.copy()
prove['Log SalePrice Prediction'] = test_y_pred
prove['SalePrice'] = np.exp(test_y_pred)
prove_show = prove[['Id','SalePrice']]
prove_show.head()
prove_show.to_csv('jono_submission1.csv')
x = prove.Id
j = plt.scatter(x, np.exp(test_y_pred), color='blue')
# plt.legend(loc=1)
plt.show()