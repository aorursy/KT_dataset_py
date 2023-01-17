# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import math as mh

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# For visualizations

import seaborn as sns

import matplotlib.pyplot as plt

% config InlineBackend.figure_format = 'png'

% matplotlib inline



# For data parsing

from datetime import datetime



# For choosing attributes that have good gaussian distribution

from scipy.stats import shapiro



# Needed for getting parameters for models

from sklearn.cross_validation import LeaveOneOut

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV



# Models

from sklearn.svm import SVR, LinearSVR

from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier

from sklearn.linear_model import Ridge, Lasso

from sklearn import cluster

from sklearn.neighbors import KNeighborsClassifier



# For scaling/normalizing values

from sklearn.preprocessing import MinMaxScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print("Train :",train.shape)

print("Test:",test.shape)
# Calculate number of samples in training and test datasets

num_train = train.shape[0]

num_test = test.shape[0]

print(num_train, num_test)



# For feature engineering, combine train and test data

data = pd.concat((train.loc[:, "Open Date" : "P37"],

                  test.loc[:, "Open Date" : "P37"]), ignore_index=True)
data.tail()
#Get name of all headers of data frame

#list(data)

data.columns.values.tolist()
print(data.isnull().sum().T) #No null values in any column, so no imputation and removal of rows

missing_df = data.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.loc[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')



ind = np.arange(missing_df.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(12,18))

rects = ax.barh(ind, missing_df.missing_count.values, color='blue')

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()
# Convert date to days

# Have to drop date 

import time

from datetime import datetime as dt

# train

all_diff = []

for date in data["Open Date"]:

    diff = dt.now() - dt.strptime(date, "%m/%d/%Y")

    all_diff.append(int(diff.days/1000))



data['Days_from_open'] = pd.Series(all_diff)

print(data.head())
#Drop Open Date Column

data = data.drop('Open Date', 1)
# Plotting mean of P-variables over each city helps us see which P-variables are highly related to City

# since we are given that one class of P-variables is geographical attributes.

distinct_cities = train.loc[:, "City"].unique()



# Get the mean of each p-variable for each city

means = []

for col in train.columns[5:42]:

    temp = []

    for city in distinct_cities:

        temp.append(train.loc[train.City == city, col].mean())     

    means.append(temp)

    

# Construct data frame for plotting

city_pvars = pd.DataFrame(columns=["city_var", "means"])

for i in range(37):

    for j in range(len(distinct_cities)):

        city_pvars.loc[i+37*j] = ["P"+str(i+1), means[i][j]]

        

# Plot boxplot

plt.rcParams['figure.figsize'] = (18.0, 6.0)

sns.boxplot(x="city_var", y="means", data=city_pvars)



# From this we observe that P1, P2, P11, P19, P20, P23, and P30 are approximately a good

# proxy for geographical location.
# K Means treatment for city (mentioned in the paper)

def adjust_cities(data, train, k):

    

    # As found by box plot of each city's mean over each p-var

    relevant_pvars =  ["P1", "P2", "P11", "P19", "P20", "P23", "P30"]

    train = train.loc[:, relevant_pvars]

    

    # Optimal k is 20 as found by DB-Index plot    

    kmeans = cluster.KMeans(n_clusters=k)

    kmeans.fit(train)

    

    # Get the cluster centers and classify city of each data instance to one of the centers

    data['City Cluster'] = kmeans.predict(data.loc[:, relevant_pvars])

    del data["City"]

    

    return data



# Convert unknown cities in test data to clusters based on known cities using KMeans

data = adjust_cities(data, train, 20)
# The two categories of City Group both appear very frequently

plt.rcParams['figure.figsize'] = (6.0, 6.0)

sns.countplot(x='City Group', data=train, palette="Greens_d")
# One hot encode City Group

data = data.join(pd.get_dummies(data['City Group'], prefix="CG"))



# Since only n-1 columns are needed to binarize n categories, drop one of the new columns.  

# And drop the original columns.

data = data.drop(["City Group", "CG_Other"], axis=1)
#Check the data type of all columns

data.dtypes
#Check the type column 

# Two of the four Restaurant Types (DT and MB), are extremely rare

sns.countplot(x='Type', data=data, palette="Greens_d")
# One hot encode Restaurant Type

data = data.join(pd.get_dummies(data['Type'], prefix="T"))

 

# Drop the original column

data = data.drop(["Type"], axis=1)
data.head()
#Count distinct values for each column in Data frame

data.apply(lambda x: len(x.unique()))
# Scale all input features to between 0 and 1.

min_max_scaler = MinMaxScaler()

data = pd.DataFrame(data=min_max_scaler.fit_transform(data),columns=data.columns, index=data.index)
train.head()
#Revenue Distribution of Train Set

# Check distribution of revenue and log(revenue) (Other Transformation could be Sqrt Transformation)

plt.rcParams['figure.figsize'] = (16.0, 6.0)

pvalue_before = shapiro(train["revenue"])[1]

pvalue_after = shapiro(np.log(train["revenue"]))[1]

graph_data = pd.DataFrame(

        {

            ("Revenue\n P-value:" + str(pvalue_before)) : train["revenue"],

            ("Log(Revenue)\n P-value:" + str(pvalue_after)) : np.log(train["revenue"])

        }

    )

graph_data.hist()



#Shapiro Wilks test for normality

# log transform revenue as it is approximately normal. If this distribution for revenue holds in the test set,

# log transforming the variable before training models will improve performance vastly.

# However, we cannot be completely certain that this distribution will hold in the test set.

train["revenue"] = np.log(train["revenue"])
# Split into train and test datasets

train_processed = data[:num_train]

test_processed = data[num_train:]
from sklearn import cross_validation, linear_model,ensemble

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from sklearn.linear_model.stochastic_gradient import SGDRegressor

from sklearn.svm import SVR



regr = linear_model.LinearRegression()

regr.get_params()

import warnings

warnings.filterwarnings('ignore')
# build model

from sklearn import cross_validation,linear_model,ensemble

from sklearn.cross_validation import train_test_split

from sklearn import metrics

from sklearn.metrics import make_scorer, mean_squared_error

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from sklearn.linear_model.stochastic_gradient import SGDRegressor

from sklearn.svm import SVR



# simple regression

print("Simple regression")



#create linear regression model object

regr = linear_model.LinearRegression()

#regr.get_params() -- Check the list of paramters for the given model



# create a parameter grid: map the parameter names to the values that should be searched

parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}



def RMSE(y_true,y_pred):

    rmse = mh.sqrt(mean_squared_error(y_true, y_pred))

    print('RMSE: %2.3f' % rmse)

    return rmse



'''def R2(y_true,y_pred):    

     r2 = r2_score(y_true, y_pred)

     print('R2: %2.3f' % r2)

     return r2

'''

    

def two_score(y_true,y_pred):

    score = RMSE(y_true,y_pred) #set score here and not below if using MSE in GridCV

    #score = R2(y_true,y_pred)

    return score



my_score = make_scorer(two_score, greater_is_better=False) # change for false if using MSE



# instantiate the grid

grid = GridSearchCV(regr, parameters, cv=LeaveOneOut(train.shape[0]), scoring='mean_squared_error')



# fit the grid with data

grid.fit(train_processed, train["revenue"])



# Re-train on full training set using the best parameters found in the last step.

# examine the best model

print("Best score :",grid.best_score_)

print("Best params :",grid.best_params_)

print("Best estimator:",grid.best_estimator_)

regr.set_params(**grid.best_params_)

regr.fit(train_processed, train["revenue"])



# results

results_regr = regr.predict(test_processed)

results_regr_exp=np.exp(results_regr)

print(results_regr_exp)
submission_lin_reg = pd.DataFrame(columns=['Prediction'],index=test.index, data=results_regr_exp)

submission_lin_reg.index.name = 'Id'
submission_lin_reg.describe().astype(int)
# Ridge model

model_grid = [{'normalize': [True, False], 'alpha': np.logspace(0,10)}]

ridge_clf = Ridge()



# Use a grid search and leave-one-out CV on the train set to find the best regularization parameter to use.

grid = GridSearchCV(ridge_clf, model_grid, cv=LeaveOneOut(train.shape[0]), scoring='mean_squared_error')

grid.fit(train_processed, train["revenue"])



# Re-train on full training set using the best parameters found in the last step.

# examine the best model

print("Best score :",grid.best_score_)

print("Best params :",grid.best_params_)

print("Best estimator:",grid.best_estimator_)

ridge_clf.set_params(**grid.best_params_)

ridge_clf.fit(train_processed, train["revenue"])



# results_ridge = np.exp(ridge_clf.predict(test_processed))

results_ridge = ridge_clf.predict(test_processed)

results_ridge_exp=np.exp(results_ridge)

print(results_ridge_exp)
# Lasso model

model_grid = [{'normalize': [True, False], 'alpha': np.logspace(0,10)}]

lasso_clf = Lasso()



# Use a grid search and leave-one-out CV on the train set to find the best regularization parameter to use.

grid = GridSearchCV(lasso_clf, model_grid, cv=LeaveOneOut(train.shape[0]), scoring='mean_squared_error')

grid.fit(train_processed, train["revenue"])



# Re-train on full training set using the best parameters found in the last step.

print("Best score :",grid.best_score_)

print("Best params :",grid.best_params_)

print("Best estimator:",grid.best_estimator_)

lasso_clf.set_params(**grid.best_params_)

lasso_clf.fit(train_processed, train["revenue"])



#Predict the test set

results_lasso = lasso_clf.predict(test_processed)

results_lasso_exp = np.exp(results_lasso)

print(results_lasso_exp)
#SVR()

from sklearn.svm import SVR, LinearSVR



svr = SVR(C=1, epsilon=0.1)

svr.fit(train_processed, train["revenue"])

results_svm = svr.predict(test_processed)

results_svm_exp = np.exp(results_svm)

print(results_svm_exp)