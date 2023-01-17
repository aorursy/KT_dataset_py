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
# Importing required libraries

from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn import preprocessing

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import linear_model

from sklearn import svm

from sklearn import tree

import xgboost as xgb

from sklearn.ensemble import BaggingRegressor

import numpy as np 

import pandas as pd
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# You can see the first columns with “head” function. If you pass some number in it, it could show the first “n” number

train.head()
sum(train.isna().sum())
# the total number of na values “the missing” values on train dataframe is 6965
sum(test.isna().sum())
# the total number of na values “the missing” values on the test is 7000
# If you groupby any column and count you see the count of the some reccuring strings or number. 

# The count values gives how many times they reccure. If we divide them with total count, we can find the possibilities of them. 

# The method I used is a aggregate filling method to all na values in a for loop

for name in train.columns:

    x = train[name].isna().sum()

    if x > 0:

        val_list = np.random.choice(train.groupby(name).count().index, x, p=train.groupby(name).count()['Id'].values /sum(train.groupby(name).count()['Id'].values))

        train.loc[train[name].isna(), name] = val_list
# First we loop on train.columns and if this column has na values that means if x > 0, we randomly choice from the list that is the groupby dataframe indexes. 

# and the probabilities are the count of them divided total sum. 

# Therefore, we collect a list with “x” element in it with most probable elements of the this serie. 

# This way, we can fill the na values according to probabistic approach and simulation technics.

# Again same process is applied into test dataframe.

for name in test.columns:

    x = test[name].isna().sum()

    if x > 0:

        val_list = np.random.choice(test.groupby(name).count().index, x, p=test.groupby(name).count()['Id'].values /sum(test.groupby(name).count()['Id'].values))

        test.loc[test[name].isna(), name] = val_list
# In the end, the na values sum is zero both for train and test dataframe

sum(train.isna().sum())

sum(test.isna().sum())
# Then other process that is encoding the string values comes. 

# With the help of labelencoder in sklearn, I create a total for loop for all columns and easily we can label them all fastest way we can. 

# First for train columns
# First I need to concat the test and train data

train_df = train.drop('SalePrice',axis = 1)

data = pd.concat([train_df,test])

le = preprocessing.LabelEncoder()

for name in data.columns:

    if data[name].dtypes == "O":

        print(name)

        data[name] = data[name].astype(str)

        train[name] = train[name].astype(str)

        test[name] = test[name].astype(str)

        le.fit(data[name])

        train[name] = le.transform(train[name])

        test[name] = le.transform(test[name])
# Second for test columns

for name in test.columns:

    if test[name].dtypes == "O":

        test[name] = test[name].to_string()

        le.fit(test[name])

        test[name] = le.transform(test[name])
# First I tran my train data with random forest algorithm

# This is a proven algorithm with its success. First I try to see results about it

X = train.drop('SalePrice',axis = 1)

y = train['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Then I train the “X” data with “y” label and take the predictions from “X_test” data which is test data features

regr = RandomForestRegressor(max_depth=2, random_state=0)

regr.fit(X_train, y_train)

predictions = regr.predict(X_test)
# The result is: 2140642920.0111065
mean_squared_error(predictions, y_test)
# That seems very high. However, the log transformations change it to very low

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
# This random search is used for random forest algorithm. You can use it for all the other machine learning algorithms if you want

# Next, I extract PCA features with PCA analysis. The total column number is three
from sklearn.decomposition import PCA

pca = PCA(n_components=3)

principalComponents_train = pca.fit_transform(X)

principalComponents_test = pca.fit_transform(test)

sum(pca.explained_variance_ratio_)
# Then, I load these features into the “train” and “test” dataframe
train['component_1'] = [i[0] for i in principalComponents_train]

train['component_2'] = [i[1] for i in principalComponents_train]

train['component_3'] = [i[2] for i in principalComponents_train]

test['component_1'] = [i[0] for i in principalComponents_test]

test['component_2'] = [i[1] for i in principalComponents_test]

test['component_3'] = [i[2] for i in principalComponents_test]
# again some steps for random forest algorithm

X = train.drop('SalePrice',axis = 1)

y = train['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

regr = RandomForestRegressor(n_estimators = 400,min_samples_split = 2,min_samples_leaf = 1,max_features= 'sqrt',max_depth =None,bootstrap= False)

regr.fit(X, y)

predictions = regr.predict(X)

mean_squared_error(predictions, y)
# The error rate is 2.166095890410959
# This method is similar to ensemble learning. I use and bagging algorithm in the end. I had just used it. 

# for details you can search the function and library on the Google. 

# I was using 7 different regressor for machine learning table to use as ensemble learning
model_1 = RandomForestRegressor(n_estimators = 400,min_samples_split = 2,min_samples_leaf = 1,max_features= 'sqrt',max_depth =None,bootstrap= False)

model_1.fit(X, y)

predict_1 = model_1.predict(X)

model_2= linear_model.Ridge()

model_2.fit(X,y)

predict_2 =model_2.predict(X)

model_3 =KNeighborsRegressor(10,weights='uniform')

model_3.fit(X,y)

predict_3 = model_3.predict(X)

model_4 = linear_model.BayesianRidge()

model_4.fit(X,y)

predict_4 =model_4.predict(X)

model_5 = tree.DecisionTreeRegressor(max_depth=1)

model_5.fit(X,y)

predict_5 =model_5.predict(X)

model_6= svm.SVR(C=1.0, epsilon=0.2)

model_6.fit(X,y)

predict_6 = model_6.predict(X)

model_7 = xgb.XGBRegressor()

model_7.fit(X,y)

predict_7 = model_7.predict(X)
# Then, I collect them in an other dataframe
final_df = pd.DataFrame()

final_df['SalePrice'] = y

final_df['RandomForest'] = predict_1

final_df['Ridge'] = predict_2

final_df['Kneighboors'] = predict_3

final_df['BayesianRidge'] = predict_4

final_df['DecisionTreeRegressor'] = predict_5

final_df['Svm'] = predict_6

final_df['XGBoost'] = predict_7
# I loaded predictions into this dataframe. Next, I will use bagging algorithm for predictions

# Again, if you print the errors on the data, the most accurate is random forest
print(mean_squared_error(final_df['SalePrice'], predict_1))

print(mean_squared_error(final_df['SalePrice'], predict_2))

print(mean_squared_error(final_df['SalePrice'], predict_3))

print(mean_squared_error(final_df['SalePrice'], predict_4))

print(mean_squared_error(final_df['SalePrice'], predict_5))

print(mean_squared_error(final_df['SalePrice'], predict_6))

print(mean_squared_error(final_df['SalePrice'], predict_7))
# After that, I take the functions and label from this final dataframe and train it with BaggingRegressor

X_final = final_df.drop ('SalePrice', axis = 1) 

y_final = final_df ['SalePrice']

model_last = RandomForestRegressor () 

model_last.fit (X_final, y_final)

predict_final = model_last.predict (X_final)

final_dt = RandomForestRegressor ()                    

model_last = BaggingRegressor (base_estimator = final_dt, n_estimators = 40, random_state = 1, oob_score = True)

model_last.fit (X_final, y_final) 

pred_final = model_last.predict (X_final)

acc_oob = model_last.oob_score_ 

print (acc_oob)
mean_squared_error(predict_final, y_final)
# Test Case

# We predict previous model the test dataframe

test_predictions_1 = model_1.predict(test)

test_predictions_2 = model_2.predict(test)

test_predictions_3 = model_3.predict(test)

test_predictions_4 = model_4.predict(test)

test_predictions_5 = model_5.predict(test)

test_predictions_6 = model_6.predict(test)

test_predictions_7 = model_7.predict(test)
# Next, I create another dataframe for test results

test_final_df = pd.DataFrame()

test_final_df['RandomForest'] = test_predictions_1

test_final_df['Ridge'] = test_predictions_2

test_final_df['Kneighboors'] = test_predictions_3

test_final_df['BayesianRidge'] = test_predictions_4

test_final_df['DecisionTreeRegressor'] = test_predictions_5

test_final_df['Svm'] = test_predictions_6

test_final_df['XGBoost'] = test_predictions_7
# Finally, I predict the last dataframe with lastly trained model

last_predictions = model_last.predict(test_final_df)
# Then, I load the submission csv

submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
# Then matching the right values with right indexis

submission['SalePrice'] = last_predictions
# I changed this last_predictions with “test_predictions_1” variable. 

# Finally I write the csv file into kaggle platform. That is it. Then, you should find the output and submit it

submission.to_csv ('submission.csv', index = False)