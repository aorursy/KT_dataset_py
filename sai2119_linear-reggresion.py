

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



data = pd.read_csv("/kaggle/input/cardataset/data.csv")

data.head()
data.tail()
data.dtypes
data.shape
data.info()
data.describe()
data.isnull().sum()
data.dropna(inplace=True,axis=0)
data.isnull().sum()
sns.boxplot(data=data,orient='h',palette='Set2')
data['new msrp'] = np.log1p(data.MSRP)

data.drop('MSRP', axis=1, inplace=True)
data.head()
sns.boxplot(data=data,orient='h',palette='Set2')
q1, q3 = np.percentile(data['Popularity'],[25,75])

iqr = q3-q1

whisker = q3 + (1.5 * iqr)

print(whisker)
data['Popularity'] = data['Popularity'].clip(upper=whisker)
sns.boxplot(data=data,orient='h',palette='Set2')
# For Label Encoder data types need to be cat 

columns_to_convert=['Make','Model','Engine Fuel Type','Transmission Type','Driven_Wheels','Market Category','Vehicle Size','Vehicle Style']

data[columns_to_convert] = data[columns_to_convert].astype('category')
data.dtypes

 # Import label encoder

from sklearn import preprocessing

  

# label_encoder object knows how to understand word labels.

label_encoder = preprocessing.LabelEncoder()

  

# Encode labels in column 'species'.

for col in ['Make','Model','Engine Fuel Type','Transmission Type','Driven_Wheels','Market Category','Vehicle Size','Vehicle Style']: data[col] = label_encoder.fit_transform(data[col])

data.head()

data.shape
import seaborn as sns

corrmat = data.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(13,13))

g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap='rainbow',linewidths=3)

X = data.corr()

X['new msrp'].sort_values(ascending=False)
xy = ['new msrp','Engine HP','Year','Engine Cylinders']

data_ve = data[xy]

New_data = data.copy()

data_ve.head()

data = data_ve

target = "new msrp"



X = data[data.columns.difference([target])]

y = data['new msrp']

print(X.shape)

y.shape



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=124421)

from sklearn.linear_model import LinearRegression

from sklearn import metrics



linear = LinearRegression()

linear.fit(X_train, y_train)

#To retrieve the intercept:

print(linear.intercept_)

#For retrieving the slope:

print(linear.coef_)
y_test_predict = linear.predict(X_test)

print(y_test_predict)

y_train_predict= linear.predict(X_train)

y_train_predict
df = pd.DataFrame({'Actual': y_test, 'Predicted':y_test_predict})

df
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error



print(mean_squared_error(y_train,y_train_predict))

print(mean_squared_error(y_test,y_test_predict))



print(r2_score(y_train,y_train_predict))

print(r2_score(y_test,y_test_predict))



print(mean_absolute_error(y_train,y_train_predict))

print(mean_absolute_error(y_test,y_test_predict))
from sklearn import metrics

print(np.sqrt(metrics.mean_squared_error(y_train,y_train_predict)))

print(np.sqrt(metrics.mean_squared_error(y_test,y_test_predict)))

xyz = ['new msrp','Engine HP','Year','Engine Cylinders','Transmission Type','Engine Fuel Type','city mpg','Make','highway MPG','Market Category','Driven_Wheels']

data_all = New_data[xyz]

data_all.head()
# dcoupling



data = data_all

target = "new msrp"



X = data[data.columns.difference([target])]

y = data['new msrp']

print(X.shape)

y.shape







#Train test spilt



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=124421)



# Fitting



linear = LinearRegression()

linear.fit(X_train, y_train)

# Predict



y_test_predict = linear.predict(X_test)

print(y_test_predict)

y_train_predict= linear.predict(X_train)

y_train_predict



#To retrieve the intercept:

print(linear.intercept_)

#For retrieving the slope:

print(linear.coef_)





#test vs pred

df1 = pd.DataFrame({'Actual': y_test, 'Predicted':y_test_predict})

df1

#Metrics

from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

print("MSE of Train =", mean_squared_error(y_train,y_train_predict))

print('MSE of Test',mean_squared_error(y_test,y_test_predict) )

print('r2_score of Train ', r2_score(y_train,y_train_predict))

print('r2_score of Test ',r2_score(y_test,y_test_predict))

print('MAE of Train ',mean_absolute_error(y_train,y_train_predict) )

print('MAE of Test ',mean_absolute_error(y_test,y_test_predict))



#metrics MSE

from sklearn import metrics

print('sqrt MSE of Train',np.sqrt(metrics.mean_squared_error(y_train,y_train_predict)))



print('sqrt MSE of Test', np.sqrt(metrics.mean_squared_error(y_test,y_test_predict)))



import statsmodels.api as sm



x_train = sm.add_constant(X_train)

model = sm.OLS(y_train, x_train)

results = model.fit()

print ("r2/variance : ", results.rsquared)
from sklearn.model_selection import GridSearchCV

model = LinearRegression()

parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True,False]}

grid = GridSearchCV(model,parameters, cv=None)

grid.fit(X_train, y_train)

print ("r2 / variance : ", grid.best_score_)

print("Residual sum of squares: %.2f" % np.mean((grid.predict(X_test) - y_test) ** 2))
from sklearn.tree import DecisionTreeRegressor

#data



var = ['new msrp','Engine HP','Year','Engine Cylinders','Transmission Type','Engine Fuel Type','city mpg','Make','highway MPG','Market Category','Driven_Wheels']

data_for_decision_tree = New_data[var]

data_for_decision_tree.head()





# dcoupling



data =data_for_decision_tree

target = "new msrp"



X = data[data.columns.difference([target])]

y = data['new msrp']



#Train test spilt



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=124421)



# Fit regression model

regr_1 = DecisionTreeRegressor(max_depth=4)

regr_2 = DecisionTreeRegressor(max_depth=5)

regr_1.fit(X_train, y_train)

regr_2.fit(X_train, y_train)



# predicting a new value 



# test the output by changing values

y_pred1 = regr_1.predict(X_test)

y_pred2 = regr_2.predict(X_test)

# print the predicted price 

print('Value for max depth 4 =',y_pred1) 

print('Value for max depth 5 =',y_pred2) 

print('mean_squared_error max depth 4 =',mean_squared_error(y_test,y_pred1))

print('r2_score max depth 4 =',r2_score(y_test,y_pred1))

print('mean_squared_error sqrt max depth 4 =',np.sqrt(metrics.mean_squared_error(y_test,y_pred1)))

print('mean_squared_error for max depth 5 =',mean_squared_error(y_test,y_pred2))

print('r2_score max depth 5 =',r2_score(y_test,y_pred2))

print('mean_squared_error sqrt max depth 5 =',np.sqrt(metrics.mean_squared_error(y_test,y_pred2)))

dtm = DecisionTreeRegressor()



param_grid = {"criterion": ["mse", "mae"],

              "min_samples_split": [10, 20, 40],

              "max_depth": [2, 6, 8,10],

              "min_samples_leaf": [20, 40, 100],

              "max_leaf_nodes": [5, 20, 100],

              }



## Comment in order to publish in kaggle.



grid_cv_dtm = GridSearchCV(dtm, param_grid, cv=5)



grid_cv_dtm.fit(X_train, y_train)



print("R-Squared::{}".format(grid_cv_dtm.best_score_))

print("Best Hyperparameters::\n{}".format(grid_cv_dtm.best_params_))
# Fitting Random Forest Regression to the dataset 

# import the regressor 

from sklearn.ensemble import RandomForestRegressor 

  

 # create regressor object 

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 

  

# fit the regressor with x and y data 

regressor.fit(X_train, y_train)  
# predicting a new value 



# test the output by changing values

y_pred = regressor.predict(X_test)

# print the predicted price 

print('Value for regressor predict=',y_pred1) 

print('mean_squared_error regressor predict =',mean_squared_error(y_test,y_pred1))

print('r2_score regressor predict =',r2_score(y_test,y_pred1))

print('mean_squared_error regressor predict =',np.sqrt(metrics.mean_squared_error(y_test,y_pred1)))
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state = 42)

from pprint import pprint

# Look at parameters used by our current forest

print('Parameters currently in use:\n')

pprint(rf.get_params())

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(5, 110, num = 11)]

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

pprint(random_grid)
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train, y_train)
rf_random.best_params_
def evaluate(model,X_test, y_test):

    predictions = model.predict(X_test)

    errors = abs(predictions - y_test)

    mape = 100 * np.mean(errors / y_test)

    accuracy = 100 - mape

    print('Model Performance')

    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

    print('Accuracy = {:0.2f}%.'.format(accuracy))

    

    return accuracy

base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)

base_model.fit(X_train, y_train)

base_accuracy = evaluate(base_model, X_test, y_test )

rf_random =  RandomForestRegressor(n_estimators= 1400,min_samples_split= 5,min_samples_leaf= 1,max_features= 'sqrt',max_depth= 78,bootstrap= False,  random_state=42, n_jobs = -1)



rf_random.fit(X_train, y_train)



predictions =rf_random.predict(X_test)

# print the predicted price 

print('Value for regressor predict=',predictions) 

print('mean_squared_error regressor predict =',mean_squared_error(y_test,predictions))

print('r2_score regressor predict =',r2_score(y_test,predictions))

print('mean_squared_error regressor predict =',np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [80, 90, 100, 110],

    'max_features': [2, 3],

    'min_samples_leaf': [3, 4, 5],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [100, 200, 300, 1000]

}

# Create a based model

rf = RandomForestRegressor()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data

grid_search.fit(X_train, y_train)

grid_search.best_params_
best_grid = grid_search.best_estimator_

grid_accuracy = evaluate(best_grid, X_test, y_test)
rf_random =  RandomForestRegressor(n_estimators= 1000,min_samples_split= 8,min_samples_leaf= 3,max_features= 3,max_depth= 110,bootstrap= True,  random_state=42, n_jobs = -1)



rf_random.fit(X_train, y_train)



predictions =rf_random.predict(X_test)

# print the predicted price 

print('Value for regressor predict=',predictions) 

print('mean_squared_error regressor predict =',mean_squared_error(y_test,predictions))

print('r2_score regressor predict =',r2_score(y_test,predictions))

print('mean_squared_error regressor predict =',np.sqrt(metrics.mean_squared_error(y_test,predictions)))
import xgboost as xgb

from xgboost.sklearn import XGBRegressor
# Various hyper-parameters to tune

xgb1 = XGBRegressor()

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'objective':['reg:linear'],

              'learning_rate': [.03, 0.05, .07], #so called `eta` value

              'max_depth': [5, 6, 7],

              'min_child_weight': [4],

              'silent': [1],

              'subsample': [0.7],

              'colsample_bytree': [0.7],

              'n_estimators': [500]}



xgb_grid = GridSearchCV(xgb1,

                        parameters,

                        cv = 2,

                        n_jobs = -1,

                        verbose=True)



xgb_grid.fit(X_train, y_train)



print(xgb_grid.best_score_)

print(xgb_grid.best_params_)