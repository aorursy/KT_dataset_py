import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import re



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import r2_score



from math import sqrt



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from mlxtend.regressor import StackingRegressor



from sklearn.preprocessing import LabelEncoder

df = pd.read_excel('../input/flight-fare-prediction-mh/Data_Train.xlsx')

df.head()
df.shape
df.describe()
df.describe(include = ['O'])
print(df['Source'].unique())

df['Destination'].unique()
df.isnull().sum()
df.sort_values('Date_of_Journey', inplace = True)
df['year'] = pd.DatetimeIndex(df['Date_of_Journey']).year

df['month'] = pd.DatetimeIndex(df['Date_of_Journey']).month

df['Day'] = pd.DatetimeIndex(df['Date_of_Journey']).day
df['Additional_Info'].unique()
plt.figure(figsize = (10, 10))

plt.title('Price VS Additional Information')

plt.scatter(df['Additional_Info'], df['Price'])

plt.xticks(rotation = 90)

plt.xlabel('Information')

plt.ylabel('Price of ticket')
plt.figure(figsize = (10 , 10))

plt.title('Count of flights month wise')

sns.countplot(x = 'month', data = df)

plt.xlabel('Month')

plt.ylabel('Count of flights')
plt.figure(figsize = (10, 10))

plt.title('Price VS Airlines')

plt.scatter(df['Airline'], df['Price'])

plt.xticks(rotation = 90)

plt.xlabel('Airline')

plt.ylabel('Price of ticket')

plt.xticks(rotation = 90)
plt.figure(figsize = (10, 10))

plt.title('Count of flights with different Airlines')

sns.countplot(x = 'Airline', data = df)

plt.xlabel('Airline')

plt.ylabel('Count of flights')

plt.xticks(rotation = 90)
df['Airline'].replace(['Trujet', 'Vistara Premium economy'], 'Another', inplace = True)
df[df['Total_Stops'].isnull()]
df.dropna(axis = 0, inplace = True)
def convert_into_stops(X):

    if X == '4 stops':

        return 4

    elif X == '3 stops':

        return 3

    elif X == '2 stops':

        return 2

    elif X == '1 stop':

        return 1

    elif X == 'non stop':

        return 0
df['Total_Stops'] = df['Total_Stops'].map(convert_into_stops)
df.fillna(0, inplace  = True)

df['Total_Stops'] = df['Total_Stops'].apply(lambda x : int(x))
def flight_dep_time(X):

    '''

    This function takes the flight Departure time 

    and convert into appropriate format.

    '''

    if int(X[:2]) >= 0 and int(X[:2]) < 6:

        return 'mid_night'

    elif int(X[:2]) >= 6 and int(X[:2]) < 12:

        return 'morning'

    elif int(X[:2]) >= 12 and int(X[:2]) < 18:

        return 'afternoon'

    elif int(X[:2]) >= 18 and int(X[:2]) < 24:

        return 'evening'
df['flight_time'] = df['Dep_Time'].apply(flight_dep_time)
plt.figure(figsize = (10, 10))

plt.title('Count of flights according to departure time')

sns.countplot(x = 'flight_time', data = df)

plt.xlabel('Flight Time')

plt.ylabel('Count of flights')
def convert_into_seconds(X):

    '''

    This function takes the total time of flight from

    one city to another and converts it into the seconds.

    '''

    a = [int(s) for s in re.findall(r'-?\d+\.?\d*', X)]

    if len(a) == 2:

        hr = a[0] * 3600

        min = a[1] * 60

    else:

        hr = a[0] * 3600

        min = 0   

    total = hr + min

    return total



df['Duration(sec)'] = df['Duration'].map(convert_into_seconds)
plt.figure(figsize = (10, 10))

plt.title('Price VS duration of flights')

plt.scatter(df['Duration(sec)'], df['Price'])

plt.xlabel('Duartion in seconds')

plt.ylabel('Price of ticket')
df.corr()
df.shape
df = df.drop_duplicates()

df.shape
df['Additional_Info'].unique()
df['Additional_Info'].replace('No Info', 'No info', inplace = True)
sns.boxplot(df['Price'])
df.to_csv('cleaned_data.csv', index = None)
df = pd.get_dummies(df, columns = ['Airline', 'Source', 'Destination', 'Additional_Info', 'flight_time'])
pd.set_option('display.max_columns', 50)

df.head()
df.drop(['Date_of_Journey', 'Route', 'Dep_Time', 'Arrival_Time', 'Duration'], axis = 1, inplace = True)
df.to_csv('final_data.csv', index = None)
y = df['Price']

X = df.drop('Price', axis = 1)
s = StandardScaler()

X = s.fit_transform(X)
print(X.mean())

X.std()
# Splitting data into train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print("The size of training input is", X_train.shape)

print("The size of training output is", y_train.shape)

print(50 *'*')

print("The size of testing input is", X_test.shape)

print("The size of testing output is", y_test.shape)
y_train_pred = np.ones(X_train.shape[0]) * y_train.mean()

y_test_pred = np.ones(X_test.shape[0]) * y_train.mean()
print("Train Results for Baseline Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for Baseline Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))

print("R-squared: ", r2_score(y_test, y_test_pred))
k_range = list(range(1, 30))

params = dict(n_neighbors = k_range)

knn_regressor = GridSearchCV(KNeighborsRegressor(), params, cv = 10, scoring = 'neg_mean_squared_error')

knn_regressor.fit(X_train, y_train)
print(knn_regressor.best_estimator_)

knn_regressor.best_params_
y_train_pred =knn_regressor.predict(X_train) ##Predict train result

y_test_pred =knn_regressor.predict(X_test) ##Predict test result
print("Train Results for KNN Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for KNN Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))

print("R-squared: ", r2_score(y_test, y_test_pred))
params ={'alpha' :[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}

ridge_regressor =GridSearchCV(Ridge(), params ,cv =5,scoring = 'neg_mean_absolute_error', n_jobs =-1)

ridge_regressor.fit(X_train ,y_train)
print(ridge_regressor.best_estimator_)

ridge_regressor.best_params_
y_train_pred =ridge_regressor.predict(X_train) ##Predict train result

y_test_pred =ridge_regressor.predict(X_test) ##Predict test result
print("Train Results for Ridge Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for Ridge Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))

print("R-squared: ", r2_score(y_test, y_test_pred))
params ={'alpha' :[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}

lasso_regressor =GridSearchCV(Lasso(), params ,cv =15,scoring = 'neg_mean_absolute_error', n_jobs =-1)

lasso_regressor.fit(X_train ,y_train)
print(lasso_regressor.best_estimator_)

lasso_regressor.best_params_
y_train_pred =lasso_regressor.predict(X_train) ##Predict train result

y_test_pred =lasso_regressor.predict(X_test) ##Predict test result
print("Train Results for Lasso Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for Lasso Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))

print("R-squared: ", r2_score(y_test, y_test_pred))
depth  =list(range(3,30))

param_grid =dict(max_depth =depth)

tree =GridSearchCV(DecisionTreeRegressor(),param_grid,cv =10)

tree.fit(X_train,y_train)
print(tree.best_estimator_)

tree.best_params_
y_train_pred =tree.predict(X_train) ##Predict train result

y_test_pred =tree.predict(X_test) ##Predict test result
print("Train Results for Decision Tree Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for Decision Tree Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))

print("R-squared: ", r2_score(y_test, y_test_pred))
tuned_params = {'n_estimators': [100, 200, 300, 400, 500], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}

random_regressor = RandomizedSearchCV(RandomForestRegressor(), tuned_params, n_iter = 20, scoring = 'neg_mean_absolute_error', cv = 5, n_jobs = -1)

random_regressor.fit(X_train, y_train)
print(random_regressor.best_estimator_)

random_regressor.best_params_
y_train_pred = random_regressor.predict(X_train)

y_test_pred = random_regressor.predict(X_test)
print("Train Results for Random Forest Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for Random Forest Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))

print("R-squared: ", r2_score(y_test, y_test_pred))
tuned_params = {'max_depth': [1, 2, 3, 4, 5], 'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [100, 200, 300, 400, 500], 'reg_lambda': [0.001, 0.1, 1.0, 10.0, 100.0]}

model = RandomizedSearchCV(XGBRegressor(), tuned_params, n_iter=20, scoring = 'neg_mean_absolute_error', cv=5, n_jobs=-1)

model.fit(X_train, y_train)
print(model.best_estimator_)

model.best_params_
y_train_pred = model.predict(X_train)

y_test_pred = model.predict(X_test)
print("Train Results for XGBoost Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for XGBoost Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))

print("R-squared: ", r2_score(y_test, y_test_pred))
# Initializing models

ridge = Ridge()

lasso = Lasso()

tree = DecisionTreeRegressor()

knn = KNeighborsRegressor()



stack = StackingRegressor(regressors = [ridge, lasso, knn], meta_regressor = tree)

stack.fit(X_train, y_train)
print(stack.regr_)

stack.meta_regr_
y_train_pred = stack.predict(X_train)

y_test_pred = stack.predict(X_test)
print("Train Results for Stacking Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for Stacking Regressor Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))

print("R-squared: ", r2_score(y_test, y_test_pred))