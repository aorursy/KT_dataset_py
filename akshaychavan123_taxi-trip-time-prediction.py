import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import datetime



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import r2_score



from math import sqrt



from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from xgboost import XGBRegressor

from mlxtend.regressor import StackingRegressor



from sklearn.externals import joblib
df = pd.read_csv('../input/train.csv')
df.shape
df.columns
df.head(10)
df.dtypes[df.dtypes == 'object']
df.isnull().sum()
df.info()
df.describe()
df.describe(include = ['object'])
df.sort_values('TIMESTAMP',inplace = True)
df.head()
df['year'] = df['TIMESTAMP'].apply(lambda x :datetime.datetime.fromtimestamp(x).year) 

df['month'] = df['TIMESTAMP'].apply(lambda x :datetime.datetime.fromtimestamp(x).month) 

df['month_day'] = df['TIMESTAMP'].apply(lambda x :datetime.datetime.fromtimestamp(x).day) 

df['hour'] = df['TIMESTAMP'].apply(lambda x :datetime.datetime.fromtimestamp(x).hour) 

df['week_day'] = df['TIMESTAMP'].apply(lambda x :datetime.datetime.fromtimestamp(x).weekday()) 
df.head()
plt.figure(figsize = (10,10))

plt.pie(df['year'].value_counts(), labels = df['year'].value_counts().keys(),autopct = '%.1f%%')
plt.figure(figsize = (5,5))

plt.title('Count of trips per day of week')

sns.countplot(y = 'week_day', data = df)

plt.xlabel('Count')

plt.ylabel('Day')
plt.figure(figsize = (10,10))

plt.title('Count of trips per month')

sns.countplot(y = 'month', data = df)

plt.xlabel('Count')

plt.ylabel('Month')
plt.figure(figsize = (10,10))

plt.title('Count of trips per hour')

sns.countplot(y = 'hour', data = df)

plt.xlabel('Count')

plt.ylabel('Hours')
df['MISSING_DATA'].value_counts()
df.drop(df[df['MISSING_DATA'] == True].index, inplace = True)
df['MISSING_DATA'].unique()
df[df['POLYLINE'] =='[]']['POLYLINE'].value_counts()
df.drop(df[df['POLYLINE'] =='[]']['POLYLINE'].index, inplace = True)
df[df['POLYLINE'] =='[]']['POLYLINE'].value_counts()
df['Polyline Length'] = df['POLYLINE'].apply(lambda x : len(eval(x))-1)
df['Trip Time(sec)'] = df['Polyline Length'].apply(lambda x : x * 15)
df.head()
df['Trip Time(sec)'].describe()
df.groupby('week_day').mean()
df['DAY_TYPE'].isnull().sum()
df = pd.get_dummies(df, columns=['CALL_TYPE'])
df.shape
df = df.drop_duplicates()

print(df.shape)
df.to_csv('Cleaned_data.csv', index = None)
df = df.iloc[:50000]
df.shape
X = df[['Polyline Length', 'CALL_TYPE_A', 'CALL_TYPE_B', 'CALL_TYPE_C']]

y = df['Trip Time(sec)']
s = StandardScaler()

X = s.fit_transform(X)
print(np.mean(X))

np.std(X)
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.3)
print("The size of training input is", X_train.shape)

print("The size of training output is", y_train.shape)

print(50 *'*')

print("The size of testing input is", X_test.shape)

print("The size of testing output is", y_test.shape)
y_train_pred = np.ones(X_train.shape[0]) * y_train.mean() #Predicting the train results
y_test_pred = np.ones(y_test.shape[0]) * y_train.mean() #Predicting the test results
print("Train Results for Baseline Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_train.values, y_train_pred)))

print("R-squared: ", r2_score(y_train.values, y_train_pred))
print("Test Results for Baseline Model:")

print(50 * '-')

print("Root mean squared error: ", sqrt(mse(y_test, y_test_pred)))

print("R-squared: ", r2_score(y_test, y_test_pred))
k_range  =list(range(1,30)) 

param =dict(n_neighbors =k_range)

knn_regressor =GridSearchCV(KNeighborsRegressor(),param,cv =10)

knn_regressor.fit(X_train,y_train)
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
win_model = RandomForestRegressor(n_estimators = 200, min_samples_split = 2, min_samples_leaf = 1)

win_model.fit(X_train, y_train)

joblib.dump(win_model, 'winnig_model_random_forest.pkl')