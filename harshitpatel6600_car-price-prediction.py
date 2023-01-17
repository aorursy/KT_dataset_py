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
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/car data.csv')
df.head()
df.shape
df.columns
df.isna().sum()

plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), cmap='Blues', annot=True)
categorical_features = [col for col in df.columns if df[col].dtypes == 'O']
categorical_features
for col in categorical_features:
    print(col, df[col].nunique())
unique_values = []
for col in categorical_features:
    unique_values.append(df[col].nunique())
unique_values
sns.set_style("white")
sns.barplot(unique_values, categorical_features, orient='h')
plt.title('Unique values of each Categorical values')
df.drop(['Car_Name'], axis=1, inplace=True)
categorical_features = [col for col in df.columns if df[col].dtypes == 'O']
categorical_features
X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']
print(X.shape)
print(y.shape)
X.head()
for col in categorical_features:
    print(col, X[col].unique())
X.head()
X = pd.get_dummies(X, drop_first=True)
X.head()
X['Current_Year'] = 2020
X['Number_of_years'] = X['Current_Year'] - X['Year']
X.drop(['Current_Year', 'Year'], axis=1, inplace=True)
X.head()
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = RandomForestRegressor()
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

min_samples_split = [2, 5, 10, 15, 100]

min_samples_leaf = [1, 2, 5, 10]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
rf = RandomizedSearchCV(estimator = model, 
                               param_distributions = random_grid,
                               scoring='neg_mean_squared_error', 
                               n_iter = 10, cv = 5, verbose=2, 
                               random_state=42, n_jobs = 1)

rf.fit(X_train,y_train)
predictions = rf.predict(X_test)
from sklearn import metrics

print('MAE:',round(metrics.mean_absolute_error(y_test, predictions),2))
print('MSE:',round(metrics.mean_squared_error(y_test, predictions),2))
print('RMSE:',round(np.sqrt(metrics.mean_squared_error(y_test, predictions)),2))
print('R2_score',round(metrics.r2_score(y_test, predictions),2))
Random_Forest_Regressor = { 'MAE': round(metrics.mean_absolute_error(y_test, predictions),2), 'MSE': round(metrics.mean_squared_error(y_test, predictions),2), 
                      'RMSE': round(np.sqrt(metrics.mean_squared_error(y_test, predictions)),2) , 'R2_score':round(metrics.r2_score(y_test, predictions),2)}
plt.figure(figsize=(8,6))
sns.scatterplot(y_test, predictions)
plt.xlabel('y_test')
plt.ylabel('Predictions')
plt.title('y_test vs Predictions (RandomForestRegressor)')
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
print('MAE:',round(metrics.mean_absolute_error(y_test, predictions),2))
print('MSE:',round(metrics.mean_squared_error(y_test, predictions),2))
print('RMSE:',round(np.sqrt(metrics.mean_squared_error(y_test, predictions)),2))
print('R2_score',round(metrics.r2_score(y_test, predictions),2))

Decision_Tree_Regressor = { 'MAE': round(metrics.mean_absolute_error(y_test, predictions),2), 'MSE': round(metrics.mean_squared_error(y_test, predictions),2), 
                      'RMSE': round(np.sqrt(metrics.mean_squared_error(y_test, predictions)),2) , 'R2_score':round(metrics.r2_score(y_test, predictions),2)}
plt.figure(figsize=(8,6))
sns.scatterplot(y_test, predictions)
plt.xlabel('y_test')
plt.ylabel('Predictions')
plt.title('y_test vs Predictions (DecisionTreeRegressor)')
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train, y_train)
predictions = tree.predict(X_test)
print('MAE:',round(metrics.mean_absolute_error(y_test, predictions),2))
print('MSE:',round(metrics.mean_squared_error(y_test, predictions),2))
print('RMSE:',round(np.sqrt(metrics.mean_squared_error(y_test, predictions)),2))
print('R2_score',round(metrics.r2_score(y_test, predictions),2))

Linear_Regression = { 'MAE': round(metrics.mean_absolute_error(y_test, predictions),2), 'MSE': round(metrics.mean_squared_error(y_test, predictions),2), 
                      'RMSE': round(np.sqrt(metrics.mean_squared_error(y_test, predictions)),2) , 'R2_score':round(metrics.r2_score(y_test, predictions),2)}
plt.figure(figsize=(8,6))
sns.scatterplot(y_test, predictions)

plt.xlabel('y_test')
plt.ylabel('Predictions')
plt.title('y_test vs Predictions (LinearRegression)')
from tomark import Tomark

data = [Random_Forest_Regressor, Decision_Tree_Regressor, Linear_Regression]

markdown = Tomark.table(data)
