import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict
bike_df = pd.read_csv('/kaggle/input/london-bike-sharing-dataset/london_merged.csv')

bike_df.head()
# no of rows and columns

bike_df.shape
# information about database

bike_df.info()
# print some statistical data

bike_df.describe()
# changning time to datatime object

bike_df['timestamp'] = pd.to_datetime(bike_df['timestamp'], format='%Y-%m-%d %H:%M:%S')

type(bike_df['timestamp'].iloc[0])
# Checking for missing values

bike_df.isnull().sum()
# getting hour, month, year and days of week from timestamp column

bike_df['hour'] = bike_df['timestamp'].apply(lambda time : time.hour)

bike_df['month'] = bike_df['timestamp'].apply(lambda time : time.month)

bike_df['year'] = bike_df['timestamp'].apply(lambda time : time.year)

bike_df['day_of_week'] = bike_df['timestamp'].apply(lambda time : time.dayofweek)
# checking the dataframe

bike_df.head()
bike_df['day_of_week'].value_counts()
# plotting corelation metrix

fig, ax = plt.subplots(figsize= (12, 10))
sns.heatmap(bike_df.corr(), annot=True, ax=ax)
# renaming columns

bike_df.rename(columns={'cnt': 'bikes_count'}, inplace=True)
bike_df
# selecting row with max humidity

bike_df.iloc[bike_df['hum'].idxmax()]

# we can see that if the humidity is high bike count is low
# selecting the row with min humidity

bike_df.iloc[bike_df['hum'].idxmin()]
# dropping timestamp columns

bike_df.drop('timestamp', axis=1, inplace=True)
# seperate dependent and independent varaible 

x = bike_df.drop('bikes_count', axis=1)
y = bike_df['bikes_count']
# seperate train and test split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
# create an instance of decision tree regressor

tree_regressor = DecisionTreeRegressor()

# fit the model

tree_regressor.fit(x_train, y_train)

# predict with 

y_predict_test = tree_regressor.predict(x_test)
y_predict_train = tree_regressor.predict(x_train)
# Evaluation of train set
print('Evaluation of train set')
print('R-square coefficient of determintion: ', r2_score(y_train, y_predict_train))
print('Mean squared error: ', mean_squared_error(y_train, y_predict_train))
print('Root Mean squared error: ', np.sqrt(mean_squared_error(y_train, y_predict_train)))

# Evaluation of test set
print('\n \nEvaluation of test set')
print('R-square coefficient of determintion: ', r2_score(y_test, y_predict_test))
print('Mean squared error: ', mean_squared_error(y_test, y_predict_test))
print('Root Mean squared error: ', np.sqrt(mean_squared_error(y_test, y_predict_test)))

fig, ax = plt.subplots(figsize = (12, 10))
ax.scatter(y_test, y_predict_test, color = 'blue', edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Ground Truth vs Predicted")
plt.show()
# actual vs predicted value

actual_predict = pd.DataFrame(data = {'actual': y_test, 'predicted': y_predict_test})

actual_predict.head()
# parameters of decision tree 

param_grid = {"criterion": ["mse", "mae"],
              "min_samples_split": [10, 15, 20, 30, 40],
              "max_depth": [2, 4, 6, 8, 10, 11],
              "min_samples_leaf": [10, 20,30, 40, 60, 100],
              "max_leaf_nodes": [5, 20, 30, 100],
              }
# Randome search

random_search = RandomizedSearchCV(tree_regressor, param_grid, cv=5)

random_search.fit(x_test, y_test)
# print the r square and best paramater

print('R-square:', random_search.best_score_)
print('Best parameter values {}'.format(random_search.best_params_))