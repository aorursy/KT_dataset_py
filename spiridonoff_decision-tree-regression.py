import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
day = pd.read_csv('../input/bike-sharing-dataset/day.csv')#,parse_dates=['dteday'])
day.describe()
hour = pd.read_csv('../input/bike-sharing-dataset/hour.csv',parse_dates=['dteday'])
hour.describe()
hour.columns
hour.isnull().sum().sum()
#Make a copy of hour to process
all_data = hour.copy()
categorical = ['yr','mnth','holiday','weekday','workingday','weathersit']
for feat in categorical:
    all_data[feat] = all_data[feat].apply(lambda x : str(x))
all_data.dtypes
all_data.drop(['dteday', 'instant', 'season'], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(all_data.drop(['casual','registered','cnt'],1), all_data['cnt'], test_size=0.33, random_state=42)
regr = DTR()
regr.fit(X_train,y_train)
y_pred = regr.predict(X_test)
mean_absolute_error(y_test,y_pred)
median_absolute_error(y_test,y_pred)
feature_importance = regr.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
#Relative Error
RE= abs(y_pred-y_test)/y_test
#Relative Error Histogram
sns.distplot(RE)
plt.show()
#Make a copy of day to process
day_data = day.copy()
categorical = ['yr','mnth','holiday','weekday','workingday','weathersit']
for feat in categorical:
    day_data[feat] = day_data[feat].apply(lambda x : str(x))
day_data.dtypes
day_data.drop(['dteday', 'instant', 'season'], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(day_data.drop(['casual','registered','cnt'],1), day_data['cnt'], test_size=0.33, random_state=42)
regr = DTR()
regr.fit(X_train,y_train)
y_pred = regr.predict(X_test)
mean_absolute_error(y_test,y_pred)
median_absolute_error(y_test,y_pred)
feature_importance = regr.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
#Relative Error
RE= abs(y_pred-y_test)/y_test
#Relative Error Histogram
sns.distplot(RE)
plt.show()
RE[RE>10]
max_err_idx = (abs(y_pred-y_test)/y_test).idxmax(axis=0)
max_err_pos = y_test.index.get_loc(max_err_idx)
y_test.iloc[max_err_pos],y_pred[max_err_pos]
(y_pred[max_err_pos]-y_test.iloc[max_err_pos])/y_test.iloc[max_err_pos]
day.loc[max_err_idx,'dteday']