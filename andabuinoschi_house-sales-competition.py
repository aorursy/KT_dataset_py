import pandas as pd

import numpy as np

import os



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.preprocessing import OneHotEncoder



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn import ensemble

import xgboost
PATH = "../input/"

os.listdir(PATH)



train_df = pd.read_csv(PATH + "train.csv")

test_df = pd.read_csv(PATH + "test.csv")



print("Train: rows:{} cols:{}".format(train_df.shape[0], train_df.shape[1]))

print("Test:  rows:{} cols:{}".format(test_df.shape[0], test_df.shape[1]))





def missing_data(data):

    total = data.isnull().sum().sort_values(ascending=False)

    percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending=False)

    return np.transpose(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']))





print(missing_data(train_df))
for df in [train_df, test_df]:

    df['date'] = pd.to_datetime(df['date'])

    df['dayofweek'] = df['date'].dt.dayofweek

    df['weekofyear'] = df['date'].dt.weekofyear

    df['dayofyear'] = df['date'].dt.dayofyear

    df['quarter'] = df['date'].dt.quarter

    df['is_month_start'] = pd.to_numeric(df['date'].dt.is_month_start)

    df['month'] = df['date'].dt.month

    df['year'] = df['date'].dt.year

    df['is_weekend'] = pd.to_numeric(df['dayofweek'] >= 5)

    df['sqft_garden'] = df['sqft_lot'] - df['sqft_living']

    df['sqft_garden15'] = df['sqft_lot15'] - df['sqft_living15']



print(train_df.head())

train_df = train_df.drop('date', axis = 1)

test_df = test_df.drop('date', axis = 1)
train_df.columns.values
corr = train_df.corr()

sns.heatmap(corr)

plt.show()

corr['price'].sort_values(ascending=False)
# waterfront_ohe = OneHotEncoder()

# view_ohe = OneHotEncoder()

# condition_ohe = OneHotEncoder()

# grade_ohe = OneHotEncoder()



# X_waterfront = waterfront_ohe.fit_transform(train_df.waterfront.values.reshape(-1,1)).toarray()

# X_view = view_ohe.fit_transform(train_df.view.values.reshape(-1,1)).toarray()

# X_condition = condition_ohe.fit_transform(train_df.condition.values.reshape(-1,1)).toarray()

# X_grade = grade_ohe.fit_transform(train_df.grade.values.reshape(-1,1)).toarray()



# dfOneHot = pd.DataFrame(X_waterfront, columns = ["Waterfront_"+str(int(i)) for i in range(X_waterfront.shape[1])])

# train_df = pd.concat([train_df, dfOneHot], axis=1)



# dfOneHot = pd.DataFrame(X_view, columns = ["View_"+str(int(i)) for i in range(X_view.shape[1])])

# train_df = pd.concat([train_df, dfOneHot], axis=1)



# dfOneHot = pd.DataFrame(X_condition, columns = ["Condition_"+str(int(i)) for i in range(X_condition.shape[1])])

# train_df = pd.concat([train_df, dfOneHot], axis=1)



# dfOneHot = pd.DataFrame(X_grade, columns = ["Grade_"+str(int(i)) for i in range(X_grade.shape[1])])

# train_df = pd.concat([train_df, dfOneHot], axis=1)



# train_df.columns.values
X_train_df = train_df[[column for column in list(train_df.columns.values) if column not in ['price', 'date']]]
y_train_df = train_df[['price']]
X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train_df, test_size=0.33, random_state=42)



reg = LinearRegression(n_jobs = 8).fit(X_train_df, y_train_df)

predicted_prices = reg.predict(X_test)



print("The RMSE for Linear Regression model is {0}".format(sqrt(mean_squared_error(y_test, predicted_prices))))
X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train_df, test_size=0.33, random_state=42)



neigh = KNeighborsRegressor(n_neighbors=5, n_jobs = 8)

neigh.fit(X_train, y_train) 



predicted_prices = neigh.predict(X_test)



print("The RMSE for K-Nearest Neighbours Regressor model is {0}".format(sqrt(mean_squared_error(y_test, predicted_prices))))
X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train_df, test_size=0.33, random_state=42)



# Fit regression model

params = {'n_estimators': 2500, 'max_depth': 5, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}

clf = ensemble.GradientBoostingRegressor(**params)



clf.fit(X_train, y_train)

predicted_prices = clf.predict(X_test)

print("The RMSE for Gradient Boosting Regressor model is {0}".format(sqrt(mean_squared_error(y_test, predicted_prices))))



predicted_price = clf.predict(test_df)

test_df['price'] = predicted_price

# test_df[['id','price']].to_csv('../output/submission.csv', index=False)



test_df = test_df.drop('price', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train_df, test_size=0.33, random_state=42)





xgb = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.08, gamma=0, subsample=0.8,

                           colsample_bytree=1, max_depth=5, n_jobs = 8)

xgb.fit(X_train,y_train)

predictions = xgb.predict(X_test)

print("The RMSE for XGBoost Regression model is {0}".format(sqrt(mean_squared_error(y_test, predictions))))



predicted_price = xgb.predict(test_df)

test_df['price'] = predicted_price

# test_df[['id','price']].to_csv('../input/submission.csv', index=False)



test_df = test_df.drop('price', axis = 1)