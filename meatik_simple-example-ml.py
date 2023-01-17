import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime 

import sklearn # ML

from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
print('Data loaded!')
(market_train_df, news_train_df) = env.get_training_data()
print('Data obtained!')
market_train_df.drop(['universe'], axis=1, inplace=True)
# Adding daily difference
new_col = market_train_df["close"] - market_train_df["open"]
market_train_df.insert(loc=6, column="daily_diff", value=new_col)
market_train_df.head()
market_train_df.iloc[:, 3:].corr(method='pearson')
returns = market_train_df[market_train_df["returnsOpenNextMktres10"].abs()<0.20]
# Imputer to remove nans
imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-9999.99)
T = pd.DataFrame(imp.fit_transform(returns), columns=returns.columns)
print('Data loaded!')
# Define data to use for X and y
n = 1500000
X = T.iloc[:n, 3:-1]
y = T[T.columns[-1]][:n]
print(X.shape, y.shape)
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Save cols order for the prediction data
cols_order=X_train.columns

# Predict
regress = sklearn.tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None) 
regress.fit(X_train, y_train)
y_predicted=regress.predict(X_test)
print("Done")
mean_absolute_error(y_test, y_predicted)
for i in range(X.shape[1]):
    print("%s (%f)" % (X.columns[i], regress.feature_importances_[i]))
df_results = X_test
df_results.insert(loc=df_results.shape[1], column="y_real", value=y_test)
df_results.insert(loc=df_results.shape[1], column="y_pred", value=y_predicted)
df_results.head()
days = env.get_prediction_days()
print("Done")
def make_random_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0
def rfr_predictions(market, news, predictions_template_df):
    print(market["time"][0])
    copy=market.copy()
    # Adding daily difference
    new_col = copy["close"] - copy["open"]
    copy.insert(loc=6, column="daily_diff", value=new_col)
    # Getting columns used on the training only and reorder
    copy=copy[cols_order]
    # Replacing missing values
    copy = pd.DataFrame(imp.fit_transform(copy), columns=copy.columns)
    # Predicting
    y_predicted=regress.predict(copy)
    mn=min(y_predicted)
    mx=max(y_predicted)
    # Converting into the confidence value, from -1 to 1
    predictions_template_df.confidenceValue = [((y-(-0.25))/(0.25-(-0.25))*2-1) for y in y_predicted]
    print("Done")
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    rfr_predictions(market_obs_df, news_obs_df, predictions_template_df)
    env.predict(predictions_template_df)
print('Prediction finished!')
env.write_submission_file()
print([filename for filename in os.listdir('.') if '.csv' in filename])