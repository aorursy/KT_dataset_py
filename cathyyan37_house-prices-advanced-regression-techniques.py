import numpy as np
import pandas as pd
# import train and test sets
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print(train.columns)
# splitting the data
from sklearn.model_selection import train_test_split

train = pd.get_dummies(train) #one-hot encoding
train.fillna(axis=0, method='ffill', inplace=True) # will fill NaN's with a more sophisticated method later

X = train.drop('SalePrice', axis=1)
y = train['SalePrice']
# print(X.head())
# print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# fitting the model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)
# get most important features
importance = model.feature_importances_
important_features = pd.DataFrame(importance, columns=['Importance']).reset_index()
sorted_important_features = important_features.sort_values('Importance', ascending=False)
print(sorted_important_features.head(10))

# function for retrieving the top five most important features
indices = []
def retrieve_5 (df):
    i = 0
    while i < 5:
        val = df.iloc[i, 0]
        indices.append(val)
        i += 1
    return indices

important = retrieve_5(sorted_important_features)
print(important)

train_important = train.iloc[:, important]
print(train_important.columns)
# fitting the model
from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(X_train, y_train)
# get most important features
importance_boost = model.feature_importances_
important_features_boost = pd.DataFrame(importance_boost, columns=['Importance']).reset_index()
sorted_important_features_boost = important_features_boost.sort_values('Importance', ascending=False)
print(sorted_important_features_boost.head(10))

indices = []
important_boost = retrieve_5(sorted_important_features_boost)
print(important_boost)

train_important_boost = train.iloc[:, important_boost]
print(train_important_boost.columns)

