import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
from mlxtend.regressor import StackingCVRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/short-term-apartment-rentals/tidy_dc_airbnb.csv')
data.head()
data = data.iloc[:, 1:]
data.dtypes
data.host_response_rate = data.host_response_rate.str.rstrip('%').astype('float') / 100.0
data.host_response_rate = data.host_response_rate.fillna(data.host_response_rate.median())
data.host_acceptance_rate = data.host_acceptance_rate.str.rstrip('%').astype('float') / 100.0
data.host_acceptance_rate = data.host_acceptance_rate.fillna(data.host_acceptance_rate.median())
data.drop(['zipcode', 'latitude', 'longitude', 'price', 'city', 'state', 'maximum_nights'], axis = 1, inplace = True)
data['room_type'] = data['room_type'].astype('str')
sns.boxplot(data.host_acceptance_rate);
idx_drop = data.index[data.host_acceptance_rate == 0]
data.drop(idx_drop, inplace = True)
sns.boxplot(data.host_listings_count);
idx_drop = data.index[data.host_listings_count > 100]
data.drop(idx_drop, inplace = True)
sns.boxplot(data.minimum_nights);
idx_drop = data.index[(data.minimum_nights > 30)]
data.drop(idx_drop, inplace = True)
sns.boxplot(data.number_of_reviews);
idx_drop = data.index[data.number_of_reviews == 0]
data.drop(idx_drop, inplace = True)
data = pd.get_dummies(data, columns=['room_type'], prefix=["Room_Type"])
y = data['tidy_price']
X = data.drop(['tidy_price'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
sns.distplot(y_train);
Scaled_X_train = X_train.copy()
col_names = X_train.columns[:-3]
features = Scaled_X_train[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
Scaled_X_train[col_names] = features

Scaled_X_test = X_test.copy()
col_names = X_test.columns[:-3]
features = Scaled_X_test[col_names]
features = scaler.transform(features.values)
Scaled_X_test[col_names] = features
def model_pipeline(X_train, X_test, y_train, y_test, model, scoring ='neg_mean_absolute_error'):

    fit_mod = model.fit(X_train, y_train)
    y_pred = fit_mod.predict(X_test)
    score = mean_absolute_error(y_test, y_pred)
    
    return [fit_mod, y_pred, score]
mods = [XGBRegressor(), Lasso(), KNeighborsRegressor()]

models_score = []

for i,mod in enumerate(mods):
    models_score.append(model_pipeline(X_train, X_test, y_train, y_test, mod))
for result in models_score:
    print('Model: {0}, MAE: {1}'.format(type(result[0]).__name__, result[2]))
stack = StackingCVRegressor(regressors=(XGBRegressor(), Lasso(), KNeighborsRegressor()),
                            meta_regressor=Lasso(), cv=10,
                            use_features_in_secondary=True,
                            store_train_meta_features=True,
                            shuffle=False,
                            random_state=1)

stack.fit(X_train, y_train)
X_test.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']
pred = stack.predict(X_test)
score = mean_absolute_error(y_test, pred)
print('Model: {0}, MAE: {1}'.format(type(stack).__name__, score))