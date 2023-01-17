# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("/kaggle/input/bluebook-for-bulldozers/trainandvalid/TrainAndValid.csv")

test = pd.read_csv("/kaggle/input/bluebook-for-bulldozers/Test.csv")
df.head()
df.describe()
df.info()
total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

f, ax = plt.subplots(figsize=(15, 6))

plt.xticks(rotation='90')

sns.barplot(x=missing_data.index, y=missing_data['Percent'])

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

missing_data.head()
df['YearMade'].unique()
df['datasource'].unique()
df['SalePrice'] = np.log(df['SalePrice'])
from sklearn.linear_model import LinearRegression



linear_regressor = LinearRegression()
from xgboost import XGBRegressor



xgb_regressor = XGBRegressor()
from sklearn.ensemble import RandomForestRegressor



random_forest_regressor = RandomForestRegressor()
from sklearn.neighbors import KNeighborsRegressor



k_neighbors_regressor = KNeighborsRegressor()
features1 = ['YearMade', 'datasource'] 

y = ['SalePrice']



X1 = df[features1]

y = df[y]
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()



df['state'] = encoder.fit_transform(df.state)

df['fiBaseModel'] = encoder.fit_transform(df.fiBaseModel)

df['fiProductClassDesc'] = encoder.fit_transform(df.state)

df['fiModelDesc'] = encoder.fit_transform(df.fiBaseModel)
features2 = ['YearMade', 'datasource', 'state', 'fiBaseModel']



X2 = df[features2]
features3 = ['YearMade', 'datasource', 'state', 'fiBaseModel', 'fiProductClassDesc' , 'fiModelDesc']



X3 = df[features3]
X_train1, X_val1, y_train1, y_val1 = train_test_split(X1, y, test_size=0.3)
X_train2, X_val2, y_train2, y_val2 = train_test_split(X2, y, test_size=0.3)
X_train3, X_val3, y_train3, y_val3 = train_test_split(X3, y, test_size=0.3)
linear_regressor.fit(X_train1, y_train1)



pred_y_val1 = linear_regressor.predict(X_val1)
score_lr_1 = np.sqrt(mean_squared_error(pred_y_val1, y_val1))

score_lr_1
xgb_regressor.fit(X_train1, y_train1)



pred_y_val1 = xgb_regressor.predict(X_val1)
score_xgb_1 = np.sqrt(mean_squared_error(pred_y_val1, y_val1))

score_xgb_1
random_forest_regressor.fit(X_train1, y_train1)



pred_y_val1 = random_forest_regressor.predict(X_val1)
score_rf_1 = np.sqrt(mean_squared_error(pred_y_val1, y_val1))

score_rf_1
k_neighbors_regressor.fit(X_train1, y_train1)



pred_y_val1 = k_neighbors_regressor.predict(X_val1)
score_kn_1 = np.sqrt(mean_squared_error(pred_y_val1, y_val1))

score_kn_1
models = pd.DataFrame({

    'Model': ['Linear Regression', 'XGB Regressor', 'Random Forest Regressior', 'KNeighbors Regressor'],

    'Score': [score_lr_1, score_xgb_1, score_rf_1, score_kn_1 ]})

models.sort_values(by='Score', ascending=False)
linear_regressor.fit(X_train2, y_train2)



pred_y_val2 = linear_regressor.predict(X_val2)
score_lr_2 = np.sqrt(mean_squared_error(pred_y_val2, y_val2))

score_lr_2
xgb_regressor.fit(X_train2, y_train2)



pred_y_val2 = xgb_regressor.predict(X_val2)
score_xgb_2 = np.sqrt(mean_squared_error(pred_y_val2, y_val2))

score_xgb_2
random_forest_regressor.fit(X_train2, y_train2)



pred_y_val2 = random_forest_regressor.predict(X_val2)
score_rf_2 = np.sqrt(mean_squared_error(pred_y_val2, y_val2))

score_rf_2
k_neighbors_regressor2 = KNeighborsRegressor()



k_neighbors_regressor2.fit(X_train2, y_train2)



pred_y_val2 = k_neighbors_regressor2.predict(X_val2)
score_kn_2 = np.sqrt(mean_squared_error(pred_y_val2, y_val2))

score_kn_2
models = pd.DataFrame({

    'Model': ['Linear Regression', 'XGB Regressor', 'Random Forest Regressior', 'KNeighbors Regressor'],

    'Score': [score_lr_2, score_xgb_2, score_rf_2, score_kn_2 ]})

models.sort_values(by='Score', ascending=False)
linear_regressor.fit(X_train3, y_train3)



pred_y_val3 = linear_regressor.predict(X_val3)
score_lr_3 = np.sqrt(mean_squared_error(pred_y_val3, y_val3))

score_lr_3
xgb_regressor.fit(X_train3, y_train3)



pred_y_val3 = xgb_regressor.predict(X_val3)
score_xgb_3 = np.sqrt(mean_squared_error(pred_y_val3, y_val3))

score_xgb_3
random_forest_regressor.fit(X_train3, y_train3)



pred_y_val3 = random_forest_regressor.predict(X_val3)
score_rf_3 = np.sqrt(mean_squared_error(pred_y_val3, y_val3))

score_rf_3
k_neighbors_regressor3 = KNeighborsRegressor()



k_neighbors_regressor3.fit(X_train3, y_train3)



pred_y_val3 = k_neighbors_regressor3.predict(X_val3)
score_kn_3 = np.sqrt(mean_squared_error(pred_y_val3, y_val3))

score_kn_3
models = pd.DataFrame({

    'Model': ['Linear Regression', 'XGB Regressor', 'Random Forest Regressior', 'KNeighbors Regressor'],

    'Score': [score_lr_3, score_xgb_3, score_rf_3, score_kn_3 ]})

models.sort_values(by='Score', ascending=False)
test.head()
test['fiModelDesc'].unique()
test['fiProductClassDesc'].unique()
test['fiBaseModel'].unique()
test['state'].unique()
test['fiModelDesc'] = encoder.fit_transform(test.fiModelDesc)

test['fiProductClassDesc'] = encoder.fit_transform(test.fiProductClassDesc)

test['fiBaseModel'] = encoder.fit_transform(test.fiBaseModel)

test['state'] = encoder.fit_transform(test.state)
features = ['datasource', 'YearMade', 'state' ,'fiBaseModel', 'fiProductClassDesc', 'fiModelDesc']



X = test[features]
pred = k_neighbors_regressor3.predict(X)

#pred = np.exp(pred)
pred.shape
pred1 = np.reshape(pred, (12457,))
pred1.shape
submission = pd.DataFrame({

        "Id": test["SalesID"],

        "SalePrice": np.exp(pred1)

    })



submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')

submission.head()