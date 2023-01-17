# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', None)

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

import category_encoders as ce

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from eli5.sklearn.explain_prediction import explain_prediction_linear_regressor

from eli5.sklearn.explain_prediction import explain_prediction_tree_regressor

import seaborn as sns

import matplotlib.pyplot as plt 

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Path of the file to read

iowa_file_path = '/kaggle/input/housing/AmesHousing.csv'

home_data = pd.read_csv(iowa_file_path)

home_data_copy = home_data.copy()



# Create target object and call it y

y = home_data.SalePrice

# Create X

X = home_data[home_data.columns[:-1]]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

val_X.head()
train_df = train_X.copy()

train_df['SalePrice'] = train_y

val_df = val_X.copy()

train_df.to_csv('train.csv', index=False)

val_df.to_csv('test.csv', index=False)



target = pd.DataFrame({'Order': val_df.Order, 'SalePrice': val_y})

target.to_csv('target.csv', index=False)
plt.scatter(val_X["Year Built"], val_y, color = "darkblue", alpha=.8)

plt.title("Sale Price vs Year Built (Validation set)")

plt.xlabel("Year Built")

plt.ylabel("Sale Price")

plt.show()
# Turn categorical features into numeric using count encoding

cat_features = home_data.select_dtypes(exclude=['int64','float64']).columns

non_cat_features = home_data.select_dtypes(include=['int64','float64']).columns[:-1]

count_enc = ce.CountEncoder(cols=cat_features)



# Learn encoding from the training set

count_enc.fit(train_X[cat_features])



# Apply encoding to the train and validation sets

train_X = train_X[non_cat_features].join(count_enc.transform(train_X[cat_features]).add_suffix('_count'))

val_X = val_X[non_cat_features].join(count_enc.transform(val_X[cat_features]).add_suffix('_count'))
to_plot = val_X.copy()

to_plot['cost_over_150000'] = val_y >= 150000

sns.pairplot(to_plot, vars=['Year Built', 'Overall Qual', 'Overall Cond'], hue='cost_over_150000')
# Deal with missing values

train_X = train_X.fillna(method='bfill')

val_X = val_X.fillna(method='bfill')
# Define the model

l_model = LinearRegression()

# fit the model

l_model.fit(train_X, train_y)



preds = l_model.predict(val_X)



# Calculate the mean absolute error of the Random Forest model on the validation data

l_val_mae = mean_absolute_error(val_y, preds)

l_val_rmse = np.sqrt(mean_squared_error(val_y, preds))

print("Validation MAE for Linear Model: {}".format(l_val_mae))

print("Validation RMSE for Linear Model: {}".format(l_val_rmse))
# Define the model

rf_model = RandomForestRegressor(random_state=1)

# fit the model

rf_model.fit(train_X, train_y)



preds = rf_model.predict(val_X)



# Calculate the mean absolute error of the Random Forest model on the validation data

rf_val_mae = mean_absolute_error(val_y, preds)

rf_val_rmse = np.sqrt(mean_squared_error(val_y, preds))

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

print("Validation RMSE for Random Forest Model: {}".format(rf_val_rmse))
import xgboost as xgb



xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 100)

xg_reg.fit(train_X, train_y)



preds = xg_reg.predict(val_X)



xgb_val_mae = mean_absolute_error(val_y, preds)

xgb_val_rmse = np.sqrt(mean_squared_error(val_y, preds))

print("Validation MAE for XGBoost model: {}".format(xgb_val_mae))

print("Validation RMSE for XGBoost model: {}".format(xgb_val_rmse))
my_predictions = pd.DataFrame({'Order': val_df.Order, 'SalePrice': preds})

my_predictions.to_csv('submission.csv', index=False)