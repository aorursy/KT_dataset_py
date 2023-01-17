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
##Training & Testing Dataset
train = pd.read_csv('/kaggle/input/food-demand-forecasting/train.csv')
test = pd.read_csv('/kaggle/input/food-demand-forecasting/test.csv')
fullfil_center = pd.read_csv('/kaggle/input/food-demand-forecasting/fulfilment_center_info.csv')
meal_info = pd.read_csv('/kaggle/input/food-demand-forecasting/meal_info.csv')
train.head()
fullfil_center.head()
meal_info.head()
data = pd.merge(train, fullfil_center, on='center_id')
data.head()
data.tail()
all_data = pd.merge(data, meal_info, on='meal_id')
all_data.head()
test.head()
all_data.shape
test.shape
##Statistical Summary of data
all_data.describe()
##Information of training data
all_data.info()
test.info()
##Checking Null values
all_data.isnull().sum()
##EDA
import matplotlib.pyplot as plt
plt.scatter(all_data['base_price'], all_data['checkout_price'])
plt.title('Scatter plot Base Price Vs Checkout Price')
plt.xlabel('Base Price')
plt.ylabel('Checkout Price')
plt.show()
plt.scatter(all_data['base_price'], all_data['num_orders'])
plt.title('Scatter plot Base Price Vs Number of Orders')
plt.xlabel('Base Price')
plt.ylabel('Number of Orders')
plt.show()
plt.scatter(all_data['checkout_price'], all_data['num_orders'])
plt.title('Scatter plot Checkout Price Vs Number of Orders')
plt.xlabel('Checkout Price')
plt.ylabel('Number of Orders')
plt.show()
##Distribution Plot
import seaborn as sns
sns.distplot(all_data['base_price'])
sns.distplot(all_data['checkout_price'])
sns.boxplot(all_data['num_orders'])
##Correlation Plot
import seaborn as sns
plt.figure(figsize=(30,15))
correlation = all_data.corr()
sns.heatmap(correlation, annot=True)
data_cp = all_data.copy()
data_cp.head()
# Label Encoding
from sklearn.preprocessing import LabelEncoder

lb_enc = LabelEncoder()
data_cp["make_Cent_type"] = lb_enc.fit_transform(data_cp["center_type"])
data_cp[["center_type", "make_Cent_type"]].head(10)
data_cp["make_category"] = lb_enc.fit_transform(data_cp["category"])
data_cp[["category", "make_category"]].head(10)
data_cp["make_cuisine"] = lb_enc.fit_transform(data_cp["cuisine"])
data_cp[["cuisine", "make_cuisine"]].head(10)
cp_data = data_cp.drop(['center_type','category','cuisine'], axis=1)
cp_data.head()
##Traning and Testing data spliting
X = cp_data.drop('num_orders', axis=1)
y = cp_data['num_orders']
y.shape
X.shape
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
###Linear Regression Model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)
print('Coefficient of model :', lin_reg_model.coef_)
print('Intercept of model :',lin_reg_model.intercept_)
# Root Mean Squared Error on training dataset
predict_train = lin_reg_model.predict(X_train)
rmse_train = mean_squared_error(y_train,predict_train)**(0.5)
print('\nRMSE on train dataset : ', rmse_train)
## prediction on test data splitting from metadata
predict_test = lin_reg_model.predict(X_test)
rmse_test = mean_squared_error(y_test,predict_test)**(0.5)
print('\nRMSE on test dataset : ', rmse_test)
# ## prediction on test data
# predict_test = lin_reg_model.predict(test)
# rmse_test = mean_squared_error(test.id,predict_test)**(0.5)
# print('\nRMSE on test dataset : ', rmse_test)
dec_reg_model = DecisionTreeRegressor(random_state=1)
dec_reg_model.fit(X_train, y_train)
## prediction on test data spliting of metadata
x_pred_dec = dec_reg_model.predict(X_test)
print("Mean Squared Log Error is ", mean_squared_log_error(y_test, x_pred_dec))
print("Root Mean Squared Error is ", mean_squared_error(y_test, x_pred_dec)**(0.5))
# ## prediction on test data
# x_pred_dec = dec_reg_model.predict(test)
# print("Mean Squared Log Error is ", mean_squared_log_error(test.id, x_pred_dec))
# print("Root Mean Squared Error is ", mean_squared_error(test.id, x_pred_dec)**(0.5))
##RandomForest Regressor
ran_reg_model = RandomForestRegressor(random_state=1)
ran_reg_model.fit(X_train, y_train)
## prediction on test data splitting from metadata
x_pred_ran = ran_reg_model.predict(X_test)
print("Mean Squared Log Error is ", mean_squared_log_error(y_test, x_pred_ran))
print("Root Mean Squared Error is ", mean_squared_error(y_test, x_pred_ran)**(0.5))
# ## prediction on test data
# x_pred_ran = ran_reg_model.predict(test)
# print("Mean Squared Log Error is ", mean_squared_log_error(test.id, x_pred_ran))
# ## prediction on test data
# x_pred_ran = ran_reg_model.predict(test)
# print("Root Mean Squared Error is ", mean_squared_error(test.id, x_pred_ran)**(0.5))
# ##File creation for submission
# final_out = pd.DataFrame({'id': test.id,'num_orders': x_pred_dec})
# final_out.to_csv('submission.csv', index=False)
