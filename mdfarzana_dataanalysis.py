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
housing_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

housing_data_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

print(housing_data.head())

housing_data = pd.get_dummies(housing_data)

housing_data_test = pd.get_dummies(housing_data_test)

housing_data.shape
housing_data.describe().transpose()
housing_data = housing_data.fillna(method='ffill')

housing_data_test = housing_data_test.fillna(method='ffill')
X = housing_data.drop('SalePrice',axis =1).values

y = housing_data['SalePrice'].values



#splitting Train and Test 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
print(housing_data.dtypes)
#standardization scaler - fit&transform on train, fit only on test

from sklearn.preprocessing import StandardScaler

s_scaler = StandardScaler()

X_train = s_scaler.fit_transform(X_train.astype(np.float))

X_test = s_scaler.transform(X_test.astype(np.float))

housing_data_test = s_scaler.fit_transform(housing_data_test.astype(np.float))
# Multiple Liner Regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()  

regressor.fit(X_train, y_train)

#evaluate the model (intercept and slope)

print(regressor.intercept_)

print(regressor.coef_)

#predicting the test set result

y_pred = regressor.predict(X_test)

#put results as a DataFrame

coeff_df = pd.DataFrame(regressor.coef_, housing_data.drop('SalePrice',axis =1).columns, columns=['Coefficient']) 

coeff_df
import seaborn as sns

import matplotlib.pyplot as plt



# visualizing residuals

fig = plt.figure(figsize=(10,5))

residuals = (y_test- y_pred)

sns.distplot(residuals)
#compare actual output values with predicted values

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

df1 = df.head(10)

print(df1)

# evaluate the performance of the algorithm (MAE - MSE - RMSE)

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  

print('MSE:', metrics.mean_squared_error(y_test, y_pred))  

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('VarScore:',metrics.explained_variance_score(y_test,y_pred))
X_test.shape
housing_data_test.shape
y_pred_test = regressor.predict(housing_data_test)