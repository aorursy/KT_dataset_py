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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set_style("whitegrid")

plt.style.use("fivethirtyeight")
USAhousing = pd.read_csv('/kaggle/input/usa-housing/USA_Housing.csv')

USAhousing.head()
USAhousing.info()
USAhousing.describe()
USAhousing.columns
sns.pairplot(USAhousing)
sns.distplot(USAhousing['Price'])
sns.heatmap(USAhousing.corr(), annot=True)
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',

               'Avg. Area Number of Bedrooms', 'Area Population']]

y = USAhousing['Price']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

from sklearn import metrics

from sklearn.model_selection import cross_val_score



def cross_val(model):

    pred = cross_val_score(model, X, y, cv=10)

    return pred.mean()



def print_evaluate(true, predicted):  

    mae = metrics.mean_absolute_error(true, predicted)

    mse = metrics.mean_squared_error(true, predicted)

    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))

    r2_square = metrics.r2_score(true, predicted)

    print('MAE:', mae)

    print('MSE:', mse)

    print('RMSE:', rmse)

    print('R2 Square', r2_square)

    

def evaluate(true, predicted):

    mae = metrics.mean_absolute_error(true, predicted)

    mse = metrics.mean_squared_error(true, predicted)

    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))

    r2_square = metrics.r2_score(true, predicted)

    return mae, mse, rmse, r2_square
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression(normalize=True)

lin_reg.fit(X_train,y_train)
# print the intercept

print(lin_reg.intercept_)
pred = lin_reg.predict(X_test)
coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])

coeff_df
plt.scatter(y_test, pred)
sns.distplot((y_test - pred), bins=50);
print_evaluate(y_test, lin_reg.predict(X_test))
results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, pred) , cross_val(LinearRegression())]], 

                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

results_df
from sklearn.linear_model import RANSACRegressor



model = RANSACRegressor()

model.fit(X_train, y_train)



pred = model.predict(X_test)

print_evaluate(y_test, pred)
results_df_2 = pd.DataFrame(data=[["Robust Regression", *evaluate(y_test, pred) , cross_val(RANSACRegressor())]], 

                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
from sklearn.linear_model import Ridge



model = Ridge()

model.fit(X_train, y_train)

pred = model.predict(X_test)



print_evaluate(y_test, pred)
results_df_2 = pd.DataFrame(data=[["Ridge Regression", *evaluate(y_test, pred) , cross_val(Ridge())]], 

                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df
from sklearn.preprocessing import PolynomialFeatures



poly_reg = PolynomialFeatures(degree=2)

X_poly = poly_reg.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.4, random_state=101)



lin_reg = LinearRegression(normalize=True)

lin_reg.fit(X_train,y_train)

pred = lin_reg.predict(X_test)



print_evaluate(y_test, pred)
results_df_2 = pd.DataFrame(data=[["Polynomial Regression", *evaluate(y_test, pred), 0]], 

                            columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', 'Cross Validation'])

results_df = results_df.append(results_df_2, ignore_index=True)

results_df