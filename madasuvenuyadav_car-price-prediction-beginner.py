# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# splitting the data

from sklearn.model_selection import train_test_split



# regression models

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import AdaBoostRegressor



# model evaluation metrics

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



car_df = pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/car data.csv')
# getting first five rows from car_df dataframe

car_df.head()
# getting Statistical info about the numerical columns in the data

car_df.describe()
# getting info about the data

car_df.info()
# we know that this dataset does not have any null values but let us clarify again in other method

car_df.isnull().sum()
# we will drop 'Car_Name' column from our car_df dataframe

car_df.drop('Car_Name', axis=1, inplace=True)
# converting categorical data columns like ['Fuel_Type', 'Seller_Type', 'Transmission'] into numerical columns

car_df = pd.get_dummies(data=car_df, drop_first=True)
# checking the columns datatypes

car_df.dtypes
car_df.head()
X = car_df.drop('Selling_Price', axis=1) # features or independent variables 

y = car_df['Selling_Price'] # outcome or target or dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# lets us put all our regression models into dictionary



models = {'Linear' : LinearRegression(),

          'RandomForest' : RandomForestRegressor(),

          'DecisionTree' : DecisionTreeRegressor(),

          'GradientBoosting' : GradientBoostingRegressor(),

          'AdaBoost' : AdaBoostRegressor()}
def fit_and_score(models, X_train, X_test, y_train, y_test):

    

    np.random.seed(42)

    

    model_scores = {}

    

    for name, model in models.items():

        

        model.fit(X_train, y_train)

        

        model_scores[name] = model.score(X_test, y_test)

        

    return model_scores
model_scores = fit_and_score(models, X_train, X_test, y_train, y_test)
model_scores
compare_models = pd.DataFrame(model_scores, index=['accuracy'])

compare_models.T.plot(kind = 'bar')
gradient_boost_model = GradientBoostingRegressor()

gradient_boost_model.fit(X_train, y_train)

gradient_boost_model.score(X_test, y_test)
predictions = gradient_boost_model.predict(X_test)
# model predictions

predictions[:5]
# Actuals

y_test[:5]
plt.scatter(y_test, predictions)

plt.xlabel('Actual values')

plt.ylabel('Model predicted values')

plt.title('RESIDUALS : Actuals vs Predicted')

plt.show()
# calculating mean_square_error, mean_absolute_error and r2_score

mse = mean_squared_error(y_test, predictions)

mae = mean_absolute_error(y_test, predictions)

r2_score = r2_score(y_test, predictions)
print("Mean Squared Error : ", mse)

print("Mean Absolute Error : ", mae)

print("R2_score : ", r2_score)
# Residual plot : MAKE SURE ITS LOOKS LIKE NORMAL DISTRIBUTION

sns.distplot((y_test - predictions), bins=50)
X_test[:1]
custom_data_prediction = gradient_boost_model.predict([[2020, 14.0, 80000, 0, 0, 1, 0, 1]])
custom_data_prediction