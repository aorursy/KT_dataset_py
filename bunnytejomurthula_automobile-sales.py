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
car = pd.read_csv('/kaggle/input/automobile-dataset/Automobile_data.csv', na_values = ['?'])
car
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
car
car.isnull().sum()

car.info()
car.describe()
cat_col = [col for col in car.columns if car[col].dtype == 'object']

cat_col
num_col = [col for col in car.columns if col not in cat_col]

num_col
# Treating Null values in Categorical columns by replacing with most repeated value



for col in cat_col:

    car[col].fillna(car[col].mode()[0],inplace=True)
# Treating Null values in Numerical columns by replacing with mean values

for col in num_col:

    car[col].fillna(car[col].mean(),inplace=True)
car.info()
# Plotting distribution of numerical columns

for i in num_col:

    sns.distplot(car[i])

    plt.title(i)

    plt.show()
# Plotting categorical columns



for i in cat_col:

    plt.figure(figsize = (10, 6))

    sns.boxplot(x = i, y = 'price', data = car)

    plt.title(i)

    plt.xticks(rotation = 90)

    plt.show()
plt.figure(figsize = (12, 12))

sns.heatmap(car.corr(), annot = True)

plt.show()
# From the above we can drop columns which are having less or no signifiance



car = car.drop(['height', 'stroke', 'compression-ratio', 'peak-rpm'], axis = 1)



# Removing the same columns from numerical



for i in ['height', 'stroke', 'compression-ratio', 'peak-rpm']:

    num_col.remove(i)
# Nominal Encoding to reduce features



car['make'] = car['make'].map(car['make'].value_counts().to_dict())

car['fuel-system'] = car['fuel-system'].map(car['fuel-system'].value_counts().to_dict())
# Label encoding



from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



car[['num-of-doors', 'num-of-cylinders']] = car[['num-of-doors','num-of-cylinders']].apply(le.fit_transform)
car.head(10)
# One hot encoding for remaining categorical columns



# Drop first removes a column for each feature



car = pd.get_dummies(car, drop_first = True)
car.head()
# Standardizing the data 



from sklearn.preprocessing import StandardScaler 



sc = StandardScaler()



for i in num_col[:-1]:

    car[i] = car[i].astype('float64')

    car[i] =  sc.fit_transform(car[i].values.reshape(-1,1))
car.head()
# Split data using train_test_split



from sklearn.model_selection import train_test_split



X = car.drop('price', axis=1)

y = car['price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# Linear regression model for prediting price



from sklearn.linear_model import LinearRegression



model1 = LinearRegression()



model1.fit(X_train, y_train)



pred1 = model1.predict(X_test)
# Checking the r2_score, mean_absolute_error and mean_square_error



from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



print('Mean Squared error of LinearRegression as :', mean_squared_error(y_test, pred1))



print('Mean Absolute error of LinearRegression as :', mean_absolute_error(y_test, pred1))



print('R2 score of LinearRegression as :', r2_score(y_test, pred1))
# Decision Tree Regressor for price prediction and checking the score



from sklearn.tree import DecisionTreeRegressor



model2 = DecisionTreeRegressor()



model2.fit(X_train, y_train) 



pred2 = model2.predict(X_test)



print('Mean Squared error of DecisionTreeRegressor as :', mean_squared_error(y_test, pred2))



print('Mean Absolute error of DecisionTreeRegressor as :', mean_absolute_error(y_test, pred2))



print('R2 Score of DecisionTreeRegressor as :', r2_score(y_test, pred2))
# Random Forest Regressor for price prediction and checking the score



from sklearn.ensemble import RandomForestRegressor



model3 = RandomForestRegressor()



model3.fit(X_train, y_train) 



pred3 = model3.predict(X_test)



print('Mean Squared error of RandomForestRegressor as :', mean_squared_error(y_test, pred3))



print('Mean Absolute error of RandomForestRegressor as :', mean_absolute_error(y_test, pred3))



print('R2 Score of RandomForestRegressor as :', r2_score(y_test, pred3))