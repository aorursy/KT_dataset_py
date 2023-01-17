import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/BlackFriday.csv')
data.head()
data.describe()
data.dtypes
data.Age.value_counts()
data.City_Category.value_counts()
data.Stay_In_Current_City_Years.value_counts()
data.Marital_Status.value_counts()
data.Product_Category_1.value_counts()
data.isnull().sum()
# filling the null entries with 0



data.fillna(0, inplace=True)
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
# Encoding the categorical variables



label_encoder_gender = LabelEncoder()

data.iloc[:, 2] = label_encoder_gender.fit_transform(data.iloc[:, 2])
data.head()
# dummy encoding



df_age = pd.get_dummies(data.Age, columns=['Age'], drop_first=True, prefix='C')
df_age.head()
df_city = pd.get_dummies(data.City_Category, columns=['City_Category'], drop_first=True, prefix='C')
df_city.head()
df_stay = pd.get_dummies(data.Stay_In_Current_City_Years, columns=['Stay_In_Current_City_Years'], drop_first=True, prefix='C')
df_stay.head()
# Combining the original data set to with the dummy datasets



data_final = pd.concat([data, df_age, df_city, df_stay], axis=1)

data_final.head()
# Dropping the the original categorical columns 



data_final.drop(['User_ID', 'Product_ID', 'Age', 'City_Category', 'Stay_In_Current_City_Years'], axis=1, inplace=True)
data_final.head()
X = data_final.drop('Purchase', axis=1)

y = data_final.Purchase
X.head()
y.head()
from sklearn.model_selection import train_test_split

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Building regression model on the data to predict 'Purchase' 



regressor = RandomForestRegressor(n_estimators=100)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(r2_score(y_test, y_pred))