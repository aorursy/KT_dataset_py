import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Path of the file to read
diabetes_file = '../input/MarketValue.csv'
data = pd.read_csv(diabetes_file)
data.head()
data.shape
data.drop(["name"], axis=1, inplace=True)
#Handling unnamed columns in python dataframe
data.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)
#1st column had no significance as it contains only ids so dropping that column
data.drop(["a"], axis=1, inplace=True)
data.drop(["rank"], axis=1, inplace=True)
data.head()
data.isnull().sum()
data['profits'].skew()

data.profits.isnull().values.any()
data.profits.isnull().sum()
data['profits'] = data['profits'].fillna((data['profits'].median()))

data.dtypes
obj_df = data.select_dtypes(include=['object']).copy()
obj_df.head()
data["country"] = data["country"].astype('category')
data["category"] = data["category"].astype('category')
data.dtypes
data["country"] = data["country"].cat.codes
data["category"] = data["category"].cat.codes
data.head()

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X=data.drop(['marketvalue'], axis=1)
Y=data['marketvalue']

x_train, x_test , y_train , y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42)
print('Training X Shape:', x_train.shape)
print('Training Y Shape:', y_train.shape)
print('Testing X Shape:', x_test.shape)
print('Testing Y Shape:', y_test.shape)
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()

linear_reg.fit(x_train,y_train)

y_predict=linear_reg.predict(x_test)
linear_reg.score(x_test,y_test)
y_predict
df1

a=np.array(y_test)
a
from sklearn import metrics
from sklearn.metrics import mean_squared_error,r2_score

mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)
r2_score = linear_reg.score(x_test, y_test)

print('Sqrt MSE : ',np.sqrt(mse))
print('R2 Score : ',r2_score)

linear_reg.rsquared



import matplotlib.pyplot as plt

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()