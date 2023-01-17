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
df = pd.read_csv(r'/kaggle/input/car-price-prediction/CarPrice_Assignment.csv')
df.shape
df.info()
df.isna().sum()
df.head()
print("No of Car Name :" + str(len(df['CarName'].unique())))
# Lets ignore CarName
df['price'].plot()
df.drop(['CarName','car_ID'] , inplace = True, axis = 1)

df.shape
import matplotlib.pyplot as plt 

import seaborn as sns 



plt.figure(figsize=(15,10))

sns.heatmap(df.corr() , cmap="YlGnBu", annot=True)
data = pd.get_dummies(df, drop_first =True)

data.head()
data.columns
df.corr()
X = data.drop('price' , axis = 1)

y = df['price']
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split( X,y , test_size = 0.3, random_state = 0) 

print(X_train.shape)

print(X_test.shape)
from sklearn.linear_model import LinearRegression



reg = LinearRegression().fit(X_train, y_train)
reg.score(X_train, y_train)
reg.coef_
reg.intercept_
y_pred = reg.predict(X_test)
from sklearn.metrics import mean_squared_error



MSE  = mean_squared_error(y_test, y_pred)

print("MSE :" , MSE)



RMSE = np.sqrt(MSE)

print("RMSE :" ,RMSE)



from sklearn.metrics import r2_score



r2 = r2_score(y_test, y_pred)

print("R2 :" ,r2)
from sklearn import linear_model

lasso  = linear_model.Lasso(alpha=1 , max_iter= 3000)



lasso.fit(X_train, y_train)

lasso.score(X_train, y_train)

y_pred_l = lasso.predict(X_test)

MSE  = mean_squared_error(y_test, y_pred_l)

print("MSE :" , MSE)



RMSE = np.sqrt(MSE)

print("RMSE :" ,RMSE)



r2 = r2_score(y_test, y_pred_l)

print("R2 :" ,r2)
from sklearn.linear_model import Ridge



ridge  = Ridge(alpha=0.1)
ridge.fit(X_train,y_train)
ridge.score(X_train, y_train)
y_pred_r = ridge.predict(X_test)
MSE  = mean_squared_error(y_test, y_pred_r)

print("MSE :" , MSE)



RMSE = np.sqrt(MSE)

print("RMSE :" ,RMSE)



r2 = r2_score(y_test, y_pred_r)

print("R2 :" ,r2)