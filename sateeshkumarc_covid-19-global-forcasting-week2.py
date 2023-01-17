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
#Read the data file
df =pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")
#View head
df.head()
#View tail
df.tail()
#View describe
df.describe()

#View info
df.info()
# Fill Nulls for Province_State Feature
df.Province_State= df.Province_State.fillna(df.Country_Region)
# Step 5- Rule # 5 - Check all the Data in Numeric
# Convert Date feature to datetime
df.Date= df.Date.apply(pd.to_datetime)
# Create new features from Date
df['Day_Of_The_Year'] = df['Date'].dt.day
df['Month_Of_The_Year'] = df['Date'].dt.month
df['Week_Of_The_Year '] = df['Date'].dt.week
df.drop("Date", inplace = True, axis =1)
df.info()

#Label Encode Province_State and Country_Region
#Label Encode city,toss_decision using Label Encoder class
from sklearn import preprocessing
le =preprocessing.LabelEncoder()

df['Province_State']=le.fit_transform(df['Province_State'].astype(str))
df['Country_Region']=le.fit_transform(df['Country_Region'].astype(str))
df.drop("Id",inplace=True, axis=1)

#Step 6 Selection of X & Y
    # Separate X and Y but there will be 2 Y's
X =df.drop(['ConfirmedCases', 'Fatalities'], axis = 1)
y_cc= df['ConfirmedCases']
y_fat= df['Fatalities']
from sklearn.model_selection import train_test_split
X_train_cc,X_test_cc, y_train_cc,y_test_cc = train_test_split(X, y_cc,test_size=0.3)
X_train_fat,X_test_fat,y_train_fat,y_test_fat = train_test_split(X, y_fat , test_size=0.3)

# Train Regressor - Linear Regresion
from sklearn.linear_model import LinearRegression
reg_cc = LinearRegression()
reg_cc.fit(X_train_cc, y_train_cc)
Y_pred_cc_lr = reg_cc.predict(X_test_cc)
print(Y_pred_cc_lr, "Predictions for CC")
reg_fat = LinearRegression()
reg_fat.fit(X_train_fat, y_train_fat)
Y_pred_fat_lr = reg_fat.predict(X_test_fat)
print(Y_pred_fat_lr, "Predictions for Fatalities")
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_cc = sqrt(mean_squared_error(y_test_cc, Y_pred_cc_lr))
rmse_fat = sqrt(mean_squared_error(y_test_fat, Y_pred_fat_lr))
print(rmse_cc, rmse_fat)
# Train Regressor - Decision Tree Regresion
from sklearn.tree import DecisionTreeRegressor
dt_cc = DecisionTreeRegressor()
dt_fat= DecisionTreeRegressor()
dt_cc.fit(X_train_cc, y_train_cc)
Y_pred_cc_dt = dt_cc.predict(X_test_cc)
print(Y_pred_cc_dt, "Predictions for CC")

dt_fat.fit(X_train_fat, y_train_fat)
Y_pred_fat_dt = dt_fat.predict(X_test_fat)
print(Y_pred_fat_dt, "Predictions for Fatalities")

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_cc = sqrt(mean_squared_error(y_test_cc, Y_pred_cc_dt))
rmse_fat = sqrt(mean_squared_error(y_test_fat, Y_pred_fat_dt))
print(rmse_cc, rmse_fat)
