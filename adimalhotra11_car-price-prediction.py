# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/car data.csv')
df.head(10)
df.shape
df.columns
df.Seller_Type.unique()
df.isnull().sum()
features = ['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type',

       'Seller_Type', 'Transmission', 'Owner']
import matplotlib.pyplot as plt

import seaborn as sns
df.info()
from sklearn.preprocessing import OneHotEncoder

df['Fuel_Type'].unique()
df.Fuel_Type.replace(regex={"Petrol":"0","Diesel":"1","CNG":"2"},inplace=True)

df.Seller_Type.replace(regex={"Dealer":"0","Individual":"1"},inplace=True)

df.Transmission.replace(regex={"Manual":"0","Automatic":"1"},inplace=True)

df[["Fuel_Type","Seller_Type","Transmission"]]=df[["Fuel_Type","Seller_Type","Transmission"]].astype(int)

df.info()
sns.pairplot(df , diag_kind="kde" ,diag_kws=dict(shade=True, bw=.03, vertical=False))
df.corr()
sns.heatmap(df.corr(),

    annot=True)


X=df.drop(["Selling_Price",'Car_Name'],axis=1)

y=df.Selling_Price
X.head()
y.head()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2)
print(X_train , X_test , y_train , y_test)
import xgboost as xgb

from sklearn.metrics import mean_squared_error
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train,y_train)



preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" % (rmse))
xgb.plot_tree(xg_reg,num_trees=0)

plt.rcParams['figure.figsize'] = [12, 12]

plt.show()
xgb.plot_importance(xg_reg)

plt.rcParams['figure.figsize'] = [12, 12]

plt.show()