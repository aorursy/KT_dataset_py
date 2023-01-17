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
# Load the Data



train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
# About the Data



train.head()
train.dtypes.sample(60)
train.describe().T
train.info()
train.shape
corr = train.corr()

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

print(corr.SalePrice)
features = ["OverallQual" , "GrLivArea","GarageCars" ,"GarageArea","TotalBsmtSF" ,"1stFlrSF" ,"FullBath"]



X = train[features]

y = train["SalePrice"]
X.head()
X.isnull().sum()
y.isnull().sum()
import seaborn as sns
sns.barplot(x="GrLivArea" , y="SalePrice", data=train)
sns.barplot(x="OverallQual", y="SalePrice", data=train)
sns.distplot(train.SalePrice,kde=False)
sns.boxplot(x="OverallQual",y="SalePrice",data=train)

              

             
sns.catplot(x="OverallQual", y = "SalePrice" , kind= "violin" ,data=train)
sns.catplot(x="GrLivArea", y = "SalePrice" , kind= "violin" ,data=train)
sns.scatterplot( x = "OverallQual" , y = "SalePrice" , data = train)
sns.scatterplot( x = "GrLivArea" , y = "SalePrice" , data = train)
sns.lmplot(x = "SalePrice" , y = "GrLivArea" , hue = "OverallQual" , data = train)
# test_train_split



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=0)
# SVR



from sklearn.svm import SVR

svr_reg = SVR(kernel = "rbf")

svr_reg.fit(x_train,y_train)
# svr prediction

predicted_svr = svr_reg.predict(x_test)
# MAE

from sklearn.metrics import mean_absolute_error





mean_absolute_error(y_test , predicted_svr)
# Root mean Square Error (R2)



from sklearn.metrics import r2_score



r2_score(y_test, svr_reg.predict(x_test)) 
# Random Forest



from sklearn.ensemble import RandomForestRegressor



rf_reg = RandomForestRegressor(n_estimators = 10, random_state = 0)



rf_reg.fit(x_train,y_train)

# RF prediction



predicted_rf = rf_reg.predict(x_test)
#MAE

mean_absolute_error(y_test,predicted_rf)
#R2

r2_score(y_test, rf_reg.predict(x_test)) 

#XGBOOST



from xgboost import XGBRegressor



xgb_regressor = XGBRegressor()

xgb_regressor.fit(x_train, y_train)
#prediction

predicted_xgb = xgb_regressor.predict(x_test)
#mae

mean_absolute_error(y_test,predicted_xgb)
#r2

r2_score(y_test,predicted_xgb)

#SUBMISSION



test_X = test[features]

predicted_xgb_test = xgb_regressor.predict(test_X)

output = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_xgb_test})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")