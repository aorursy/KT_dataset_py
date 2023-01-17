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
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

sample_submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
train.head()
test.head()
train_and_test = pd.concat([train.iloc[:,:-1],test],axis=0).drop(columns=['Id'],axis=1)
for column in train_and_test.columns:    

    if train_and_test[column].dtype  == 'object':

        train_and_test[column].fillna(value = 'UNKNOWN', inplace=True)

    else:

        train_and_test[column].fillna(value = train_and_test[column].median(), inplace=True)   
train_and_test = pd.get_dummies(train_and_test)
train_data = train_and_test.iloc[:1460,:]

test_data = train_and_test.iloc[1460:,:]
from sklearn.model_selection import train_test_split
X = train_data

y = np.log(train.SalePrice)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred_lr = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score



print('mean_squared_error: ',mean_squared_error(y_test, y_pred_lr),

     '\nr2_score: ',r2_score(y_test, y_pred_lr)

     )
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor()

rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
print('mean_squared_error: ',mean_squared_error(y_test, y_pred_rf),

     '\nr2_score: ',r2_score(y_test, y_pred_rf)

     )


from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor()

GBR.fit(X_train, y_train)
y_pred_gbr = GBR.predict(X_test)
print('mean_squared_error: ',mean_squared_error(y_test, y_pred_gbr),

     '\nr2_score: ',r2_score(y_test, y_pred_gbr)

     )
print('Linear regression :', r2_score(y_test, y_pred_lr),

      '\nRandom Forest regression :', r2_score(y_test, y_pred_rf),

      '\nGradient Boosting regression :', r2_score(y_test, y_pred_gbr),

)
result = np.exp(regressor.predict(test_data))

submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': result})

submission.to_csv('result.csv', index=False)