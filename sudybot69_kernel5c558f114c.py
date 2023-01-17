# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('../input/vehicle-dataset-from-cardekho/car data.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')

df.drop(df.columns[[0,3]],axis = 1,inplace = True)

df.Fuel_Type.replace(regex={"Petrol":"0","Diesel":"1","CNG":"2"},inplace = True)

df.Seller_Type.replace(regex={"Dealer":"0","Individual":"1"},inplace = True)

df.Transmission.replace(regex={"Manual":"0","Automatic":"1"},inplace = True)

df.head(10)
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression

X = df.iloc[:,[0,2,3,4,5,6]].values

y = df.iloc[:,1].values



#Backward Elimination

import statsmodels.api as sm

X = np.append(arr = np.ones((301,1)).astype(int),values = X,axis = 1)

X_opt = X[:,[0,1,3,4,5]]

X_opt = np.array(X_opt,dtype = float)

ols = sm.OLS(endog = y,exog = X_opt).fit()



X_train,X_test,y_train,y_test = train_test_split(X_opt,y,test_size = 0.3,random_state = 42)

regressor = LinearRegression()

regressor.fit(X_train,y_train)



y_pred = regressor.predict(X_test)

print('R2 SCORE -> ',r2_score(y_pred,y_test))