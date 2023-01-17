import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset = pd.read_csv('../input/50_Startups.csv')

x = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 4].values
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(

    [('one_hot_encoder', OneHotEncoder(), [3])],    #number 3 means which Column

    remainder='passthrough'                         # Leave the rest of the columns untouched

)

x = np.array(ct.fit_transform(x), dtype=np.float)
x=x[:,1:] # Avoiding the Dummy Variable Trap
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_teast=train_test_split(x,y,test_size=0.2,random_state=0)
#traning

from sklearn.linear_model import LinearRegression

regg=LinearRegression()

regg.fit(x_train,y_train)

y_pred=regg.predict(x_test)
import statsmodels.api as sm

x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)

x_opt=x[:,[0,1,2,3,4,5]]

regg_ols=sm.OLS(endog=y,exog=x_opt).fit()

regg_ols.summary()
x_opt=x[:,[0,1,3,4,5]]

regg_ols=sm.OLS(endog=y,exog=x_opt).fit()

regg_ols.summary()
x_opt=x[:,[0,3,4,5]]

regg_ols=sm.OLS(endog=y,exog=x_opt).fit()

regg_ols.summary()
x_opt=x[:,[0,3,5]]

regg_ols=sm.OLS(endog=y,exog=x_opt).fit()

regg_ols.summary()
x_opt=x[:,[0,3]]

regg_ols=sm.OLS(endog=y,exog=x_opt).fit()

regg_ols.summary()