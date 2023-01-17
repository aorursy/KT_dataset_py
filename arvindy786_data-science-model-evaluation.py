import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error

from scipy import stats

%matplotlib inline 

pd.options.display.max_columns = 400

import statsmodels.api as sm

filname='../input/hsng.csv'

df = pd.read_csv(filname)

df.head(2)
model1_df=df[['OverallQual','BsmtQual','FullBath','TotRmsAbvGrd']]

model2_df=df[['OverallQual','GarageCars','FullBath','TotRmsAbvGrd','MSZoning','Neighborhood','Condition1','Condition2',

              'HouseStyle','ExterQual','BsmtQual','BsmtCond','KitchenQual','GarageType','GarageFinish','PoolQC',

              'GrLivArea','GarageArea','TotalBsmtSF']]

model3_df=df[['2ndFlrSF','OpenPorchSF','LotArea','BsmtUnfSF']]
model1_fit = sm.OLS(df['SalePrice'],model1_df,missing='drop').fit()

model1_prediction=model1_fit.predict(model1_df) 



model2_fit = sm.OLS(df['SalePrice'],model2_df,missing='drop').fit()

model2_prediction=model2_fit.predict(model2_df) 



model3_fit = sm.OLS(df['SalePrice'],model3_df,missing='drop').fit()

model3_prediction=model3_fit.predict(model3_df) 



plt.figure(figsize=(13, 10))



ax1 = sns.distplot(df['SalePrice'], hist=False, color="r", label="Actual Value")

sns.distplot(model1_prediction, hist=False, color="b", label="Predicted Values Model 1" , ax=ax1)

sns.distplot(model2_prediction, hist=False, color="g", label="Predicted Values Model 2" , ax=ax1)

sns.distplot(model3_prediction, hist=False, color="k", label="Predicted Values Model 3" , ax=ax1)



plt.title('Actual vs Predicted Values for Price')

plt.xlabel('SalePrice')

plt.ylabel('parameter of House')

plt.show()

plt.close()
print("Model_1_AIC: ",model1_fit.aic)

print("Model_2_AIC: ",model2_fit.aic)

print("Model_3_AIC: ",model3_fit.aic)
print("Model_1_BIC: ",model1_fit.aic)

print("Model_2_BIC: ",model2_fit.bic)

print("Model_3_BIC: ",model3_fit.bic)
print("R_Sqaured_Model_1: ",model1_fit.rsquared)

print("R_Sqaured_Model_2: ",model2_fit.rsquared)

print("R_Sqaured_Model_3: ",model3_fit.rsquared)
print("Adj_R_Sqaured_Model_1_BIC: ",model1_fit.rsquared_adj)

print("Adj_R_Sqaured_Model_2_BIC: ",model2_fit.rsquared_adj)

print("Adj_R_Sqaured_Model_3_BIC: ",model3_fit.rsquared_adj)
print("MSE_Model_1: ",mean_squared_error(df['SalePrice'],model1_prediction))

print("MSE_Model_2: ",mean_squared_error(df['SalePrice'],model2_prediction))

print("MSE_Model_3: ",mean_squared_error(df['SalePrice'],model3_prediction))