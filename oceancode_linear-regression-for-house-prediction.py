# Import libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



#linear regression

from sklearn import linear_model

import statsmodels.api as sm
df_train =  pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

sample_submission= pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
df_train.head(10) #this is used to display the first 10 rows of the data
df_train.shape
df_test.shape
df_train.info()
df_train.isnull().sum()[df_train.isnull().sum()>0]
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False)

df_train['LotFrontage']=df_train['LotFrontage'].fillna(df_train['LotFrontage'].mean())
df_test['LotFrontage']=df_test['LotFrontage'].fillna(df_test['LotFrontage'].mean())
df_train.Alley.value_counts()
df_train.drop(['Alley'],axis=1,inplace=True)
df_test.drop(['Alley'],axis=1,inplace=True)
df_train.MasVnrType.value_counts()
df_train.MasVnrArea.value_counts()
df_train['MasVnrType']=df_train['MasVnrType'].fillna(df_train['MasVnrType'].mode()[0])

df_train['MasVnrArea']=df_train['MasVnrArea'].fillna(df_train['MasVnrArea'].mode()[0])
df_test['MasVnrType']=df_test['MasVnrType'].fillna(df_train['MasVnrType'].mode()[0])

df_test['MasVnrArea']=df_test['MasVnrArea'].fillna(df_train['MasVnrArea'].mode()[0])
#df_train.BsmtQual.value_counts()

df_train.BsmtCond.value_counts()

#df_train.BsmtExposure.value_counts()

#df_train.BsmtFinType1.value_counts()
df_train['BsmtCond']=df_train['BsmtCond'].fillna(df_train['BsmtCond'].mode()[0])

df_train['BsmtQual']=df_train['BsmtQual'].fillna(df_train['BsmtQual'].mode()[0])

df_train['BsmtExposure']=df_train['BsmtExposure'].fillna(df_train['BsmtExposure'].mode()[0])

df_train['BsmtFinType1']=df_train['BsmtFinType1'].fillna(df_train['BsmtFinType1'].mode()[0])

df_train['BsmtFinType2']=df_train['BsmtFinType2'].fillna(df_train['BsmtFinType2'].mode()[0])
df_test['BsmtCond']=df_test['BsmtCond'].fillna(df_test['BsmtCond'].mode()[0])

df_test['BsmtQual']=df_test['BsmtQual'].fillna(df_test['BsmtQual'].mode()[0])

df_test['BsmtExposure']=df_test['BsmtExposure'].fillna(df_test['BsmtExposure'].mode()[0])

df_test['BsmtFinType1']=df_train['BsmtFinType1'].fillna(df_test['BsmtFinType1'].mode()[0])

df_test['BsmtFinType2']=df_test['BsmtFinType2'].fillna(df_test['BsmtFinType2'].mode()[0])
df_train.Electrical.value_counts()

df_train.FireplaceQu.value_counts()
df_train['Electrical']=df_train['Electrical'].fillna(df_train['Electrical'].mode()[0])
df_train['FireplaceQu']=df_train['FireplaceQu'].fillna(df_train['FireplaceQu'].mode()[0])

df_test['Electrical']=df_test['Electrical'].fillna(df_test['Electrical'].mode()[0])

df_test['FireplaceQu']=df_test['FireplaceQu'].fillna(df_test['FireplaceQu'].mode()[0])

df_train['GarageType']=df_train['GarageType'].fillna(df_train['GarageType'].mode()[0])

df_train.drop(['GarageYrBlt'],axis=1,inplace=True)

df_train['GarageFinish']=df_train['GarageFinish'].fillna(df_train['GarageFinish'].mode()[0])

df_train['GarageQual']=df_train['GarageQual'].fillna(df_train['GarageQual'].mode()[0])

df_train['GarageCond']=df_train['GarageCond'].fillna(df_train['GarageCond'].mode()[0])
df_test['GarageType']=df_test['GarageType'].fillna(df_test['GarageType'].mode()[0])

df_test.drop(['GarageYrBlt'],axis=1,inplace=True)

df_test['GarageFinish']=df_test['GarageFinish'].fillna(df_test['GarageFinish'].mode()[0])

df_test['GarageQual']=df_test['GarageQual'].fillna(df_test['GarageQual'].mode()[0])

df_test['GarageCond']=df_test['GarageCond'].fillna(df_test['GarageCond'].mode()[0])
df_train.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

df_train.drop(['Id'],axis=1,inplace=True)
df_test.drop(['PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

df_test.drop(['Id'],axis=1,inplace=True)
df_train.isnull().sum()[df_train.isnull().sum()>0]
index = df_test.isnull().sum()[df_test.isnull().sum()>0].index
df_test['BsmtFinSF1'].value_counts()

df_test['BsmtFinSF2'].value_counts()

df_test['BsmtUnfSF'].value_counts()

df_test['BsmtFullBath'].value_counts()

df_test['BsmtHalfBath'].value_counts()

df_test['GarageCars'].value_counts()

df_test['GarageArea'].value_counts()
for i in index:

    df_test[i]=df_test[i].fillna(df_test[i].mode()[0])

df_test.isnull().sum()[df_test.isnull().sum()>0].index
#df_train.corr("SalePrice")

df_train.MSZoning.value_counts()
columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',

         'Condition2','BldgType','Condition1','HouseStyle','SaleType',

        'SaleCondition','ExterCond',

         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',

         'CentralAir',

         'Electrical','KitchenQual','Functional',

         'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']
def category_onehot_multcols(multcolumns):

    df_final=final_df

    i=0

    for fields in multcolumns:

        

        df1=pd.get_dummies(final_df[fields],drop_first=True)

        

        final_df.drop([fields],axis=1,inplace=True)

        if i==0:

            df_final=df1.copy()

        else:

            

            df_final=pd.concat([df_final,df1],axis=1)

        i=i+1

       

        

    df_final=pd.concat([final_df,df_final],axis=1)

        

    return df_final
final_df=pd.concat([df_train,df_test],axis=0)

final_df=category_onehot_multcols(columns)

final_df =final_df.loc[:,~final_df.columns.duplicated()]



df_train=final_df.iloc[:1460,:]

df_test=final_df.iloc[1460:,:]
df_test["Shed"]
train_variables = df_train.drop('SalePrice',axis='columns')

price = df_train.SalePrice

train_variables =sm.add_constant(train_variables)

test_variables = df_test.drop('SalePrice',axis='columns')

test_variables=sm.add_constant(test_variables)
est = sm.OLS(price, train_variables)

model= est.fit()

model.summary()

index = model.pvalues[model.pvalues > 0.05].index

train_variables.drop(index, axis = 1, inplace = True)

test_variables.drop(index, axis = 1, inplace = True)

while len(index) > 0 :

    

    est = sm.OLS(price, train_variables)

    model= est.fit()

    index = model.pvalues[model.pvalues > 0.05].index

    train_variables.drop(index, axis = 1, inplace = True)

    test_variables.drop(index, axis = 1, inplace = True)
est = sm.OLS(price, train_variables)

model= est.fit()

model.summary()
SalePrice_prediction = model.predict(test_variables)
submission_df = pd.DataFrame({'Id': sample_submission.Id, 'SalePrice': SalePrice_prediction.values})
submission_df.to_csv('submission.csv',index = False)
train_variables = df_train.drop('SalePrice',axis='columns')

price = df_train.SalePrice

train_variables =sm.add_constant(train_variables)

test_variables = df_test.drop('SalePrice',axis='columns')

test_variables=sm.add_constant(test_variables)
column_list = []

all_columns = train_variables.columns

best_aic = 100000

local_best_aic = 99999

while local_best_aic != best_aic:

    local_best_aic = best_aic



    for i in all_columns:

        index = column_list + [i]

        train_variables

        est = sm.OLS(price, train_variables[index])

        model= est.fit()

        AIC = model.aic

        if AIC < best_aic:

            best_index = index

            best_aic = AIC

            best_i = i

    if local_best_aic == best_aic:

        break

    

    column_list = best_index

    all_columns.drop(best_i)

    print(best_i)



len(column_list)



est = sm.OLS(price, train_variables[column_list])

model= est.fit()

SalePrice_prediction = model.predict(test_variables[column_list])

submission_df = pd.DataFrame({'Id': sample_submission.Id, 'SalePrice': SalePrice_prediction.values})

submission_df.to_csv('submission.csv',index = False)
from sklearn.linear_model import Ridge, ElasticNet, Lasso

from sklearn.model_selection import cross_val_score

train_variables = df_train.drop('SalePrice',axis='columns')

price = df_train.SalePrice

test_variables = df_test.drop('SalePrice',axis='columns')

def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, train_variables, price, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]

cv_ridge  #best alpha = 10
alphas = [ 75, 100, 125, 150, 200, 250]

cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() 

            for alpha in alphas]

cv_lasso  #best alpha = 125
alphas = [0.0001, 0.001,0.005, 0.01, 0.1]

cv_elastic = [rmse_cv(ElasticNet(alpha = alpha)).mean() 

            for alpha in alphas]

cv_elastic  #best alpha = 1
lassoreg = Lasso(alpha = 125)

lassoreg.fit(train_variables,price)

SalePrice_prediction = lassoreg.predict(test_variables)



submission_df = pd.DataFrame({'Id': sample_submission.Id, 'SalePrice': SalePrice_prediction})

submission_df.to_csv('submission.csv',index = False)