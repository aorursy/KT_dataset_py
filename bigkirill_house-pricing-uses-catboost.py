import pandas as pd
from numpy import *
import numpy as np
from catboost import Pool, CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')
train_data = train_data.drop(train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<300000)].index)
eq_val=['TA','Gd','Ex','Fa','Po']
eq_s=[0,1,2,3,4]
eq_typ=dict(zip(eq_val,eq_s))
y_val=['Y','N']
y_s=[0,1]
y_typ=dict(zip(y_val,y_s))
style_val=['1Story','2Story','1.5Fin','SLvl','SFoyer','1.5Unf','2.5Unf','2.5Fin']
style_s=[0,1,2,3,4,5,6,7]
style_typ=dict(zip(style_val,style_s))
train_data['BsmtQual']=train_data['BsmtQual'].fillna('Fa')
train_data=train_data.replace({'ExterQual':eq_typ})
train_data=train_data.replace({'BsmtQual':eq_typ})
train_data=train_data.replace({'HeatingQC':eq_typ})
train_data=train_data.replace({'KitchenQual':eq_typ})
train_data=train_data.replace({'CentralAir':y_typ})
train_data=train_data.replace({'HouseStyle':style_typ})
train_data['GarageYrBlt']=train_data['GarageYrBlt'].fillna(0)
train_data['LotFrontage']=train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean())
train_data
#quantiles
#GrLivArea
sp_q_01=train_data['GrLivArea'].quantile(0.01)
sp_q_99=train_data['GrLivArea'].quantile(0.99)
train_data = train_data.drop(train_data[(train_data['GrLivArea']>sp_q_99)].index)
train_data = train_data.drop(train_data[(train_data['GrLivArea']<sp_q_01)].index)

#LotFrontage
b_01=train_data['LotFrontage'].quantile(0.01)
b_99=train_data['LotFrontage'].quantile(0.99)
train_data = train_data.drop(train_data[(train_data['LotFrontage']>b_99)].index)
train_data = train_data.drop(train_data[(train_data['LotFrontage']<b_01)].index)

#TotRmsAbvGrd
c_01=train_data['TotRmsAbvGrd'].quantile(0.01)
c_99=train_data['TotRmsAbvGrd'].quantile(0.99)
train_data = train_data.drop(train_data[(train_data['TotRmsAbvGrd']>c_99)].index)
train_data = train_data.drop(train_data[(train_data['TotRmsAbvGrd']<c_01)].index)

#TotalBsmtSF,LotArea,YearBuilt,1stFlrSF
mean_vals=['LotArea','OverallQual','YearBuilt','KitchenQual','YearRemodAdd','TotalBsmtSF','1stFlrSF','GrLivArea','GarageCars','GarageArea','ExterQual','BsmtQual','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd','Fireplaces','LotFrontage','GarageYrBlt','BsmtFinSF1','OverallCond','HouseStyle','MSSubClass','KitchenAbvGr','WoodDeckSF']
train=train_data.filter(mean_vals,axis=1)
train.head()
test_data['BsmtQual']=test_data['BsmtQual'].fillna('Fa')
test_data['GarageCars']=test_data['GarageCars'].fillna(0)
test_data['GarageArea']=test_data['GarageArea'].fillna(0)
test_data['BsmtFinSF1']=test_data['BsmtFinSF1'].fillna(0)
test_data['GarageYrBlt']=test_data['GarageYrBlt'].fillna(0)
test_data['BsmtUnfSF']=test_data['BsmtUnfSF'].fillna(0)
test_data['TotalBsmtSF']=test_data['BsmtFinSF1']+test_data['BsmtUnfSF']
test_data['TotalBsmtSF']=test_data['TotalBsmtSF'].fillna(0)
test_data['KitchenQual']=test_data['KitchenQual'].fillna('Po')
test_data=test_data.replace({'ExterQual':eq_typ})
test_data=test_data.replace({'BsmtQual':eq_typ})
test_data=test_data.replace({'HeatingQC':eq_typ})
test_data=test_data.replace({'KitchenQual':eq_typ})
test_data=test_data.replace({'CentralAir':y_typ})
test_data=test_data.replace({'HouseStyle':style_typ})
test_data['LotFrontage']=test_data['LotFrontage'].interpolate()
scaler = MinMaxScaler()
scaler.fit(train)
train=scaler.transform(train)
test=test_data.filter(mean_vals,axis=1)
test=scaler.transform(test)
train_sub=train_data['SalePrice']
train_X, test_X, train_y, test_y = train_test_split(train, train_sub, test_size=0.2, random_state=42)
train_pool = Pool(train_X, train_y)
test_pool = Pool(test_X, test_y.values) 
model2 = CatBoostRegressor(
    iterations=30000,
    depth=10,
    learning_rate=0.001,
    l2_leaf_reg= 0.1,#def=3
    loss_function='RMSE',
    eval_metric='MAE',
    random_strength=0.001,
    bootstrap_type='Bayesian',#Poisson (supported for GPU only);Bayesian;Bernoulli;No
    bagging_temperature=1,#for Bayesian bootstrap_type; 1=exp;0=1
    leaf_estimation_method='Newton', #Gradient;Newton
    leaf_estimation_iterations=2,
    boosting_type='Ordered' #Ordered-small data sets; Plain
    ,task_type = "GPU"
    ,feature_border_type='Median' #Median;Uniform;UniformAndQuantiles;MaxLogSum;MinEntropy;GreedyLogSum
    ,random_seed=1234
)
model2.fit(train_pool, eval_set=test_pool, plot=True)
pred_test=model2.predict(test)
res=test_data.filter(['Id'],axis=1)
res['SalePrice']=pred_test
res.to_csv('submission.csv', index=False)