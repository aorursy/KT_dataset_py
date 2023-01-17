import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.svm import SVR

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
#Importing the dataset

dataset_train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

dataset_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print("Training Set")

dataset_train.head(n=10)
print("Test Set")

dataset_test.head(n=10)
dataset=pd.concat([dataset_train,dataset_test],axis=0)

X=dataset.drop(labels=['SalePrice','Id'],axis=1)

y=dataset[['SalePrice']]

print("Independent Variables")

X.head(n=10)

print("Shape of X is",X.shape)
print("Dependent Variable")

y.head(n=10)
missing_count=2919-X.count()

print(missing_count)
pd.options.mode.chained_assignment=None

for i in ["LotFrontage","MasVnrArea"]:

    X[i]=X[i].fillna(X[i].mean())

for i in ["MSZoning","Utilities","Exterior1st","Exterior2nd","MasVnrType","Electrical","KitchenQual","Functional","SaleType"]:

    X[i]=X[i].fillna(X[i].mode()[0])

for i in ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtFinType2","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","BsmtFullBath","BsmtHalfBath","FireplaceQu","GarageType","GarageYrBlt","GarageFinish","GarageCars","GarageArea","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]:

    X[i]=X[i].fillna(0)
X_encoded=pd.DataFrame()

for i in ["MSSubClass","MSZoning","Street","Alley","LotShape","LandContour","LotConfig","LandSlope","Utilities","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","OverallQual","OverallCond","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","ExterQual","ExterCond","Foundation","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir","Electrical","KitchenQual","Functional","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive","PoolQC","Fence","MiscFeature","SaleType","SaleCondition"]:

    X_encoded=pd.concat([X_encoded,pd.get_dummies(X[i],prefix=i,drop_first=True)],axis=1)

for i in ["LotFrontage","LotArea","YearBuilt","YearRemodAdd","MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces","GarageYrBlt","GarageCars","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal","MoSold","YrSold"]:

    X_encoded=pd.concat([X_encoded,X[[i]]],axis=1)

X=X_encoded

print("Onehot encoded independent variables")

X.head(n=10)
X=X.to_numpy()

y=y.to_numpy()

X[:,276]=X_encoded['YearBuilt'].to_numpy()
X_train=X[0:1460,:]

X_test=X[1460:,:]

y_train=y[0:1460]
sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)

X_test=sc_X.transform(X_test)
def scorer(estimator,X,y):

    y_pred=estimator.predict(X)

    y_true=y

    score=r2_score(y_true,y_pred)

    return score
regressor1=LinearRegression()

regressor1.fit(X_train,y_train)



scores1=cross_val_score(regressor1,X_train,y_train.reshape(-1,),cv=10,scoring=scorer)

print("Linear Regression 10-fold R^2 is",scores1.mean())
regressor2=SVR(kernel='rbf') 

regressor2.fit(X_train,y_train.reshape(-1,))



scores2=cross_val_score(regressor2,X_train,y_train.reshape(-1,),cv=10,scoring=scorer)

print("Support Vector Regression 10-fold R^2 is",scores2.mean())
regressor3=DecisionTreeRegressor(random_state=0)

regressor3.fit(X_train,y_train)



scores3=cross_val_score(regressor3,X_train,y_train.reshape(-1,),cv=10,scoring=scorer)

print("Decision Tree Regression 10-fold R^2 is",scores3.mean())
regressor4=RandomForestRegressor(n_estimators=100,random_state=0)

regressor4.fit(X_train,y_train.reshape(-1,))



scores4=cross_val_score(regressor4,X_train,y_train.reshape(-1,),cv=10,scoring=scorer)

print("Random Forest Regression 10-fold R^2 is",scores4.mean())
regressor5=XGBRegressor(n_estimators=1000,learning_rate=0.02,random_state=0)

regressor5.fit(X_train,y_train.reshape(-1,))



scores5=cross_val_score(regressor5,X_train,y_train.reshape(-1,),cv=10,scoring=scorer)

print("XGBoost Regression 10-fold R^2 is",scores5.mean())
params={

        'gamma':[0.5,1,1.5],

        'reg_lambda':[0,0.01,0.05],

        'reg_alpha':[0,0.01,0.05],

        'subsample': [0.6,0.8,1.0],

        'colsample_bytree':[0.6,0.8,1],

        'max_depth': [3, 4, 5]

        }

grid=GridSearchCV(regressor5,param_grid=params,n_jobs=4,cv=5,verbose=3,scoring='r2')

grid.fit(X_train,y_train.reshape(-1,))

best_r2=grid.best_score_

best_parameters=grid.best_params_

print("Best Accuracy is",best_r2)

print("Best Parameters are",best_parameters)
regressor=XGBRegressor(n_estimators=1000,learning_rate=0.02,

                       gamma=0.5,

                       reg_lambda=0,

                       reg_alpha=0.01,

                       subsample=0.8,

                       colsample_bytree=0.6,

                       max_depth=4

                      )

regressor.fit(X_train,y_train.reshape(-1,))
y_test_pred=regressor.predict(X_test).reshape(-1,1)

dataset2=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

dataset2=pd.to_numeric(dataset2['Id'])

pred_output=pd.DataFrame(y_test_pred,index=None,columns=["SalePrice"])

output=pd.concat([dataset2,pred_output],axis=1)

output.to_csv('output.csv',index=False)

print("Test set predictions:")

output.head(n=10)