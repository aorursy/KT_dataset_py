# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv("../input/train.csv")

df_test=pd.read_csv("../input/test.csv")

df_train.drop('Id',axis=1,inplace=True)

df_test.drop('Id',axis=1,inplace=True)

df_train.head(6)
import seaborn as sns

sns.heatmap(df_train.isnull())
print("Total df Values",str(len(df_train)))

for i in df_train.columns:

    print(i,"\t\t",str(df_train[i].isnull().sum()))
df_lot=pd.DataFrame()

df_lot["LotArea"]=df_train["LotArea"]

df_lot["LotFrontage"]=df_train["LotFrontage"]

df_lot.head(20)



Missing_lotFrontage_Df=pd.DataFrame()

Missing_lotFrontage_Df=df_lot[ df_lot.LotFrontage.isnull()]

print("Missing Values",str(len(Missing_lotFrontage_Df)))



Not_Missing_lotFrontage_Df=pd.DataFrame()

Not_Missing_lotFrontage_Df=df_lot[ df_lot.LotFrontage.notnull()]

print("Not Missing Values",str(len(Not_Missing_lotFrontage_Df)))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

   Not_Missing_lotFrontage_Df["LotArea"],Not_Missing_lotFrontage_Df["LotFrontage"], test_size=0.3, random_state=0)



X_train=np.asarray(X_train)

X_train=X_train.reshape(-1,1)

X_test=np.asarray(X_test)

X_test=X_test.reshape(-1,1)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,y_train)

y_head_lr = lr.predict(X_test)

from sklearn.metrics import r2_score

print("r_square score: ", r2_score(y_test,y_head_lr))








from sklearn import tree

dtr = tree.DecisionTreeClassifier()

dtr.fit(X_train,y_train)

y_head_dtr = dtr.predict(X_test)

print("r_square score (train dataset): ", r2_score(y_test,y_head_dtr))
import matplotlib.pyplot as plt

plt.plot(y_test, color = 'green', label = 'Actual')

plt.plot(y_head_dtr , color = 'blue', label = 'Predicted')

plt.grid(alpha = 0.3)

plt.xlabel('Number of Candidate')

plt.ylabel('Score')

plt.title('Actual vs Predicted')

plt.legend()

plt.show()
df_train["LotFrontage"]=df_train["LotFrontage"].fillna(df_train["LotFrontage"].mean())

df_train["LotFrontage"].isnull().sum()

#ook missing values are fill with mean
df_train["Alley"]=df_train["Alley"].fillna("No")

df_train["Alley"].isnull().sum()
df_basement=pd.DataFrame()

df_basement["BsmtQual"]=df_train["BsmtQual"]

df_basement["BsmtCond"]=df_train["BsmtCond"]

df_basement["BsmtExposure"]=df_train["BsmtExposure"]

df_basement["BsmtFinType1"]=df_train["BsmtFinType1"]

df_basement["BsmtFinSF1"]=df_train["BsmtFinSF1"]

df_basement["BsmtFinType2"]=df_train["BsmtFinType2"]

df_basement["BsmtFinSF2"]=df_train["BsmtFinSF2"]

df_basement["BsmtUnfSF"]=df_train["BsmtUnfSF"]

df_basement["TotalBsmtSF"]=df_train["TotalBsmtSF"]

df_basement.head(20)



df_basement[ df_basement.BsmtQual.isnull()]
df_train["BsmtQual"]=df_train["BsmtQual"].fillna("No")

df_train["BsmtCond"]=df_train["BsmtCond"].fillna("No")

df_train["BsmtExposure"]=df_train["BsmtExposure"].fillna("No")

df_train["BsmtFinType1"]=df_train["BsmtFinType1"].fillna("No")

df_train["BsmtFinSF1"]=df_train["BsmtFinSF1"].fillna(0)

df_train["BsmtFinType2"]=df_train["BsmtFinType2"].fillna("No")

df_train["BsmtFinSF2"]=df_train["BsmtFinSF2"].fillna(0)

df_train["BsmtUnfSF"]=df_train["BsmtUnfSF"].fillna(0)

df_train["TotalBsmtSF"]=df_train["TotalBsmtSF"].fillna(0)



df_train["BsmtQual"].isnull().sum()

# ok we can now say that all values are filled
df_train["Electrical"]=df_train["Electrical"].fillna("SBrkr")
df_Fireplaces=pd.DataFrame()

df_Fireplaces["Fireplaces"]=df_train["Fireplaces"]

df_Fireplaces["FireplaceQu"]=df_train["FireplaceQu"]

df_Fireplaces[df_Fireplaces["FireplaceQu"].isnull()]
df_train["FireplaceQu"]=df_train["FireplaceQu"].fillna("No")

df_train["FireplaceQu"].isnull().sum()
df_basement=pd.DataFrame()

df_basement["BsmtQual"]=df_train["BsmtQual"]

df_basement["BsmtCond"]=df_train["BsmtCond"]

df_basement["BsmtExposure"]=df_train["BsmtExposure"]

df_basement["BsmtFinType1"]=df_train["BsmtFinType1"]

df_basement["BsmtFinSF1"]=df_train["BsmtFinSF1"]

df_basement["BsmtFinType2"]=df_train["BsmtFinType2"]

df_basement["BsmtFinSF2"]=df_train["BsmtFinSF2"]

df_basement["BsmtUnfSF"]=df_train["BsmtUnfSF"]

df_basement["TotalBsmtSF"]=df_train["TotalBsmtSF"]

df_basement.head(20)



df_basement[ df_basement.BsmtQual.isnull()]
df_Garage=pd.DataFrame()

df_Garage["GarageType"]=df_train["GarageType"]

df_Garage["GarageYrBlt"]=df_train["GarageYrBlt"]

df_Garage["GarageFinish"]=df_train["GarageFinish"]

df_Garage["GarageCars"]=df_train["GarageCars"]

df_Garage["GarageArea"]=df_train["GarageArea"]

df_Garage["GarageQual"]=df_train["GarageQual"]

df_Garage["GarageCond"]=df_train["GarageCond"]





df_Garage[ df_Garage.GarageCond.isnull()]
df_train["GarageType"]=df_train["GarageType"].fillna("No")

df_train["GarageYrBlt"]=df_train["GarageYrBlt"].fillna(0)

df_train["GarageFinish"]=df_train["GarageFinish"].fillna("No")

df_train["GarageCars"]=df_train["GarageCars"].fillna(0)

df_train["GarageArea"]=df_train["GarageArea"].fillna(0)

df_train["GarageQual"]=df_train["GarageQual"].fillna("No")

df_train["GarageCond"]=df_train["GarageCond"].fillna("No")



df_train["GarageCond"].isnull().sum()
df_train["PoolQC"]=df_train["PoolQC"].fillna("No")

df_train["Fence"]=df_train["Fence"].fillna("No")

df_train["MiscFeature"]=df_train["MiscFeature"].fillna("No")

df_train["MasVnrType"]=df_train["MasVnrType"].fillna("No")

df_train["MasVnrArea"]=df_train["MasVnrArea"].fillna(0)
sns.set(rc={'figure.figsize':(5,5)})

sns.heatmap(df_train.isnull())

df_train["MasVnrArea"].isnull().sum()
from sklearn.preprocessing import LabelEncoder

le =LabelEncoder()

String_coloum=[]

for i in df_train.columns:

    if(type(df_train[i].iloc[45])==str ):

        String_coloum.append(i)

print("String Coloum in dataset",len(String_coloum))

print("All Coloums of Dataset",len(df_train.columns))    

df_train['ExterQual'] = le.fit_transform(df_train['ExterQual'].values)

for i in String_coloum:

    df_train[i] = le.fit_transform(df_train[i].values)

df_train.head(5)
sns.set(rc={'figure.figsize':(30,30)})

sns.heatmap(df_train.corr())
df_train.corr()
X=df_train.drop("SalePrice",axis=1)

y=df_train["SalePrice"]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

   X,y, test_size=0.1, random_state=0)

print(X_train.shape)

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)



results=cross_val_score(clf,X_train ,y_train , cv=10)

print(results)

Average=sum(results) / len(results) 

print("Average Accuracy :",Average)





from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

lr = LinearRegression()

lr.fit(X_train, y_train)

rr = Ridge(alpha=0.01) # higher the alpha value, more restriction on the coefficients; low alpha > more generalization, coefficients are barely

# restricted and in this case linear and ridge regression resembles

rr.fit(X_train, y_train)

rr100 = Ridge(alpha=100) #  comparison with alpha value

rr100.fit(X_train, y_train)

train_score=lr.score(X_train, y_train)

test_score=lr.score(X_test, y_test)

Ridge_train_score = rr.score(X_train,y_train)

Ridge_test_score = rr.score(X_test, y_test)

Ridge_train_score100 = rr100.score(X_train,y_train)

Ridge_test_score100 = rr100.score(X_test, y_test)

print ("linear regression train score:", train_score)

print ("linear regression test score:", test_score)

print ("ridge regression train score low alpha:", Ridge_train_score)

print ("ridge regression test score low alpha:", Ridge_test_score)

print ("ridge regression train score high alpha:", Ridge_train_score100)

print ("ridge regression test score high alpha:", Ridge_test_score100)

sns.set(rc={'figure.figsize':(10,10)})

plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 0.01$',zorder=7) # zorder for ordering the markers

plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$') # alpha here is for transparency

plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')

plt.xlabel('Coefficient Index',fontsize=16)

plt.ylabel('Coefficient Magnitude',fontsize=16)

plt.legend(fontsize=13,loc=4)

plt.show()


lr_prediction=lr.predict(X_test)

rr_prediction=rr.predict((X_test))

rr100_prediction=rr100.predict(X_test)

plt.plot(lr_prediction,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'lr_prediction',zorder=7) # zorder for ordering the markers

plt.plot(rr_prediction,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'rr_prediction') # alpha here is for transparency

plt.plot(rr100_prediction,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='rr100_prediction')

plt.plot(y_test,alpha=0.7,linestyle='none',marker="p",markersize=5,color='Yellow',label=r'Y_test')

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.kernel_ridge import KernelRidge

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
import xgboost as xgb

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
from sklearn.model_selection import KFold, cross_val_score, train_test_split

#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)

    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)


score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
from mlxtend.regressor import StackingRegressor

from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf')

stregr = StackingRegressor(regressors=[model_lgb, GBoost,lasso], 

                           meta_regressor=model_xgb)

stregr.fit(X_train, y_train)

stregr_prediction=stregr.predict(X_test)



score = rmsle_cv(stregr)

print("Stacking: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
model_lgb.fit(X_train,y_train)

GBoost.fit(X_train,y_train)

stregr.fit(X_train, y_train)





stregr_prediction=stregr_prediction

lgb_prediction=model_lgb.predict(X_test)

gb=GBoost.predict(X_test)





plt.plot(stregr_prediction,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'lr_prediction',zorder=7) # zorder for ordering the markers

plt.plot(lgb_prediction,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'rr_prediction') # alpha here is for transparency

plt.plot(gb,alpha=0.7,linestyle='none',marker='o',markersize=5,color='orange',label=r'lr_prediction',zorder=7) # zorder for ordering the markers



plt.plot(y_test,alpha=0.7,linestyle='none',marker="p",markersize=5,color='Yellow',label=r'Y_test')

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor

rng = np.random.RandomState(1)

lasso_Bag=AdaBoostRegressor(lasso,

                          n_estimators=10, random_state=rng)

lasso_Bag.fit(X_train,y_train)

score = rmsle_cv(lasso_Bag)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))








lasso_Bag_prediction=lasso_Bag.predict(X_test)







plt.plot(lasso_Bag_prediction,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'rr_prediction') # alpha here is for transparency



plt.plot(y_test,alpha=0.7,linestyle='none',marker="p",markersize=5,color='Yellow',label=r'Y_test')

score = rmsle_cv(lasso_Bag)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.ensemble import  BaggingRegressor



Bagging_clf = BaggingRegressor(base_estimator=ENet )

Bagging_clf.fit(X_train, y_train)









Bagging_clf_pre=Bagging_clf.predict(X_test)







plt.plot(Bagging_clf_pre,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'rr_prediction') # alpha here is for transparency



plt.plot(y_test,alpha=0.7,linestyle='none',marker="p",markersize=5,color='Yellow',label=r'Y_test')

score = rmsle_cv(vote_clf)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

df_test=pd.read_csv("../input/test.csv")

df_id=df_test["Id"]

df_test.drop('Id',axis=1,inplace=True)

df_test.isnull().sum()

#first we remove null values
df_test["LotFrontage"]=df_test["LotFrontage"].fillna(df_test["LotFrontage"].mean())

df_test["BsmtFullBath"]=df_test["BsmtFullBath"].fillna(df_test["BsmtFullBath"].mean())

df_test["BsmtHalfBath"]=df_test["BsmtHalfBath"].fillna(df_test["BsmtHalfBath"].mean())

df_test["Alley"]=df_test["Alley"].fillna("No")

df_test["Alley"].isnull().sum()

df_test["BsmtQual"]=df_test["BsmtQual"].fillna("No")

df_test["BsmtCond"]=df_test["BsmtCond"].fillna("No")

df_test["BsmtExposure"]=df_test["BsmtExposure"].fillna("No")

df_test["BsmtFinType1"]=df_test["BsmtFinType1"].fillna("No")

df_test["BsmtFinSF1"]=df_test["BsmtFinSF1"].fillna(0)

df_test["BsmtFinType2"]=df_test["BsmtFinType2"].fillna("No")

df_test["BsmtFinSF2"]=df_test["BsmtFinSF2"].fillna(0)

df_test["BsmtUnfSF"]=df_test["BsmtUnfSF"].fillna(0)

df_test["TotalBsmtSF"]=df_test["TotalBsmtSF"].fillna(0)

df_test["GarageType"]=df_test["GarageType"].fillna("No")

df_test["GarageYrBlt"]=df_test["GarageYrBlt"].fillna(0)

df_test["GarageFinish"]=df_test["GarageFinish"].fillna("No")

df_test["GarageCars"]=df_test["GarageCars"].fillna(0)

df_test["GarageArea"]=df_test["GarageArea"].fillna(0)

df_test["GarageQual"]=df_test["GarageQual"].fillna("No")

df_test["GarageCond"]=df_test["GarageCond"].fillna("No")

df_test["PoolQC"]=df_test["PoolQC"].fillna("No")

df_test["FireplaceQu"]=df_test["FireplaceQu"].fillna("No")

df_test["Fence"]=df_test["Fence"].fillna("No")

df_test["MiscFeature"]=df_test["MiscFeature"].fillna("No")

df_test["MasVnrType"]=df_test["MasVnrType"].fillna("No")

df_test["MasVnrArea"]=df_test["MasVnrArea"].fillna(0)

df_test["MSZoning"]=df_test["MSZoning"].fillna(df_test["MSZoning"].mode())

df_test["MSZoning"]=df_test["MSZoning"].fillna("RL")

df_test["Utilities"]=df_test["Utilities"].fillna("AllPub")

df_test["Exterior1st"]=df_test["Exterior1st"].fillna("VinylSd")

df_test["Exterior2nd"]=df_test["Exterior2nd"].fillna("VinylSd")

df_test["Functional"]=df_test["Functional"].fillna("Typ")

df_test["SaleType"]=df_test["SaleType"].fillna("WD")

df_test["SaleType"]=df_test["SaleType"].fillna("WD")

df_test["KitchenQual"]=df_test["KitchenQual"].fillna("TA")

sns.heatmap(df_test.isnull())
le =LabelEncoder()

String_coloum=[]

for i in df_test.columns:

    if(type(df_test[i].iloc[45])==str ):

        String_coloum.append(i)

print("String Coloum in dataset",len(String_coloum))

print("All Coloums of Dataset",len(df_test.columns))    

df_test['ExterQual'] = le.fit_transform(df_test['ExterQual'].values)

for i in String_coloum:

    df_test[i] = le.fit_transform(df_test[i].values)

df_test.head(5)
stregr_prediction=stregr.predict(df_test)

sub = pd.DataFrame()

sub['Id'] = df_id

sub['SalePrice'] = stregr_prediction

sub.to_csv('submission.csv',index=False)