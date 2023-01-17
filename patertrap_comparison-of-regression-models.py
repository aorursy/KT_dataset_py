#import some necessary librairies



import numpy as np 

import pandas as pd 

%matplotlib inline

import matplotlib.pyplot as plt 

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)





from scipy import stats

from scipy.stats import norm, skew #for some statistics

import missingno as missing



pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x)) #Limiting floats output to 3 decimal points



print("All imported OK")
#Now let's import and put the train and test datasets in  pandas dataframe



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

##display the first five rows of the train dataset.

train.head(5)
##display the first five rows of the test dataset.

test.head(5)

#check the numbers of samples and features

print("The train data size before dropping Id feature is : {} ".format(train.shape))

print("The test data size before dropping Id feature is : {} ".format(test.shape))



#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)



#check again the data size after dropping the 'Id' variable

print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 

print("The test data size after dropping Id feature is : {} ".format(test.shape))
fig, ax= plt.subplots()

ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])

plt.xlabel('Living area (sq. feet)', fontsize=15)

plt.ylabel('Sale Price ($)', fontsize=15)

#The dropping is tricky. Always train.drop(train.isnull()), so train.drop(train.something)



train=train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

fig, ax= plt.subplots()

ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])

plt.xlabel('Living area (sq. feet)', fontsize=15)

plt.ylabel('Sale Price ($)', fontsize=15)
#save the price

y=train["SalePrice"]

#Double sum to avoid one value for each column

test.isnull().sum().sum()

#ok, there are a lot. So imputing will be necessary.

test.isnull().sum()
train_price=train.SalePrice.values

all_data=pd.concat([train,test]).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

all_data.shape
#Use missingno to check the values



missing.bar(all_data);#Black is present data, missing values are removed. Like Alley

missing.heatmap(all_data);
#Now let's see where the missing values are



all_data_nan=(all_data.isnull().sum())/(len(all_data))

all_data_nan=all_data_nan.drop(all_data_nan[all_data_nan==0].index).sort_values(ascending=False)

#in the [] it's all_data_nan==0, not all_data_nan.isnull()==0, because I'm dropping the ones without

#nulls, which I already counted.



miss_data=pd.DataFrame({'Missing proportion':all_data_nan})

miss_data.head()

#Now I make a bar plot, missingno it's not great for this. I'll make my own also, I don't have much experience with seaborn



plt.subplots(figsize=(13,13))

plt.xticks(rotation=90)

ax = sns.barplot(x=miss_data.index, y=miss_data['Missing proportion'])

plt.xlabel('Features', size=15)

plt.ylabel('Missing proportion', size=15)

all_data["PoolQC"]=all_data["PoolQC"].fillna("None")
all_data["MiscFeature"]=all_data["MiscFeature"].fillna("None")
all_data["Alley"]=all_data["Alley"].fillna("None")
all_data["Fence"]=all_data["Fence"].fillna("None")
all_data["FireplaceQu"]=all_data["FireplaceQu"].fillna("None")
#This can be done with a lambda function. I don't like lambda function because they don't help to read the code but I don't see an easy way of doing it without using a lambda.

#They hurt readability, but they have a reason to exist. I don't like them but I use them if I need to.



#First, group by neighborhood. Then, take the "LotFrontage" column and fill NaN with the median of each neighborhood.



all_data["LotFrontage"]=all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ("GarageQual", "GarageType", "GarageCond", "GarageFinish"):

    all_data[col]=all_data[col].fillna("None")
#0, not "0" because I want a number, not a string!!

#I can also copy paste

for col in ("GarageYrBlt", "GarageArea", "GarageCars"):

    all_data[col]=all_data[col].fillna(0)
for col in ("BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF","TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"):

    all_data[col]=all_data[col].fillna(0)
for col in ("BsmtQual", "BsmtCond", "BsmtExposure","BsmtFinType1", "BsmtFinType2"):

    all_data[col]=all_data[col].fillna("None")
mszo=all_data["MSZoning"].value_counts()

mszo_ratio=mszo/(mszo.sum())

print("The number of values are: ",mszo)

print("The ratio of values are: ",mszo_ratio)

all_data["MSZoning"]=all_data["MSZoning"].fillna("RL")
uti=all_data["Utilities"].value_counts()

uti_ratio=uti/(uti.sum())

print("The number of values are: ",uti)

print("The ratio of values are: ",uti_ratio)
all_data["Utilities"]=all_data["Utilities"].fillna("AllPub")
all_data["Functional"]=all_data["Functional"].fillna("Typ")
ele=all_data["Electrical"].value_counts()

ele_ratio=ele/(ele.sum())

print("The number of values are: ",ele)

print("The ratio of values are: ",ele_ratio)
all_data["Electrical"]=all_data["Electrical"].fillna("SBrkr")
kit=all_data["KitchenQual"].value_counts()

kit_ratio=kit/(kit.sum())

print("The number of values are: ",kit)

print("The ratio of values are: ",kit_ratio)
all_data["KitchenQual"]=all_data["KitchenQual"].fillna("TA")
e1=all_data["Exterior1st"].value_counts()

e1_ratio=e1/(e1.sum())

print("The number of values are: ",e1)

print("The ratio of values are: ",e1_ratio)
e2=all_data["Exterior2nd"].value_counts()

e2_ratio=e2/(e2.sum())

print("The number of values are: ",e2)

print("The ratio of values are: ",e2_ratio)
all_data["Exterior1st"]=all_data["Exterior1st"].fillna("VinylSd")

all_data["Exterior2nd"]=all_data["Exterior2nd"].fillna("VinylSd")
e2=all_data["SaleType"].value_counts()

e2_ratio=e2/(e2.sum())

print("The number of values are: ",e2)

print("The ratio of values are: ",e2_ratio)
all_data["SaleType"]=all_data["SaleType"].fillna("WD")
all_data["MasVnrType"]=all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"]=all_data["MasVnrArea"].fillna(0)
e2=all_data["MSSubClass"].value_counts()

e2_ratio=e2/(e2.sum())

print("The number of values are: ",e2)

print("The ratio of values are: ",e2_ratio)
all_data["MSSubClass"]=all_data["MSSubClass"].fillna("None")
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
#missing.bar(all_data);

check=all_data.isnull().sum().sum()

print("Total number of NaN values is: ", check)

print(all_data.shape)
#I do this with the TRAIN set, not the test, which obviously doesn't have a price column



sns.distplot(train["SalePrice"], fit=norm);

#Get mu and sigma used by the distribution



[mu,sigma]=norm.fit(train["SalePrice"])

print("mu = {:.2f} and sigma = {:.2f}".format(mu, sigma))



#Now make the plot a bit more decent



plt.legend(["Normal dist $\mu$={:.2f}, $\sigma$={:.2f}".format(mu,sigma)])



plt.ylabel("Frequency")

plt.xlabel("Sale price ($)")

plt.title("Sale price distribution")

fig=plt.figure()

qq=stats.probplot(train["SalePrice"], plot=plt);

plt.show()

#y=np.log1p(y)

train["SalePrice"]=np.log1p(train["SalePrice"])

y=train["SalePrice"]

#y
#And now let's check again that the transformation is useful

sns.distplot(train["SalePrice"], fit=norm);

#Get mu and sigma used by the distribution



[mu,sigma]=norm.fit(train["SalePrice"])

print("mu = {:.2f} and sigma = {:.2f}".format(mu, sigma))



#Now make the plot a bit more decent



plt.legend(["Normal dist $\mu$={:.2f}, $\sigma$={:.2f}".format(mu,sigma)])



plt.ylabel("Frequency")

plt.xlabel("Sale price ($)")

plt.title("Sale price distribution")



#And QQ' plot to save space

fig=plt.figure()

qq=stats.probplot(train["SalePrice"], plot=plt);

plt.show()
#first, calculate skewness



numeric=all_data.dtypes[all_data.dtypes!="object"].index

skewed_f=all_data[numeric].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewed=pd.DataFrame({"Skewness" : skewed_f})

skewed.head()
#box cox transform, without knowing lambda. I know the skewness of a normal distribution is 3, so I will set a limit on 1



skewed = skewed[abs(skewed)>0.5]

skewed.dropna(inplace=True)

print("The number of skewed features is: ",skewed.shape[0])
from scipy.special import boxcox1p, inv_boxcox1p

feat=skewed.index



lmbda=0.25 #Arbitrary value. boxcox from scipy.stats has an optimizer, to find lambda. 0.25 is relatively high, but should work



for col in feat:

    all_data[col]=boxcox1p(all_data[col], lmbda)

#for col in feat:

#    all_data[col]=inv_boxcox1p(all_data[col],lmbda)
#check skewness



numeric2=all_data.dtypes[all_data.dtypes!="object"].index

skewed_2=all_data[numeric2].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewed2=pd.DataFrame({"Skewness" : skewed_2})

skewed2.head()
print("DF shape before encoding is: ",all_data.shape)



all_data = pd.get_dummies(all_data)

print(all_data.shape)



print("DF shape after encoding is: ",all_data.shape)
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import ElasticNet, Lasso, Ridge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split, cross_val_score, KFold

from sklearn.metrics import mean_absolute_error as MAE

from sklearn.metrics import r2_score



from xgboost.sklearn import XGBRegressor

import lightgbm as LGB


train = all_data[:len(train.index)]

test = all_data[len(train.index):]

#train.head()

#trial

#y=log1pp of train["SalePrice"], saved from before

X=train



train_X,val_X,train_y,val_y=train_test_split(X,y,test_size=0.2,shuffle=True,random_state=13)
#This algorithm is sensitive to outliers. That's why I imported RobustScaler()

Lasso_reg=Lasso()

param_alpha={"alpha":[1e-6,1e-4, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 20]} #I think this values are enough

#La_train_X=RobustScaler().fit(train_X)

CV_Lasso=GridSearchCV(estimator=Lasso_reg,param_grid=param_alpha,scoring="neg_mean_absolute_error",cv=5,n_jobs=-1, refit=True)

#I can directly fit and predict with the best param!!



CV_Lasso_fit=CV_Lasso.fit(train_X,train_y)

pred_Lasso=CV_Lasso.predict(val_X)

pred_Lasso=np.expm1(pred_Lasso)

val_y_u=np.expm1(val_y)



alpha_good=CV_Lasso.best_params_

print("The best parameter alpha for the Lasso regressor is:",CV_Lasso.best_params_)

print("The best score (NegMAE) for the Lasso regressor is:",-CV_Lasso.best_score_)



MAE_Lasso=MAE(val_y_u,pred_Lasso)

r2_Lasso=r2_score(val_y_u,pred_Lasso)



print("The MAE of the prediction is:", MAE_Lasso)

print("The R^2 of the prediction is:", r2_Lasso)

#This algorithm is sensitive to outliers. That's why I imported RobustScaler()

Ridge_reg=Ridge()

param_alpha={"alpha":[1e-6,1e-4, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 20]} #I think this values are enough

#La_train_X=RobustScaler().fit(train_X)

CV_Ridge=GridSearchCV(estimator=Ridge_reg,param_grid=param_alpha,scoring="neg_mean_absolute_error",cv=5,n_jobs=-1)

CV_Ridge=CV_Ridge.fit(train_X,train_y)

#I can directly predict with the best param!!

pred_Ridge=CV_Ridge.predict(val_X)

pred_Ridge=np.expm1(pred_Ridge)



print("The best parameter alpha for the Ridge regressor is:",CV_Ridge.best_params_)

print("The best score (NegMAE) for the Ridge regressor is:",-CV_Ridge.best_score_)



MAE_Ridge=MAE(val_y_u,pred_Ridge)

r2_Ridge=r2_score(val_y_u,pred_Ridge)



print("The MAE of the prediction is:", MAE_Ridge)

print("The R^2 of the prediction is:", r2_Ridge)
#This will take longer, it iterates over a much bigger grid

ENet_reg=ElasticNet()

param_alpha={"alpha":[1e-6,1e-4, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 20],

            "l1_ratio":[1e-6,1e-4, 0.01, 0.1, 0.25, 0.5, 0.75, 1]} #I think this values are enough

#La_train_X=RobustScaler().fit(train_X)

CV_ENet=GridSearchCV(estimator=ENet_reg,param_grid=param_alpha,scoring="neg_mean_absolute_error",cv=5,n_jobs=-1)

CV_ENet=CV_ENet.fit(train_X,train_y)

#I can directly predict with the best param!!

pred_ENet=CV_ENet.predict(val_X)

pred_ENet=np.expm1(pred_ENet)



print("The best parameter alpha for the Elastic Net regressor is:",CV_ENet.best_params_)

print("The best score (NegMAE) for the Elastic Net regressor is:",-CV_ENet.best_score_)



MAE_ENet=MAE(val_y_u,pred_ENet)

r2_ENet=r2_score(val_y_u,pred_ENet)



print("The MAE of the prediction is:", MAE_ENet)

print("The R^2 of the prediction is:", r2_ENet)
Lasso_AIC=LassoLarsIC(criterion="aic")

Lasso_AIC.fit(train_X,train_y)

alpha_aic_ = Lasso_AIC.alpha_

pred_AIC = Lasso_AIC.predict(val_X)

pred_AIC = np.expm1(pred_AIC)



MAE_AIC=MAE(val_y_u,pred_AIC)

r2_AIC=r2_score(val_y_u,pred_AIC)



print("The alpha value using AIC is:",alpha_aic_)

print("The MAE of the prediction using AIC is:", MAE_AIC)

print("The R^2 of the prediction using AIC is:", r2_AIC)



Lasso_BIC=LassoLarsIC(criterion="bic")

Lasso_BIC.fit(train_X,train_y)

alpha_bic_ = Lasso_BIC.alpha_

pred_BIC = Lasso_BIC.predict(val_X)

pred_BIC = np.expm1(pred_BIC)



MAE_BIC=MAE(val_y_u,pred_BIC)

r2_BIC=r2_score(val_y_u,pred_BIC)



print("The alpha value using AIC is:",alpha_bic_)

print("The MAE of the prediction using BIC is:", MAE_BIC)

print("The R^2 of the prediction using BIC is:", r2_BIC)



KRidge_reg=KernelRidge()

param_alpha={"kernel":["linear","poly","rbf"],

            "alpha":[1e-4, 1e-2, 1, 2, 5, 10, 20],

            "gamma":[1e-4, 1e-2, 1, 2, 5, 10, 20]} #I think this values are enough

#La_train_X=RobustScaler().fit(train_X)

CV_KerRidge=GridSearchCV(estimator=KRidge_reg,param_grid=param_alpha,scoring="neg_mean_absolute_error",cv=5,n_jobs=-1)

CV_KerRidge=CV_KerRidge.fit(train_X,train_y)

#I can directly predict with the best param!!

pred_KerRidge=CV_KerRidge.predict(val_X)

pred_KerRidge=np.expm1(pred_KerRidge)



print("The best parameter alpha for the Kernel Ridge regressor is:",CV_KerRidge.best_params_)

print("The best score (NegMAE) for the Kernel Ridge regressor is:",-CV_KerRidge.best_score_)



MAE_KerRidge=MAE(val_y_u,pred_KerRidge)

r2_KerRidge=r2_score(val_y_u,pred_KerRidge)



print("The MAE of the prediction is:", MAE_KerRidge)

print("The R^2 of the prediction is:", r2_KerRidge)
%%time

#RF will fit by a brute force approach, generating many trees. Should be faster because trees are fast.

RF_reg=RandomForestRegressor()

param_alpha={"n_estimators":[100,200,500],

            "max_depth":[3,5,10,None], #More than 3 may overfit unnecessarily. Small gamma is also tricky

            "max_features":[10, "sqrt", "log2", None]} #I think this values are enough

#La_train_X=RobustScaler().fit(train_X)

CV_RF=GridSearchCV(estimator=RF_reg,param_grid=param_alpha,scoring="neg_mean_absolute_error",cv=5,n_jobs=-1)

CV_RF=CV_RF.fit(train_X,train_y)

#I can directly predict with the best param!!

pred_RF=CV_RF.predict(val_X)

pred_RF=np.expm1(pred_RF)



print("The best parameter alpha for the RF regressor is:",CV_RF.best_params_)

print("The best score (NegMAE) for the RF regressor is:",-CV_RF.best_score_)



MAE_RF=MAE(val_y_u,pred_RF)

r2_RF=r2_score(val_y_u,pred_RF)



print("The MAE of the prediction is:", MAE_RF)

print("The R^2 of the prediction is:", r2_RF)
%%time

#For comparison later.

#GB is sequential

GB_reg=GradientBoostingRegressor()

param_alpha={"loss":["huber"], #Don't increase the grid too much, loss comparison is a bit unnecessary

            "learning_rate":[1e-2, 5e-2, 1e-1, 2e-1],

            "n_estimators":[300, 500],#Remove max_depth as a parameter, the best was None

            "max_features":[10, "sqrt", "log2", None]} #I think this values are enough

#La_train_X=RobustScaler().fit(train_X)

CV_GB=GridSearchCV(estimator=GB_reg,param_grid=param_alpha,scoring="neg_mean_absolute_error",cv=5,n_jobs=-1)

CV_GB=CV_GB.fit(train_X,train_y)

#I can directly predict with the best param!!

pred_GB=CV_GB.predict(val_X)

pred_GB=np.expm1(pred_GB)



print("The best parameter alpha for the GB regressor is:",CV_GB.best_params_)

print("The best score (NegMAE) for the GB regressor is:",-CV_GB.best_score_)



MAE_GB=MAE(val_y_u,pred_GB)

r2_GB=r2_score(val_y_u,pred_GB)



print("The MAE of the prediction is:", MAE_GB)

print("The R^2 of the prediction is:", r2_GB)
%%time

#For comparison later.

#GB is sequential

XGB_reg=XGBRegressor()

param_alpha={"eta":[1e-6,1e-4,1e-2,5e-2,1e-1,2.5e-1], #Don't increase the grid too much, loss comparison is a bit unnecessary

            "gamma":[1e-3,1e-1,1,2,5,10,20],

             "n_estimators":[300,500],

            "max_depth":[5,7,10]} #I think this values are enough

#La_train_X=RobustScaler().fit(train_X)

CV_XGB=GridSearchCV(estimator=XGB_reg,param_grid=param_alpha,scoring="neg_mean_absolute_error",cv=5,n_jobs=-1)

CV_XGB=CV_XGB.fit(train_X,train_y)

#I can directly predict with the best param!!

pred_XGB=CV_XGB.predict(val_X)

pred_XGB=np.expm1(pred_XGB)



print("The best parameter alpha for the GB regressor is:",CV_XGB.best_params_)

print("The best score (NegMAE) for the GB regressor is:",-CV_XGB.best_score_)



MAE_XGB=MAE(val_y_u,pred_XGB)

r2_XGB=r2_score(val_y_u,pred_XGB)



print("The MAE of the prediction is:", MAE_XGB)

print("The R^2 of the prediction is:", r2_XGB)
%%time

#For comparison later.

#GB is sequential

LGB_reg=LGB.LGBMRegressor()

param_alpha={"boosting_type":["gbdt"],

            "learning_rate":[1e-2, 5e-2, 1e-1, 2e-1, 3e-1],

            "num_leaves":[10, 15, 20],

            "n_estimators":[300,500],

            "stratified":[False],

            "max_depth":[5,7,10]} #I think this values are enough

#La_train_X=RobustScaler().fit(train_X)

CV_LGB=GridSearchCV(estimator=LGB_reg,param_grid=param_alpha,scoring="neg_mean_absolute_error",cv=5,n_jobs=-1)

CV_LGB=CV_LGB.fit(train_X,train_y)

#I can directly predict with the best param!!

pred_LGB=CV_LGB.predict(val_X)

pred_LGB=np.expm1(pred_LGB)



print("The best parameter alpha for the LGB regressor is:",CV_LGB.best_params_)

print("The best score (NegMAE) for the LGB regressor is:",-CV_LGB.best_score_)



MAE_LGB=MAE(val_y_u,pred_LGB)

r2_LGB=r2_score(val_y_u,pred_LGB)



print("The MAE of the prediction is:", MAE_LGB)

print("The R^2 of the prediction is:", r2_LGB)
%%time

#COMMENTED FOR SPEED!! UNCOMMENT IF TIME IS NOT CRITICAL ~tens of minutes

#SVR are slow, especially with large datasets (not this case), also, a 3D grid will increase time significantly.

#SVR_reg=SVR()

#param_alpha={"kernel":["linear","poly","rbf"], #poly and rbf are not a good idea, this is linearized

#            "degree":[2,3], #More than 3 may overfit unnecessarily. Small gamma is also tricky

#            "gamma":[1e-3, 1e-2, 1e-1, 1, 2, 5]} #I think this values are enough

#La_train_X=RobustScaler().fit(train_X)

#CV_SVR=GridSearchCV(estimator=SVR_reg,param_grid=param_alpha,scoring="neg_mean_absolute_error",cv=5,n_jobs=-1)

#CV_SVR=CV_SVR.fit(train_X,train_y)

#I can directly predict with the best param!!

#pred_SVR=CV_SVR.predict(val_X)

#pred_SVR=np.expm1(pred_SVR)



#print("The best parameter alpha for the SVR regressor is:",CV_SVR.best_params_)

#print("The best score (NegMAE) for the SVR regressor is:",-CV_SVR.best_score_)



#MAE_SVR=MAE(val_y_u,pred_SVR)

#r2_SVR=r2_score(val_y_u,pred_SVR)



#print("The MAE of the prediction is:", MAE_SVR)

#print("The R^2 of the prediction is:", r2_SVR)
models=["Lasso", "Ridge", "Elastic Net","Lasso AIC", "Lasso BIC", "Kernel Ridge","Random Forest", "GBDT", "XGBoost","LightBM"]

maes=[MAE_Lasso, MAE_Ridge, MAE_ENet, MAE_AIC, MAE_BIC, MAE_KerRidge, MAE_RF, MAE_GB, MAE_XGB, MAE_LGB]

r2s=[r2_Lasso, r2_Ridge, r2_ENet, r2_AIC, r2_BIC, r2_KerRidge, r2_RF, r2_GB, r2_XGB, r2_LGB]



#Add SVR at the end if uncommented

plt.subplots(figsize=(8,5))

plt.xticks(rotation=90)

ax = sns.barplot(x=models, y=maes)

sns.plt.ylim(14e3, 18e3)

plt.xlabel('Model', size=15)

plt.ylabel('MAE ($)', size=15)
plt.subplots(figsize=(8,5))

plt.xticks(rotation=90)

ax = sns.barplot(x=models, y=r2s)

sns.plt.ylim(0.8, 1)

plt.xlabel('Model', size=15)

plt.ylabel('R^2 of every model (-)', size=15)