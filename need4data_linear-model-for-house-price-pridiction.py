#packages

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

from scipy import stats





#importing data 

data=pd.read_csv('../input/train.csv')

anwsering_this=pd.read_csv('../input/test.csv')

submission_id=anwsering_this.Id.values
#inspecting data 

data.head()
data.info()
#check "Id" column whether or not have duplicates

if len(set(data.Id))==len(data.Id):

    print("No duplicates for Id column")



#drop "Id" column

data.drop("Id", axis = 1, inplace = True)

f,a=plt.subplots(figsize=(8,6))

plt.scatter(data.GrLivArea, data.SalePrice, c = "blue",marker='s')

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")

data=data[data.GrLivArea < 4000]

print(data.shape)

print("Four instances droped")
#target

print(data['SalePrice'].describe())



#plotting target column

plt.style.use('seaborn')

f,a=plt.subplots(figsize=(8,6))

plt.title('Before log(1+x)')

sns.distplot(data['SalePrice'],fit=stats.norm)

f,a=plt.subplots(figsize=(8,6))

stats.probplot(data['SalePrice'], plot=plt)





#log it 

data['SalePrice'] = np.log1p(data['SalePrice'])



#plot again

f,a=plt.subplots(figsize=(8,6))

plt.title('After log(1+x)')

sns.distplot(data['SalePrice'],fit=stats.norm)



f,a=plt.subplots(figsize=(8,6))



stats.probplot(data['SalePrice'],plot=plt)



#correlation analysis

corr=data.corr()

f,a=plt.subplots(figsize=(30,26))

sns.heatmap(corr,annot=True)

print('correlation with SalePrice Rank list')

print(corr['SalePrice'].sort_values(ascending=False))

# concatenate train and test datasets

# store the location of separate

sep=data.shape[0]

data_fts=data.drop(['SalePrice'],axis=1)

test_fts=anwsering_this.drop(['Id'],axis=1)

all_instances=pd.concat([data_fts,test_fts]).reset_index(drop=True)





#count missing values

null_counts=all_instances.isnull().sum().sort_values(ascending=False)

print(null_counts.head(38))
# Filling missing values





# NaN of PoolQC means "No Pool"

all_instances.loc[:,'PoolQC']=all_instances.loc[:,'PoolQC'].fillna('No')

# NaN of MiscFeature means "No Miscellaneous feature"

all_instances.loc[:,'MiscFeature']=all_instances.loc[:,'MiscFeature'].fillna('No')

# NaN of Alley means "No alley access"

all_instances.loc[:,'Alley']=all_instances.loc[:,'Alley'].fillna('No')

# NaN of Fence means "No Fence"

all_instances.loc[:,'Fence']=all_instances.loc[:,'Fence'].fillna('No')

# NaN of FireplaceQu means "No Fireplace"

all_instances.loc[:,'FireplaceQu']=all_instances.loc[:,'FireplaceQu'].fillna('No')

# NaN of LotFrontage means "No street connected to property"

all_instances.loc[:,'LotFrontage']=all_instances.loc[:,'LotFrontage'].fillna(0)

# NaN of GarageCond means "No Garage"

all_instances.loc[:,'GarageCond']=all_instances.loc[:,'GarageCond'].fillna('No')

# NaN of GarageQual means "No Garage"

all_instances.loc[:,'GarageQual']=all_instances.loc[:,'GarageQual'].fillna('No')

# NaN of GarageYrBlt means "No Garage"

all_instances.loc[:,'GarageYrBlt']=all_instances.loc[:,'GarageYrBlt'].fillna(0)

# NaN of GarageFinish means "No Garage"

all_instances.loc[:,'GarageFinish']=all_instances.loc[:,'GarageFinish'].fillna('No')

# NaN of GarageType means "No Garage"

all_instances.loc[:,'GarageType']=all_instances.loc[:,'GarageType'].fillna('No')

# NaN of BsmtCond means "No Basement"

all_instances.loc[:,'BsmtCond']=all_instances.loc[:,'BsmtCond'].fillna('No')

# NaN of BsmtExposure means "No Basement"

all_instances.loc[:,'BsmtExposure']=all_instances.loc[:,'BsmtExposure'].fillna('No_B')

# NaN of BsmtQual means "No Basement"

all_instances.loc[:,'BsmtQual']=all_instances.loc[:,'BsmtQual'].fillna('No')

# NaN of BsmtFinType2 means "No Basement"

all_instances.loc[:,'BsmtFinType2']=all_instances.loc[:,'BsmtFinType2'].fillna('No')

# NaN of BsmtFinType1 means "No Basement"

all_instances.loc[:,'BsmtFinType1']=all_instances.loc[:,'BsmtFinType1'].fillna('No')

# NaN of MasVnrType means "No Masonry veneer"

all_instances.loc[:,'MasVnrType']=all_instances.loc[:,'MasVnrType'].fillna('No')

# NaN of MasVnrArea means "No Masonry veneer" 

all_instances.loc[:,'MasVnrArea']=all_instances.loc[:,'MasVnrArea'].fillna(0)



# NaN of MSZoning fill with "RL",Mode

all_instances.loc[:,'MSZoning']=all_instances.loc[:,'MSZoning'].fillna("RL")



# NaN of BsmtHalfBath means "No Basement"

all_instances.loc[:,'BsmtHalfBath']=all_instances.loc[:,'BsmtHalfBath'].fillna(0)

# NaN of Utilities fill with "AllPub",Mode

all_instances.loc[:,'Utilities']=all_instances.loc[:,'Utilities'].fillna("AllPub")

# NaN of Functional fill with "Typ",Mode

all_instances.loc[:,'Functional']=all_instances.loc[:,'Functional'].fillna("Typ")

# NaN of BsmtFullBath fill with 0

all_instances.loc[:,'BsmtFullBath']=all_instances.loc[:,'BsmtFullBath'].fillna(0)

# NaN of BsmtFinSF2 fill with 0

all_instances.loc[:,'BsmtFinSF2']=all_instances.loc[:,'BsmtFinSF2'].fillna(0)

# NaN of BsmtFinSF1 fill with 0 

all_instances.loc[:,'BsmtFinSF1']=all_instances.loc[:,'BsmtFinSF1'].fillna(0) 

# NaN of Exterior2nd fill with "VinylSd",Mode

all_instances.loc[:,'Exterior2nd']=all_instances.loc[:,'Exterior2nd'].fillna("VinylSd") 

# NaN of BsmtUnfSF fill with 0

all_instances.loc[:,'BsmtUnfSF']=all_instances.loc[:,'BsmtUnfSF'].fillna(0) 

# NaN of TotalBsmtSF fill with 0

all_instances.loc[:,'TotalBsmtSF']=all_instances.loc[:,'TotalBsmtSF'].fillna(0) 

# NaN of Exterior1st fill with "VinylSd",Mode

all_instances.loc[:,'Exterior1st']=all_instances.loc[:,'Exterior1st'].fillna("VinylSd") 

# NaN of SaleType fill with "WD",Mode

all_instances.loc[:,'SaleType']=all_instances.loc[:,'SaleType'].fillna("WD") 

# NaN of Electrical fill with "SBrkr",Mode

all_instances.loc[:,'Electrical']=all_instances.loc[:,'Electrical'].fillna("SBrkr")

# NaN of KitchenQual fill with "TA",Mode

all_instances.loc[:,'KitchenQual']=all_instances.loc[:,'KitchenQual'].fillna("TA")

# NaN of GarageArea fill with 0

all_instances.loc[:,'GarageArea']=all_instances.loc[:,'GarageArea'].fillna(0)

# NaN of GarageCars fill with 0

all_instances.loc[:,'GarageCars']=all_instances.loc[:,'GarageCars'].fillna(0)



# check if any missing values remain

print("Remain "+str(all_instances.isnull().sum().sum())+" missing values.")

















# numeric features suppose to be categorical feature

n2c=['MSSubClass','MoSold']

for i in n2c:

    all_instances[i]=all_instances[i].astype(str)

    



# summerize categorical features suppose to be numeric feature

c2n_=['GarageCond','GarageQual','GarageFinish','FireplaceQu','KitchenQual','HeatingQC','BsmtFinType2','BsmtFinType1','BsmtExposure','BsmtCond','BsmtQual','ExterCond','ExterQual','Fence']



# sequential categorical features can be transform to numerical feature

lv4_NA=['PoolQC']

lv5_full=['ExterQual','ExterCond','HeatingQC','KitchenQual']

lv5_NA=['BsmtQual','BsmtCond','FireplaceQu','GarageQual','GarageCond']

Bsmt_Type=['BsmtFinType2','BsmtFinType1']



# lv4_NA

all_instances['PoolQC']=all_instances['PoolQC'].replace({'No':0,'Fa':1,'TA':2,'Gd':3,'Ex':4})



# lv5_full

for i in lv5_full:

    all_instances[i]=all_instances[i].replace({'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})



# lv5_NA    

for i in lv5_NA:

    all_instances[i]=all_instances[i].replace({'No':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5})



# Bsmt_Type

for i in Bsmt_Type:

    all_instances[i]=all_instances[i].replace({'No':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6})





# rest of it 

all_instances['GarageFinish']=all_instances['GarageFinish'].replace({'No':0,'Unf':1,'RFn':2,'Fin':3})

all_instances['BsmtExposure']=all_instances['BsmtExposure'].replace({'No_B':0,'No':1,'Mn':2,'Av':3,'Gd':4})

all_instances['Fence']=all_instances['Fence'].replace({'No':0,'MnWw':1,'GdWo':2,'MnPrv':3,'GdPrv':4})





# Combine features, you can combine more if you like.

# Overall combination

all_instances["OverallRate"] = all_instances["OverallQual"] * all_instances["OverallCond"]

# Garage combination

all_instances["GarageRate"] = all_instances["GarageQual"] * all_instances["GarageCond"]

# Exterior combination

all_instances["ExterRate"] = all_instances["ExterQual"] * all_instances["ExterCond"]

# Bsmt combination

all_instances["BsmtRate"]=all_instances["BsmtQual"] * all_instances["BsmtCond"]



# Pool combination

all_instances["PoolRate"]=all_instances["PoolQC"] * all_instances["PoolArea"]



# Make polynomial features 

# Adding Polynomial features for top 10 correlated features

top_10_corr=list(corr['SalePrice'].sort_values(ascending=False).index)[1:11]

for i in top_10_corr:

    all_instances[i+'_s2']=all_instances[i] ** 2

    all_instances[i+'_s3']=all_instances[i] ** 3

    all_instances[i+'_sqrt']=np.sqrt(all_instances[i])

print(all_instances.shape)
from scipy.stats import skew



# Seperate numeric columns and object columns.

categorical_features = all_instances.select_dtypes(include = ["object"]).columns

numerical_features = all_instances.select_dtypes(exclude = ["object"]).columns



print("Numerical features : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))

all_instances_num = all_instances[numerical_features]

all_instances_cat = all_instances[categorical_features]





# log(1+x) transform all skewed columns

skewness = all_instances_num.apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.5]

print(str(skewness.shape[0]) + " skewed numerical features to log transform")

skewed_features = skewness.index

all_instances_num[skewed_features] = np.log1p(all_instances_num[skewed_features])



# concate them back

all_instances = pd.concat([all_instances_num, all_instances_cat], axis = 1)

print(all_instances.shape)

all_instances=pd.get_dummies(all_instances,drop_first=True)

print(all_instances.shape)
from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV,Ridge,Lasso,ElasticNet

from sklearn.metrics import mean_squared_error, make_scorer



# Slicing our pre-processed dataset as before we combine train dataset and test dataset

new_data=all_instances[:sep]

y=data['SalePrice']

new_test=all_instances[sep:]

# split dataset first

X_train,X_test,y_train,y_test=train_test_split(new_data,y,test_size=0.3,random_state=111)



# Standardize our train dataset and test dataset separatly

stdSc = StandardScaler()



# X_train dataset we use "fit_transform" and X_test we use "transform" because in that case means we use the "fit"(mean and std) from X_train to transform

# our X_test,this make sure our data is equally rescaled.



# Another thing,StandardScaler will automatic converte values to numeric,so we need serparate numeric and other type of columns. 

X_train.loc[:, numerical_features] = stdSc.fit_transform(X_train.loc[:, numerical_features])

X_test.loc[:, numerical_features] = stdSc.transform(X_test.loc[:, numerical_features])

# sign-flip the RMSE for parameter search

scorer = make_scorer(mean_squared_error, greater_is_better = False)



# real cross validation score 

def rmse_cv_train(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))

    return(rmse)



def rmse_cv_test(model):

    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))

    return(rmse)

# Linear Regression

lr = LinearRegression()



# Look at predictions on training and validation set

print("RMSE on Training set :", rmse_cv_train(lr).mean())

print("RMSE on Test set :", rmse_cv_test(lr).mean())

# RigdeCV implements ridge regression with built-in cross-validation of the alpha parameter, similar to GridSearchCV to Parameter optimize.

ridge = RidgeCV(alphas = [0.01, 0.04, 0.08, 0.1, 0.4, 0.8, 1, 4, 8, 10, 40, 80])

ridge.fit(X_train, y_train)

alpha= ridge.alpha_

print("Best alpha :", alpha)



print("Try again for more precision with alphas centered around " + str(alpha))

ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 

                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,

                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 

                cv = 10)

ridge.fit(X_train, y_train)

alpha_ridge = ridge.alpha_

print("Best alpha :", alpha_ridge)



print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())

print("Ridge RMSE on Test set :", rmse_cv_test(ridge).mean())



# Final scoring

ridge_=Ridge(alpha=alpha_ridge)

ridge_.fit(X_train,y_train)

ridge_predict=ridge_.predict(X_test)

ridge_RMSE=np.sqrt(mean_squared_error(ridge_predict,y_test))

print("Ridge regression RMSE :",ridge_RMSE)











# LassoCV similar to RigdeCV

lasso = LassoCV(alphas = [0.0001, 0.0004, 0.0008, 0.001, 0.004, 0.008, 0.01, 0.04, 0.08, 0.1, 

                          0.4, 0.8, 1], 

                max_iter = 50000, cv = 10)

lasso.fit(X_train, y_train)

alpha = lasso.alpha_

print("Best alpha :", alpha)



print("Try again for more precision with alphas centered around " + str(alpha))

lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 

                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 

                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 

                          alpha * 1.4], 

                max_iter = 50000, cv = 10)

lasso.fit(X_train, y_train)

alpha_lasso = lasso.alpha_

print("Best alpha :", alpha_lasso)



print("Lasso RMSE on Training set :", rmse_cv_train(lasso).mean())

print("Lasso RMSE on Test set :", rmse_cv_test(lasso).mean())



# Final scoring

lasso_=Lasso(alpha=alpha_lasso)

lasso_.fit(X_train,y_train)

lasso_predict=lasso_.predict(X_test)

lasso_RMSE=np.sqrt(mean_squared_error(lasso_predict,y_test))

print("Lasso regression RMSE :",lasso_RMSE)
elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],

                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 

                                    0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 

                          max_iter = 50000, cv = 10)

elasticNet.fit(X_train, y_train)

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )



print("Try again for more precision with l1_ratio centered around " + str(ratio))

elasticNet = ElasticNetCV(l1_ratio = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],

                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 

                          max_iter = 50000, cv = 10)

elasticNet.fit(X_train, y_train)

if (elasticNet.l1_ratio_ > 1):

    elasticNet.l1_ratio_ = 1    

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )



print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) + 

      " and alpha centered around " + str(alpha))

elasticNet = ElasticNetCV(l1_ratio = ratio,

                          alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9, 

                                    alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, 

                                    alpha * 1.35, alpha * 1.4], 

                          max_iter = 50000, cv = 10)

elasticNet.fit(X_train, y_train)

if (elasticNet.l1_ratio_ > 1):

    elasticNet.l1_ratio_ = 1    

alpha_eln = elasticNet.alpha_

ratio_eln = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio_eln)

print("Best alpha :", alpha_eln )



print("ElasticNet RMSE on Training set :", rmse_cv_train(elasticNet).mean())

print("ElasticNet RMSE on Test set :", rmse_cv_test(elasticNet).mean())



# Final scoring

elasticNet_=ElasticNet(l1_ratio=ratio_eln,alpha=alpha_eln)

elasticNet_.fit(X_train,y_train)

elasticNet_predict=elasticNet_.predict(X_test)

elasticNet_RMSE=np.sqrt(mean_squared_error(elasticNet_predict,y_test))

print("elasticNet RMSE :",elasticNet_RMSE)
# create new model object first

Final_model=Ridge(alpha=alpha_ridge)



# fit all data and Standardize first



stdSc_ = StandardScaler()

new_data.loc[:, numerical_features] = stdSc_.fit_transform(new_data.loc[:, numerical_features])



# Standardize test

new_test.loc[:, numerical_features] = stdSc_.transform(new_test.loc[:, numerical_features])



Final_model.fit(new_data,y)

submission=Final_model.predict(new_test)



# reverse our data since we log(1+x) transformed before

submission=np.expm1(submission)



df_submission=pd.DataFrame({'Id':submission_id,'SalePrice':submission})



df_submission.to_csv('submission.csv',index=False)