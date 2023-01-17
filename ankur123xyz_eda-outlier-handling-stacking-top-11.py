import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm,skew

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor , StackingRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.linear_model import Ridge,Lasso,LinearRegression

from sklearn.model_selection import KFold,cross_val_score, GridSearchCV

from sklearn.preprocessing import PolynomialFeatures

from xgboost import XGBRegressor

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
print("Dimensions of train set are:-",train.shape[0],",",train.shape[1])

print("Dimensions of test set are:-",test.shape[0],",",test.shape[1])
pd.set_option("display.max_columns",81)
train.head()
corr = train.corr()
corr["SalePrice"].sort_values(ascending=False)
x=corr["SalePrice"].sort_values(ascending=False)[:11].index

mask = np.zeros_like(corr.loc[x,x])

mask[np.triu_indices_from(mask)]=True



sns.heatmap(corr.loc[x,x],mask=mask,annot=True,cmap="coolwarm")
sns.pairplot(train[x[0:6]],x_vars=list(x[1:6]),y_vars=[x[0]],diag_kind="kde")
f,ax = plt.subplots(1,2,figsize=(15,8))

sns.boxplot("OverallQual","SalePrice",data=train,ax=ax[0])

sns.boxplot("GarageCars","SalePrice",data=train,ax=ax[1])
index = len(train)

y=train["SalePrice"]

train.drop("SalePrice",axis=1,inplace=True)

dataset = pd.concat([train,test]).reset_index(drop=True)
dataset.isnull().sum()[dataset.isnull().sum()>0].sort_values(ascending=False)
df_mszon=dataset.groupby(["Neighborhood","MSZoning"])["Id"].count().reset_index().groupby(["Neighborhood"])

maximum_mszon = df_mszon.max()

maximum_mszon= maximum_mszon.drop("Id",axis=1)



max_dict_mszon = maximum_mszon.to_dict()

def mapper_mszon(x):

    for index,val in zip(max_dict_mszon,max_dict_mszon.values()):

        for index1,val1 in val.items():

            if(x==index1):

                return val1



dataset.loc[dataset["MSZoning"].isnull(),"MSZoning"] = dataset.loc[dataset["MSZoning"].isnull(),"Neighborhood"].apply(lambda x: mapper_mszon(x))
df_lotf=dataset.groupby(["Neighborhood","LotConfig"])["LotFrontage"].mean()



max_dict_lotf = df_lotf.to_dict()

def mapper_lotf(x1,x2):

    for index,val in zip(max_dict_lotf,max_dict_lotf.values()):

        if((x1==index[0]) & (x2==index[1])):

            return val



dataset.loc[dataset["LotFrontage"].isnull(),"LotFrontage"] = dataset.loc[dataset["LotFrontage"].isnull(),["Neighborhood","LotConfig"]].apply(lambda x: mapper_lotf(x[0],x[1]),axis=1)

dataset["LotFrontage"] = dataset.groupby(["Neighborhood"])["LotFrontage"].transform(lambda x: x.median())
dataset["Alley"].fillna("NA",inplace=True)

dataset["Utilities"].fillna("NA",inplace=True)

dataset["Exterior1st"].fillna("NA",inplace=True)

dataset["Exterior2nd"].fillna("NA",inplace=True)

dataset["MasVnrType"].fillna("NA",inplace=True)

dataset["MasVnrArea"].fillna(0,inplace=True)

dataset["BsmtQual"].fillna("NA",inplace=True)

dataset["BsmtCond"].fillna("NA",inplace=True)

dataset["BsmtExposure"].fillna("No",inplace=True)

dataset["BsmtFinType1"].fillna("NA",inplace=True)

dataset["BsmtFinSF1"].fillna(0,inplace=True)

dataset["BsmtFinType2"].fillna("NA",inplace=True)

dataset["BsmtFinSF2"].fillna(0,inplace=True)

dataset["BsmtUnfSF"].fillna(0,inplace=True)

dataset["TotalBsmtSF"].fillna(0,inplace=True)

dataset["BsmtFullBath"].fillna(0,inplace=True)

dataset["BsmtHalfBath"].fillna(0,inplace=True)

dataset["FireplaceQu"].fillna("NA",inplace=True)

dataset["GarageType"].fillna("NA",inplace=True)

dataset["GarageYrBlt"].fillna(0,inplace=True)

dataset["GarageFinish"].fillna("NA",inplace=True)

dataset["GarageCars"].fillna(0,inplace=True)

dataset["GarageArea"].fillna(0,inplace=True)

dataset["GarageQual"].fillna("NA",inplace=True)

dataset["GarageCond"].fillna("NA",inplace=True)

dataset["PoolQC"].fillna("NA",inplace=True)

dataset["Fence"].fillna("NA",inplace=True)

dataset["MiscFeature"].fillna("NA",inplace=True)
dataset["Functional"].fillna("Typ",inplace=True)

dataset["Electrical"].fillna(dataset["Electrical"].mode()[0],inplace=True)

dataset["KitchenQual"].fillna(dataset["KitchenQual"].mode()[0],inplace=True)

dataset["SaleType"].fillna("Oth",inplace=True)
dataset["MSSubClass"]=dataset["MSSubClass"].astype("category")
x=sns.distplot(y)

x.set_title("Distribution plot for Sale Price")
x=sns.distplot(np.log1p(y),fit=norm)

x.set_title("Distribution plot for Sale Price")
y= np.log1p(y)
dataset = dataset.replace({"Alley" : {"Grvl" : 1, "Pave" : 2,"NA":0},

                       "BsmtCond" : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},

                       "BsmtQual" : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "FireplaceQu" : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 

                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},

                       "GarageCond" : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "GarageQual" : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "GarageFinish" : {"Fin" : 3, "RFn" : 2, "Unf" : 1, "NA" : 0},

                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

                       "PoolQC" : {"NA" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

                       "Street" : {"Grvl" : 1, "Pave" : 2, "NA":0},

                       "Utilities" : {"ELO" : 1, "NASeWa" : 2, "NASewr" : 3, "AllPub" : 4}}

                     )
cats = ["basement_flag","fire_flag","wooddeck_flag","garage_flag","pool_flag","fence_flag"]

dataset["basement_flag"]=np.where(dataset["TotalBsmtSF"]>0,1,0)

dataset["fire_flag"]=np.where(dataset["Fireplaces"]>0,1,0)

dataset["wooddeck_flag"]=np.where(dataset["WoodDeckSF"]>0,1,0)

dataset["garage_flag"]=np.where(dataset["GarageArea"]>0,1,0)

dataset["porch_flag"]=np.where((dataset["OpenPorchSF"]+dataset["EnclosedPorch"]+dataset["3SsnPorch"]\

                                +dataset["ScreenPorch"])>0,1,0)

dataset["pool_flag"] = np.where(dataset["PoolArea"]>0,1,0)

dataset["fence_flag"] = np.where(dataset["Fence"]=="NA",1,0)

dataset[cats]=dataset[cats].astype("category")
dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']
outlier_features = ["GrLivArea","TotalSF","GarageArea"]



means,sds = np.mean(dataset[outlier_features]),np.std(dataset[outlier_features])

lower,upper = means - 2.5*sds , means + 2.5*sds

def compare(x):

    count=0

    for col in outlier_features:

        if x[col]>lower[col] and x[col]<upper[col]:

            count=count+1

    if count==len(outlier_features):

        return True

    else: 

        return False



y= y.array

sub_data = dataset.loc[:index-1]  

test_1 = dataset.loc[index:,:]



data_train_new = sub_data.loc[sub_data.apply(lambda x: compare(x),axis=1)]

y= y[sub_data.apply(lambda x: compare(x),axis=1)]



index = len(data_train_new)

dataset = pd.concat([data_train_new,test_1]).reset_index(drop=True)
numeric_feats = dataset.dtypes[(dataset.dtypes != "object") & (dataset.dtypes != "category")].index

skewed_feats = dataset[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness = skewness[abs(skewness) > 0.5]



skewed_features = skewness.index

dataset[skewed_features] = np.log1p(dataset[skewed_features])
dataset.drop("Id",axis=1,inplace=True)
dataset = pd.get_dummies(dataset, drop_first= True)

train_1 = dataset.loc[:index-1,:]

test_1 = dataset.loc[index:,:]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(train_1,y,test_size=0.2, random_state=42)
from sklearn.preprocessing import RobustScaler

rs = RobustScaler()

X_train= rs.fit_transform(X_train)

X_test = rs.transform(X_test)

X_submit = rs.transform(test_1.values)
kf = KFold(n_splits=10,shuffle=True,random_state=42)

rf = RandomForestRegressor(n_estimators=500)

xgb = XGBRegressor()

gb= GradientBoostingRegressor()

svr =SVR(C=70,epsilon=.115)

ridge = Ridge(alpha=1.75,solver ="lsqr")

lasso = Lasso(alpha=.0009)



clf=[\

     ("XGBoost",xgb,{"n_estimators":[200],"max_depth":[3],"learning_rate":[.1],"subsample":[1],"colsample_bytree":[1],"gamma":[0],"lambda":[.001,.01]})\

     ,("Gradient Boost",gb,{})\

     ,("SVR",svr,{"C":[60,65,70,75,80],"epsilon":[.11,.115,.12]})\

    ,("Ridge",ridge,{"alpha":[1.4,1.5,1.6,1.75,1.8]})\

     ,("Lasso",lasso,{"alpha":[.0005,.0006,.0007,.001,.01,.1]})]



est=[]

results_data=pd.DataFrame(columns=["Name","Train_Score","Test_Score"])



i=0

for name,reg,param_grid in clf:

    gs= GridSearchCV(reg,param_grid=param_grid,cv=kf,scoring="neg_mean_squared_error")

    gs.fit(X_train,y_train)

    y_pred = gs.best_estimator_.predict(X_test)

    test_score = np.sqrt(mean_squared_error(y_test, y_pred))

    results_data.loc[i,]= [name,-gs.best_score_,test_score]

    i=i+1

    est.append([name,gs.best_estimator_])

    

sc = StackingRegressor(estimators=est[1:],final_estimator= ridge,cv=kf)

sc.fit(X_train,y_train)

y_trn= sc.predict(X_train)

y_tst= sc.predict(X_test)    

train_score = np.sqrt(mean_squared_error(y_train, y_trn))

test_score = np.sqrt(mean_squared_error(y_test, y_tst))

results_data.loc[i,]= ["Stack",train_score,test_score]
results_data
y_submit= np.expm1(sc.predict(X_submit))
data_submit= pd.DataFrame({"Id":test.Id,"SalePrice":y_submit})
data_submit.to_csv("submit_housing.csv",index=False)