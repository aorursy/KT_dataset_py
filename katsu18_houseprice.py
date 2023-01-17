#pip install XGBoost
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import seaborn as sns

import matplotlib.pyplot as plt
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

test.head()
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

y = train["SalePrice"]

train.tail()
#missing data

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
print(train["GarageCond"].value_counts())

print(train["GarageCond"].isnull().sum())
train["Electrical"]=train["Electrical"].fillna("SBrkr")

train["MasVnrArea"]=train["MasVnrArea"].fillna(train["MasVnrArea"].median())

train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



train["GarageYrBlt"]=train["GarageYrBlt"].fillna(train["GarageYrBlt"].median())





d_list = ["FireplaceQu","ExterQual","BsmtQual","GarageQual","ExterCond","BsmtCond","HeatingQC","KitchenQual","GarageCond","PoolQC"]

d_class = ["Ex","Gd","TA","Fa","Po",np.NaN]

d_int = [10,8,6,4,2,0]

for x in d_list:

    for a,b in [[d_class,d_int]]:

        train[x] = train[x].replace(a,b)

    train[x]=train[x].astype(int)



    

d2_class = ['GLQ','ALQ','BLQ','Rec','LwQ','Unf',np.NaN ]

d2_int = [10,8,6,4,2,0,-2]

d2_list = ["BsmtFinType1","BsmtFinType2"]

for x in d2_list:

    for a,b in [[d2_class,d2_int]]:

        train[x] = train[x].replace(a,b)

    train[x]=train[x].astype(int)

    



d3_class = ['RFn','Unf','Fin',np.NaN ]

d3_int = [4,-4,2,0]

d3_list = ["GarageFinish"]

for x in d3_list:

    for a,b in [[d3_class,d3_int]]:

        train[x] = train[x].replace(a,b)

    train[x]=train[x].astype(int)

    

d5_class = ['BrkFace', 'None' ,'Stone' ,'BrkCmn', np.nan]

d5_int = [2,-4,3,0,0]

d5_list = ["MasVnrType"]

for x in d5_list:

    for a,b in [[d5_class,d5_int]]:

        train[x] = train[x].replace(a,b)

    train[x]=train[x].astype(int)



list_a = ["Exterior2nd","Exterior1st"]

for a in list_a:

    for i in train[a].unique():

        if i == "VinylSd":

            train[a] = train[a].replace(i,3)

        elif i =="CmentBd":

            train[a] = train[a].replace(i,1)

        elif i =="AsbShng":

            train[a] = train[a].replace(i,-1)

        elif i =="MetalSd":

            train[a] = train[a].replace(i,-1)

        elif i =="HdBoard":

            train[a] = train[a].replace(i,-1)

        elif i =="Wd Sdng":

            train[a] = train[a].replace(i,-2)

        else:

            train[a] = train[a].replace(i,0)

train[a]=train[a].astype(int)



for i in train["GarageType"].unique():

    if i == "Attchd":

        train["GarageType"] = train["GarageType"].replace(i,4)

    elif i =="BuiltIn":

        train["GarageType"] = train["GarageType"].replace(i,3)

    elif i =="Detchd":

        train["GarageType"] = train["GarageType"].replace(i,-3)

    else:

        train["GarageType"] = train["GarageType"].replace(i,0)

train["GarageType"]=train["GarageType"].astype(int)
list_A = ["MiscFeature","Alley","Fence","GarageType","GarageFinish","BsmtExposure","MasVnrType"]
for a in list_A:

    print(a)

    print(train[a].unique())
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#total2 = (train==0).sum().sort_values(ascending=False)

#percent2 = ((train==0).sum()/(train==0).count()).sort_values(ascending=False)

#zero_data = pd.concat([total2, percent2], axis=1, keys=['Total', 'Percent'])

#zero_data.head(16)
missing_data.index[:14]
#zero_data.index[:16]
del_data =[]

for i in missing_data.index[:18]:

    del_data.append(i)

    

#for i in zero_data.index[:18]:

#    if i not in del_data:

#        del_data.append(i)

    

del_data
for i in del_data:

    train_del = train.drop([i], axis=1)

train_del



train_del_int = train_del.select_dtypes(include=int)

train_del_int
#price_year = ["YearBuilt","YearRemodAdd","SalePrice"]

#for i in train_del_int.columns:

#        if i not in price_year:

#            train_del_int[i] = train_del_int[i]+1

#            train_del_int[i] = np.log(train_del_int[i])

#train_del_int
corrmat = train_del_int.corr()

#f, ax = plt.subplots(figsize=(37, 17))

#sns.set(font_scale=1.25)

#sns.heatmap(corrmat, vmax=.8, square=True,cbar=True, annot=True,fmt='.2f', annot_kws={'size': 10});

corrmat["SalePrice"].sort_values(ascending=False)
train_c= pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

train_c = train_c.drop(['SalePrice'], axis=1)

train_c = pd.concat([train_c,test])
train_c["Electrical"]=train_c["Electrical"].fillna("SBrkr")

train_c["MasVnrArea"]=train_c["MasVnrArea"].fillna(train_c["MasVnrArea"].median())

train_c["LotFrontage"] = train_c.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
train["Neighborhood"].unique()
for i in train["Neighborhood"].unique():

    if i == "NridgHt":

        train_c["Neighborhood"] = train_c["Neighborhood"].replace(i,8)

    elif i =="NoRidge":

        train_c["Neighborhood"] = train_c["Neighborhood"].replace(i,6)

    elif i =="StoneBr":

        train_c["Neighborhood"] = train_c["Neighborhood"].replace(i,2)

    elif i =="Timber":

        train_c["Neighborhood"] = train_c["Neighborhood"].replace(i,2)

    else:

        train_c["Neighborhood"] = train_c["Neighborhood"].replace(i,1)

train_c["Neighborhood"]=train_c["Neighborhood"].astype(int)

        
train["Foundation"].unique()
for i in train["Foundation"].unique():

    if i == "PConc":

        train_c["Foundation"] = train_c["Foundation"].replace(i,5)

    elif i =="CBlock":

        train_c["Foundation"] = train_c["Foundation"].replace(i,-3)

    elif i =="BrkTil":

        train_c["Foundation"] = train_c["Foundation"].replace(i,-2)

    elif i =="Slab":

        train_c["Foundation"] = train_c["Foundation"].replace(i,-1)

    else:

        train_c["Foundation"] = train_c["Foundation"].replace(i,0)

train_c["Foundation"]=train_c["Foundation"].astype(int)

        
train_c["Foundation"].value_counts()
train_c["Neighborhood"].value_counts()
train["SaleType"].unique()
for i in train_c["CentralAir"].unique():

    if i == "Y":

        train_c["CentralAir"] = train_c["CentralAir"].replace(i,1)

    else:

        train_c["CentralAir"] = train_c["CentralAir"].replace(i,0)

train_c["CentralAir"]=train_c["CentralAir"].astype(int)
for i in train_c["LotShape"].unique():

    if i == "Reg":

        train_c["LotShape"] = train_c["LotShape"].replace(i,-3)

    elif i == "IR1":

        train_c["LotShape"] = train_c["LotShape"].replace(i,2)

    elif i == "IR2":

        train_c["LotShape"] = train_c["LotShape"].replace(i,1)

    else:

        train_c["LotShape"] = train_c["LotShape"].replace(i,0)

train_c["LotShape"]=train_c["LotShape"].astype(int)
for i in train_c["Electrical"].unique():

    if i == "SBrkr":

        train_c["Electrical"] = train_c["Electrical"].replace(i,3)

    elif i == "FuseA":

        train_c["Electrical"] = train_c["Electrical"].replace(i,-2)

    elif i == "FuseF":

        train_c["Electrical"] = train_c["Electrical"].replace(i,-1)

    else:

        train_c["Electrical"] = train_c["Electrical"].replace(i,0)

train_c["Electrical"]=train_c["Electrical"].astype(int)
for i in train_c["RoofStyle"].unique():

    if i == "Gable":

        train_c["RoofStyle"] = train_c["RoofStyle"].replace(i,-2)

    elif i == "Hip":

        train_c["RoofStyle"] = train_c["RoofStyle"].replace(i,3)

    else:

        train_c["RoofStyle"] = train_c["RoofStyle"].replace(i,0)

train_c["RoofStyle"]=train_c["RoofStyle"].astype(int)
train_c["RoofStyle"].unique()
for i in train_c["PavedDrive"].unique():

    if i == "Y":

        train_c["PavedDrive"] = train_c["PavedDrive"].replace(i,1)

    elif i == "N":

        train_c["PavedDrive"] = train_c["PavedDrive"].replace(i,-1)

    else:

        train_c["PavedDrive"] = train_c["PavedDrive"].replace(i,0)

train_c["PavedDrive"]=train_c["PavedDrive"].astype(int)
for i in train_c["SaleType"].unique():

    if i == "New":

        train_c["SaleType"] = train_c["SaleType"].replace(i,4)

    elif i =="WD":

        train_c["SaleType"] = train_c["SaleType"].replace(i,-2)

    else:

        train_c["SaleType"] = train_c["SaleType"].replace(i,0)

train_c["SaleType"]=train_c["SaleType"].astype(int)
cont = "SaleCondition"

for i in train_c[cont].unique():

    if i == "Partial":

        train_c[cont] = train_c[cont].replace(i,4)

    elif i =="Abnorml":

        train_c[cont] = train_c[cont].replace(i,-1)

    elif i =="Normal":

        train_c[cont] = train_c[cont].replace(i,-1)

    else:

        train_c[cont] = train_c[cont].replace(i,0)

train_c[cont]=train_c[cont].astype(int)
cont = "MSZoning"

for i in train_c[cont].unique():

    if i == "C (all)":

        train_c[cont] = train_c[cont].replace(i,-1)

    elif i =="RL":

        train_c[cont] = train_c[cont].replace(i,3)

    elif i =="RM":

        train_c[cont] = train_c[cont].replace(i,-3)

    else:

        train_c[cont] = train_c[cont].replace(i,0)

train_c[cont]=train_c[cont].astype(int)
train_c["HouseStyle"].unique()
cont = "HouseStyle"

for i in train_c[cont].unique():

    if i == "2Story":

        train_c[cont] = train_c[cont].replace(i,3)

    elif i =="1.5Fin":

        train_c[cont] = train_c[cont].replace(i,-2)

    elif i =="SFoyer":

        train_c[cont] = train_c[cont].replace(i,-1)

    elif i =="1.5Unf":

        train_c[cont] = train_c[cont].replace(i,-1)

    else:

        train_c[cont] = train_c[cont].replace(i,0)

train_c[cont]=train_c[cont].astype(int)
cont = "BsmtExposure"

for i in train_c[cont].unique():

    if i == "Gd":

        train_c[cont] = train_c[cont].replace(i,3)

    elif i =="Av":

        train_c[cont] = train_c[cont].replace(i,2)

    elif i =="Mn":

        train_c[cont] = train_c[cont].replace(i,0)

    else:

        train_c[cont] = train_c[cont].replace(i,-3)

train_c[cont]=train_c[cont].astype(int)
train_c["BsmtExposure"].unique()
cont = "RoofMatl"

for i in train_c[cont].unique():

    if i == "WdShngl":

        train_c[cont] = train_c[cont].replace(i,2)

    elif i =="WdShake":

        train_c[cont] = train_c[cont].replace(i,1)

    elif i =="CompShg":

        train_c[cont] = train_c[cont].replace(i,-1)

    else:

        train_c[cont] = train_c[cont].replace(i,0)

train_c[cont]=train_c[cont].astype(int)
train_c["RoofMatl"].unique()
cont = "BldgType"

for i in train_c[cont].unique():

    if i == "1Fam":

        train_c[cont] = train_c[cont].replace(i,2)

    elif i =="Duplex":

        train_c[cont] = train_c[cont].replace(i,-1)

    elif i =="Twnhs":

        train_c[cont] = train_c[cont].replace(i,-1)

    elif i =="2fmCon":

        train_c[cont] = train_c[cont].replace(i,-1)

    else:

        train_c[cont] = train_c[cont].replace(i,0)

train_c[cont]=train_c[cont].astype(int)
cont = "Fence"

for i in train_c[cont].unique():

    if i == "MnPrv":

        train_c[cont] = train_c[cont].replace(i,-2)

    elif i =="GdWo":

        train_c[cont] = train_c[cont].replace(i,1)

    else:

        train_c[cont] = train_c[cont].replace(i,0)

train_c[cont]=train_c[cont].astype(int)
train_c["Fence"].unique()
cont = "Condition1"

for i in train_c[cont].unique():

    if i == "Norm":

        train_c[cont] = train_c[cont].replace(i,11)

    elif i =="PosN":

        train_c[cont] = train_c[cont].replace(i,5)

    elif i =="PosA":

        train_c[cont] = train_c[cont].replace(i,4)

    elif i =="Feedr":

        train_c[cont] = train_c[cont].replace(i,-1)

    elif i =="RRAe":

        train_c[cont] = train_c[cont].replace(i,-5)

    else:

        train_c[cont] = train_c[cont].replace(i,0)

train_c[cont]=train_c[cont].astype(int)
train_c["Condition1"].unique()
cont = "Condition2"

for i in train_c[cont].unique():

    if i == "PosN":

        train_c[cont] = train_c[cont].replace(i,5)

    elif i =="PosA":

        train_c[cont] = train_c[cont].replace(i,5)

    elif i =="PosA":

        train_c[cont] = train_c[cont].replace(i,4)

    elif i =="Norm":

        train_c[cont] = train_c[cont].replace(i,3)

    elif i =="Feedr":

        train_c[cont] = train_c[cont].replace(i,-5)

    elif i =="RRNn":

        train_c[cont] = train_c[cont].replace(i,-4)

    elif i =="Artery":

        train_c[cont] = train_c[cont].replace(i,-3)

    else:

        train_c[cont] = train_c[cont].replace(i,0)

train_c[cont]=train_c[cont].astype(int)
train_c["Condition2"].unique()
list_a = ["Exterior2nd","Exterior1st"]

for a in list_a:

    for i in train_c[a].unique():

        if i == "VinylSd":

            train_c[a] = train_c[a].replace(i,3)

        elif i =="CmentBd":

            train_c[a] = train_c[a].replace(i,1)

        elif i =="AsbShng":

            train_c[a] = train_c[a].replace(i,-1)

        elif i =="MetalSd":

            train_c[a] = train_c[a].replace(i,-1)

        elif i =="HdBoard":

            train_c[a] = train_c[a].replace(i,-1)

        elif i =="Wd Sdng":

            train_c[a] = train_c[a].replace(i,-2)

        else:

            train_c[a] = train_c[a].replace(i,0)

train_c[a]=train_c[a].astype(int)



for i in train_c["GarageType"].unique():

    if i == "Attchd":

        train_c["GarageType"] = train_c["GarageType"].replace(i,4)

    elif i =="BuiltIn":

        train_c["GarageType"] = train_c["GarageType"].replace(i,3)

    elif i =="Detchd":

        train_c["GarageType"] = train_c["GarageType"].replace(i,-3)

    else:

        train_c["GarageType"] = train_c["GarageType"].replace(i,0)

train_c["GarageType"]=train_c["GarageType"].astype(int)
train_c["Electrical"]=train_c["Electrical"].fillna("SBrkr")

train_c["MasVnrArea"]=train_c["MasVnrArea"].fillna(train_c["MasVnrArea"].median())

train_c["LotFrontage"] = train_c.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))



d_list = ["FireplaceQu","ExterQual","BsmtQual","GarageQual","ExterCond","BsmtCond","HeatingQC","KitchenQual","GarageCond","PoolQC"]

d_class = ["Ex","Gd","TA","Fa","Po",np.NaN]

d_int = [10,8,6,4,2,0]

for x in d_list:

    for a,b in [[d_class,d_int]]:

        train_c[x] = train_c[x].replace(a,b)

    train_c[x]=train_c[x].astype(int)



    

d2_class = ['GLQ','ALQ','BLQ','Rec','LwQ','Unf',np.NaN ]

d2_int = [10,8,6,4,2,0,-2]

d2_list = ["BsmtFinType1","BsmtFinType2"]

for x in d2_list:

    for a,b in [[d2_class,d2_int]]:

        train_c[x] = train_c[x].replace(a,b)

    train_c[x]=train_c[x].astype(int)

    



d3_class = ['RFn','Unf','Fin',np.NaN ]

d3_int = [4,-4,2,0]

d3_list = ["GarageFinish"]

for x in d3_list:

    for a,b in [[d3_class,d3_int]]:

        train_c[x] = train_c[x].replace(a,b)

    train_c[x]=train_c[x].astype(int)

    

d5_class = ['BrkFace', 'None' ,'Stone' ,'BrkCmn', np.nan]

d5_int = [2,-4,3,0,0]

d5_list = ["MasVnrType"]

for x in d5_list:

    for a,b in [[d5_class,d5_int]]:

        train_c[x] = train_c[x].replace(a,b)

    train_c[x]=train_c[x].astype(int)
#for i in del_data:

#    train_c = train_c.drop([i], axis=1)

#train_c
train_c.columns
train_c_int = train_c.select_dtypes(include=int)

train_c_int = train_c_int.drop("Id",axis=1)

train_c_int
#price_year = ["YearBuilt","YearRemodAdd","SalePrice"]

#for i in train_c_int.columns:

#    if i not in price_year:

#        train_c_int[i] = train_c_int[i]+1    

#        train_c_int[i] = np.log(train_c_int[i])

#train_c_int

train_c_objects = train_c.select_dtypes(include=object)

train_c_objects
train_c_objects=pd.get_dummies(train_c_objects)

train_c_objects
train_c_int["Id"]=train_c["Id"]

train_c_objects["Id"]=train_c["Id"]



train_mg = pd.merge(train_c_int,train_c_objects,on="Id").fillna(0)

train_mg
train_X = train_mg[:1460]

train_X
train_X = pd.merge(train_X,train[["SalePrice","Id"]],on="Id")

corrmat = train_X.corr()

#f, ax = plt.subplots(figsize=(12, 9))

#sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix

#k = 20 #number of variables for heatmap

#f, ax = plt.subplots(figsize=(k, k/2))

#cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

#cm = np.corrcoef(train_X[cols].values.T)

#sns.set(font_scale=1.25)

#hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

#plt.show()
cols2 = corrmat.nlargest(219, 'SalePrice')['SalePrice'].index

cols = cols2[:70]

cols
cols_back = np.flip(cols2)[:30]

cols_back
cols_sel = []

for i in cols:

    if i not in cols_sel:

        cols_sel.append(i)

#cols_sel.remove('SaleType')

#cols_sel.remove('GarageQual')

#cols_sel.remove('GarageArea')

#cols_sel.remove('Exterior1st_VinylSd')

#cols_sel.remove('SaleCondition_Partial')

#cols_sel.remove('Exterior1st_VinylSd')



cols_sel
for a in cols_back:

    if a not in cols_sel:

        cols_sel.append(a)

cols_sel
corrmat = train_X.corr()

corrmat["SalePrice"].sort_values(ascending=False)
#saleprice correlation matrix

f, ax = plt.subplots(figsize=(200,100))

cm = np.corrcoef(train_X[cols_sel].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols_sel, xticklabels=cols_sel)

plt.show()
cols_sel.remove('SalePrice')
train_X = train_mg[:1460][cols_sel]

train_X.head()
test_X = train_mg[1460:][cols_sel].reset_index(drop=True)

test_X.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import KernelPCA

from sklearn.metrics import mean_squared_error

import xgboost as xgb
#params ={'n_estimators': [5000],'random_state': [0],"criterion":["mse"],"bootstrap":[True]}

#rfr = RandomForestRegressor()

#model = GridSearchCV(rfr,params,cv = 5,n_jobs = -1,verbose=True)



clf = xgb.XGBRegressor()

params = {"colsample_bytree": [0.41],

          "learning_rate":[0.1],

          "max_depth": [3],

          "min_child_weight":[1],

          "n_estimators":[10000],

          "nthread":[-1],

          "objective": ['reg:squarederror'],

          "random_state":[0],

          "silent":[0,1],

          "reg_alpha":[0.30879],

          "reg_lambda":[0.8571],

          "subsample":[0.5]

         }

# ハイパーパラメータ探索

clf_cv = GridSearchCV(clf,params, verbose=1,n_jobs = -1,cv = 10 )

clf_cv.fit(train_X, y)

print(clf_cv.best_params_, clf_cv.best_score_)



# 改めて最適パラメータで学習

model = xgb.XGBRegressor(**clf_cv.best_params_)

model.fit(train_X,y)
#model.fit(train_X,y)
y_pred_train = model.predict(train_X)

#model.best_estimator_
y_pred_train = model.predict(train_X)


plt.figure(figsize=(20,8))



plt.plot(y_pred_train,color="red",label="pred",alpha=0.8)

plt.plot(y,color="b",label="y",alpha=0.5)

plt.ylabel("cnt")

plt.xlabel("index")

plt.title("ture and ans")

plt.legend(loc="upper right")
print(mean_squared_error(y,y_pred_train))

print(mean_squared_error(y,y_pred_train)-112813358.78)

print(mean_squared_error(y,y_pred_train)-128303074.52467023)

print(mean_squared_error(y,y_pred_train)-142352.56307125353)

print(mean_squared_error(y,y_pred_train)-116753.57060873999)

print(mean_squared_error(y,y_pred_train)-7233.087039790741)
y_pred_test = model.predict(test_X)

y_pred_test
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission["SalePrice"] =y_pred_test

submission.to_csv("house_submission.csv",index=None)
y