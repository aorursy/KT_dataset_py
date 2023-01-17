%%time

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv("/kaggle/input/predict-the-housing-price/train.csv")

test = pd.read_csv("/kaggle/input/predict-the-housing-price/Test.csv")

test1 = pd.read_csv("/kaggle/input/predict-the-housing-price/Test.csv")

train.head(3)

#test.head(3)
train.info()
print(test.shape)

print(train.shape)
nul =  train.isnull().sum().reset_index()

nul[nul[0]>0]
for i in train.columns:

    print(i)

    print(train[i].unique())
train = train.drop(["Id"],axis=1)

test = test.drop(["Id"],axis=1)
obj_col = train.select_dtypes(object).columns

num_col = train.select_dtypes(exclude=object).columns



#Fill nan values with None 

train[obj_col] = train[obj_col].fillna("None")

test[obj_col] = test[obj_col].fillna("None")



#viewing null values

nul =  train.isnull().sum().reset_index()

nul[nul[0]>0]
a = list(nul[nul[0]>0]["index"])

train[a].describe()
test["MasVnrArea"].mode()
#masvnrarea has only 6 null values & Quantile of 50% is also vary much from mean. so, drop it

train = train[np.isfinite(train['MasVnrArea'])]



#Both LotFrontage & GarageYrBlt has mean mostly near to quantile 50%

train["LotFrontage"] = train["LotFrontage"].fillna(train["LotFrontage"].mean())

train["GarageYrBlt"] = train["GarageYrBlt"].fillna(train["GarageYrBlt"].mean())

test["LotFrontage"] = test["LotFrontage"].fillna(test["LotFrontage"].mean())

test["GarageYrBlt"] = test["GarageYrBlt"].fillna(test["GarageYrBlt"].mean())

test["MasVnrArea"] = test["MasVnrArea"].fillna(0)





nul =  test.isnull().sum().reset_index()

nul[nul[0]>0]
#countplot



for i in obj_col:

    #plt.figure(figsize=[8,8])

    sns.set(style="whitegrid")

    sns.countplot(x = i,data=train)

    plt.show()
# to find outlier

for i in num_col:

    #plt.figure(figsize=[8,8])

    sns.set(style="whitegrid")

    sns.violinplot(train[i])

    plt.show()

    
num_col1 = num_col.drop('SalePrice')

for i in num_col1:

    #plt.figure(figsize=[8,8])

    sns.set(style="whitegrid")

    sns.jointplot(train[i],y=train["SalePrice"],kind="reg")

    plt.show()
from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()

train[num_col1] = scalar.fit_transform(train[num_col1])

train[num_col1].describe()
sns.pairplot(data = train,x_vars=train.columns,y_vars="SalePrice")
corr = train.corr()

plt.figure(figsize=[20,20])

sns.heatmap(corr, linewidths=1, annot=True)
#Geting top 10 corr attributes

train.corr()["SalePrice"].reset_index().sort_values(["SalePrice"], ascending=False)[:10]
train_dum = pd.get_dummies(train)

train_dum.head()



test_dum = pd.get_dummies(test)

test_dum.head()

train_dum.head()



print(train_dum.shape)

print(test_dum.shape)
from sklearn.model_selection import train_test_split



X= train_dum[[]]

y = train_dum[["SalePrice"]]

X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.7,test_size=0.3,random_state=100)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)
from sklearn.model_selection import train_test_split



X= train_dum.drop(["SalePrice"],axis=1)

y = train_dum[["SalePrice"]]

X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.7,test_size=0.3,random_state=100)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)
from sklearn import datasets, linear_model

from sklearn.metrics import r2_score

model = linear_model.LinearRegression()
model.fit(X_train[num_col1],y_train)

preds = model.predict(X_test[num_col1])



print("R2 score : %.2f" % r2_score(y_test,preds))
model.fit(X_train[["OverallQual"]],y_train)

preds = model.predict(X_test[["OverallQual"]])



print("R2 score : %.2f" % r2_score(y_test,preds))
model.fit(X_train[["GrLivArea"]],y_train)

preds = model.predict(X_test[["GrLivArea"]])



print("R2 score : %.2f" % r2_score(y_test,preds))
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
from sklearn.feature_selection import RFE

rfe = RFE(logreg, n_features_to_select=10)
rfe = rfe.fit(X_train, y_train)
#print(rfe.support_)

rank = rfe.ranking_

print(rfe.ranking_)           
sel_fea = X_train.columns[rfe.ranking_<175]

sel_fea.size
a = zip(X_train,rank)

rfe_col = pd.DataFrame(a, columns = ['Col', 'Rank']).sort_values("Rank")

top_col = list(rfe_col["Col"])



#print(top_col)

#rfe_col
logreg2 = LogisticRegression()

model2 = logreg2.fit(X_train[sel_fea], y_train)

#model2.coef_

#model2.intercept_
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()

vif['Features'] = X_train[sel_fea].columns

vif['VIF'] = [variance_inflation_factor(X_train[sel_fea].values, i) for i in range(X_train[sel_fea].shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif[:5]
vif_col = vif[:180]

col = list(vif_col["Features"])

print(col)
from statsmodels.sandbox.regression.predstd import wls_prediction_std

X = sm.add_constant(X_train[col])

y = y_train
model = sm.OLS(y, X)

results = model.fit()

print(results.summary())
x = ["BsmtFinType1_GLQ","HouseStyle_1Story","HeatingQC_Ex","BsmtQual_Gd","MSZoning_RL","KitchenQual_Gd","FireplaceQu_None",

     "BsmtCond_TA","SaleType_WD","ExterQual_Gd","RoofStyle_Hip","Functional_Typ","PavedDrive_Y","ExterQual_TA",

     "RoofMatl_CompShg","Condition2_Norm","BsmtFinType1_Unf","HeatingQC_TA","Exterior2nd_VinylSd","BsmtExposure_Av",

    "BsmtFinType1_ALQ","Neighborhood_CollgCr","HouseStyle_2Story","FireplaceQu_Gd","SaleCondition_Abnorml","OverallQual",

    "FireplaceQu_TA","YearRemodAdd","SaleType_WD","FullBath","Neighborhood_OldTown","BldgType_TwnhsE","Neighborhood_NridgHt",

     "HalfBath","LandSlope_Gtl","Neighborhood_CollgCr","Neighborhood_Edwards","Neighborhood_SawyerW","OverallQual",

    "SaleCondition_Abnorml","BldgType_Twnhs","LandContour_Lvl","YearRemodAdd","BedroomAbvGr","BsmtFullBath","Neighborhood_CollgCr",

    "Neighborhood_NridgHt","Neighborhood_Edwards","Exterior1st_BrkFace","Neighborhood_NoRidge","Neighborhood_SawyerW",

    "OpenPorchSF","WoodDeckSF","ScreenPorch","Neighborhood_NoRidge","FireplaceQu_Ex","Neighborhood_Crawfor","Functional_Typ",

    "Neighborhood_StoneBr","Neighborhood_Timber","OpenPorchSF","WoodDeckSF","ScreenPorch"]

model = linear_model.LinearRegression()

model.fit(X_train[x],y_train)

preds = model.predict(X_test[x])



#print("R2 score : %.2f" % r2_score(y_test,preds))
test_dum[num_col1].head()
from sklearn import datasets, linear_model

from sklearn.metrics import r2_score



model = linear_model.LinearRegression()

model1 = LogisticRegression()



X_train = train_dum[x]

X_test = test_dum[x]

y_train = train_dum["SalePrice"]



model.fit(X_train,y_train)

model1.fit(X_train,y_train)



preds = model.predict(X_test)

pre_log = model1.predict(X_test)

ids = test1["Id"].to_list()
file = open("result.csv", "w")

file.write("Id,SalePrice\n")

    

for id_, pred in zip(ids, pre_log):

    file.write("{},{}\n".format(id_, pred))

file.close()