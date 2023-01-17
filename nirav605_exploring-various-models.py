# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import sklearn 

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from matplotlib import cm as cm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

fulldata = train.append(test)

all_features = train.columns
TypeGrouping = train.columns.to_series().groupby(train.dtypes).groups

TypeGrouping
floatFeatures = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',

        'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',

        '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',

        'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',

        'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF',

        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

        'MiscVal', 'MoSold', 'YrSold','LotFrontage', 'MasVnrArea', 'GarageYrBlt']

print(len(floatFeatures))
train[floatFeatures].hist(figsize=(16,20))

plt.show()
fulldata["TotalLivingArea"] = fulldata["1stFlrSF"] + fulldata["2ndFlrSF"] + fulldata["GrLivArea"] + fulldata["WoodDeckSF"] + fulldata["OpenPorchSF"] + fulldata["EnclosedPorch"] + fulldata["3SsnPorch"] + fulldata["ScreenPorch"] + fulldata["ScreenPorch"] + fulldata["PoolArea"] 

fulldata["GrandTotalsquareFt"] =  fulldata["LotArea"] + fulldata["TotalBsmtSF"] + fulldata["TotalLivingArea"] + fulldata["GarageArea"]
fulldata["totalbath"] = fulldata["FullBath"] + 0.5*fulldata["HalfBath"]
train = fulldata[0:1459]

test = fulldata[1460:]

print(train.shape, test.shape)
train["SalePrice"].hist(figsize=(4,4))

train['SalePrice'] = np.log(train['SalePrice'])

train["SalePrice"].hist(figsize=(4,4))

train = train[train["TotalLivingArea"]<9900]

plt.scatter(x=train["TotalLivingArea"],y=train["SalePrice"])
#train = train[train["TotalLivingArea"]<9900]

#plt.scatter(x=train["totalbath"],y=train["SalePrice"])

boxplot=train.boxplot(column=["SalePrice"],by=["totalbath"],return_type='axes',figsize=(15,6),rot=90)

train = train[train["totalbath"]>0.5]
corr = train.corr()

print(corr.shape)

corr = corr.sort_values(by="SalePrice",axis=0,ascending=False).sort_values(by="SalePrice",axis=1,ascending=False)



plt.figure(figsize=(16, 16), dpi= 300)

plt.rcParams['figure.figsize'] = [20, 10]

cmap = cm.get_cmap('jet', 38)

#plt.imshow(df.corr(), interpolation="nearest", cmap=cmap)

plt.matshow(corr.values, interpolation="nearest", cmap=cmap)

plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns,rotation=90);

plt.yticks(range(len(corr.columns)), corr.columns);

#plt.show()
corr.sort_values(by="SalePrice")["SalePrice"]
print("Positively Correlated features \n")



boxplot=train.boxplot(column=["SalePrice"],by=["OverallQual"],return_type='axes',figsize=(12,6),rot=90)

plt.show()

train.plot.scatter(x="TotalLivingArea",y="SalePrice",figsize=(12,6))

plt.show()

train.plot.scatter(x="GrLivArea",y="SalePrice",figsize=(12,6))

plt.show()

boxplot=train.boxplot(column=["SalePrice"],by=["GarageCars"],return_type='axes',figsize=(12,6),rot=90)

plt.show()

train.plot.scatter(x="GarageArea",y="SalePrice",figsize=(12,6))

plt.show()

train.plot.scatter(x="TotalBsmtSF",y="SalePrice",figsize=(12,6))

plt.show()

print("\n")

print("Negatively Correlated features \n")

boxplot=train.boxplot(column=["SalePrice"],by=["KitchenAbvGr"],return_type='axes',figsize=(12,6),rot=90)

plt.show()

train.plot.scatter(x="EnclosedPorch",y="SalePrice",figsize=(12,6))

plt.show()

boxplot=train.boxplot(column=["SalePrice"],by=["MSSubClass"],return_type='axes',figsize=(12,6),rot=90)

plt.show()

nulloremptyfeatures = []

for i,fea in enumerate(all_features):

    tmp = pd.isna(fulldata[fea]).sum()

    if(tmp>0): 

        print(fea, " \t \t",  tmp, " \t \t ", round(tmp/len(fulldata),3), " \t \t ",fulldata[fea].dtype)

    
fulldata = fulldata.drop(["Alley","PoolQC","Fence","MiscFeature","FireplaceQu", "Id"],axis=1)

fulldata = pd.get_dummies(fulldata)

fulldata = fulldata.fillna(0)
train = fulldata[0:1459]

test = fulldata[1460:]

print(train.shape, test.shape)
train = train[train["totalbath"]>0.5]
train = pd.get_dummies(train)

train = train.fillna(0)
test = pd.get_dummies(test)

test = test.fillna(0)
def clean_data(data): 

    data = data.drop(["Alley","PoolQC","Fence","MiscFeature","FireplaceQu"],axis=1)

    data = pd.get_dummies(data)

    data = data.fillna(0)

    return data; 
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

fulldata = train.append(test)



## ---- remove nans and perform one-hot encoding -----

fulldata = clean_data(fulldata)



## ---- add a few new features we built -----

fulldata["TotalLivingArea"] = fulldata["1stFlrSF"] + fulldata["2ndFlrSF"] + fulldata["GrLivArea"] + fulldata["WoodDeckSF"] + fulldata["OpenPorchSF"] + fulldata["EnclosedPorch"] + fulldata["3SsnPorch"] + fulldata["ScreenPorch"] + fulldata["ScreenPorch"] + fulldata["PoolArea"] 

fulldata["GrandTotalsquareFt"] =  fulldata["LotArea"] + fulldata["TotalBsmtSF"] + fulldata["TotalLivingArea"] + fulldata["GarageArea"]

fulldata["totalbath"] = fulldata["FullBath"] + 0.5*fulldata["HalfBath"]



## ---- Split the full data into train and test ---- 

train = fulldata[0:1459]

test = fulldata[1460:]



## ---- change SalePrice to have normal distribution and remove a few outliers from training set ---

train['SalePrice'] = np.log(train['SalePrice'])

train = train[train["totalbath"]>0.5]

train = train[train["TotalLivingArea"]<9900]

train = train.drop(["Id"],axis=1)



## ---- Define all the features ---- 

target_feature = "SalePrice"

all_features = list(train.columns)

all_features.remove(target_feature)
from sklearn.linear_model import ElasticNetCV
ElasticNetCVModel2 = ElasticNetCV(l1_ratio=[1, 0.99, 0.98, 0.97, 0.96, 0.95, 0.9, 0.8, 0.7], eps=0.001, n_alphas=50, normalize=True, random_state=1,verbose=True, max_iter=500)

ElasticNetCVModel2.fit(train[all_features],train[target_feature])
x = np.array(ElasticNetCVModel2.alphas_)

y = np.array(ElasticNetCVModel2.mse_path_)

print(x.shape, y.shape)



ind = 0;

plt.scatter(x[ind,:],y[ind,:,0])

plt.scatter(x[ind,:],y[ind,:,1])

plt.scatter(x[ind,:],y[ind,:,2])

#plt.plot([ElasticNetCVModel2.alpha_,ElasticNetCVModel2.alpha_],[0,3e9])

#plt.xscale("log")

#plt.xlim(left=0,right=0.002)

#plt.ylim(top=3e9)
print("training score = ", ElasticNetCVModel2.score(train[all_features],train[target_feature]))

print("best parameters = alpha, l1_ratio = ", ElasticNetCVModel2.alpha_, ElasticNetCVModel2.l1_ratio_)
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsRegressor
knnr = KNeighborsRegressor()

param_grid = {'n_neighbors': np.arange(1, 25)}

KNeighborRegModel3 = GridSearchCV(knnr, param_grid, cv=5)

KNeighborRegModel3.fit(train[all_features],train[target_feature])
print("training score = ", KNeighborRegModel3.score(train[all_features],train[target_feature]))

print("best number of neighbors = ", KNeighborRegModel3.best_params_)
from sklearn.ensemble import AdaBoostRegressor
AdaBoostModel3 = AdaBoostRegressor(random_state=1, n_estimators=500)

AdaBoostModel3.fit(train[all_features],train[target_feature]) 
print("training score = ", AdaBoostModel3.score(train[all_features],train[target_feature]))
AddBoost_imprt = pd.Series(AdaBoostModel3.feature_importances_,index=all_features)

print((AddBoost_imprt[abs(AddBoost_imprt)>1e-2]).sort_values())
from keras.models import Sequential

from keras.layers import Dense


NN_model = Sequential()



# The Input Layer :

NN_model.add(Dense(128, kernel_initializer='normal',input_dim = train.shape[1]-1, activation='relu'))



# The Hidden Layers :

NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))



# The Output Layer :

NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))



# Compile the network :

NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

NN_model.summary()
NN_model.fit(train[all_features], train[target_feature], epochs=500, batch_size=32, validation_split = 0.2)

y_pred = np.asarray(NN_model.predict(train[all_features]))

y_true = np.asarray(train["SalePrice"].values)

y_pred = y_pred.reshape((-1,1))

y_true = y_true.reshape((-1,1))



residual = np.subtract(y_true,y_pred)

diff = y_true - np.mean(y_true)



RSS = np.vdot(residual,residual.T)

std = np.vdot(diff,diff)

NNModel_Score = 1 - RSS/std

print("RSS = ", RSS, "      and R^2 = ", NNModel_Score)

print("ElasticNet Model Score      = \t ", ElasticNetCVModel2.score(train[all_features],train[target_feature]))

print("Nearest Neighbor Reg. Score = \t ", KNeighborRegModel3.score(train[all_features],train[target_feature]))

print("AdaBoost Model Score        = \t ", AdaBoostModel3.score(train[all_features],train[target_feature]))

print("NeuralNet Model Score       = \t ", NNModel_Score)



output = pd.DataFrame()

output["Id"] = test["Id"]

output["LogSalePrice"] = ElasticNetCVModel2.predict(test[all_features])

output["SalePrice"] = np.exp(output["LogSalePrice"])

output = output.drop(["LogSalePrice"],axis=1)
output.to_csv("submission.csv",index=False)