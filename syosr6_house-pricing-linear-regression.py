import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

import random

import os

print(os.listdir("../input"))

DataTrain = pd.read_csv("../input/train.csv")

DataTest = pd.read_csv("../input/test.csv")
Y = DataTrain["SalePrice"].copy()

QuantTrain = DataTrain.select_dtypes(include=['float','integer']).copy()

QuantTrain.drop("SalePrice", inplace=True, axis=1)

QualTrain = DataTrain.select_dtypes(include=['object']).copy()



QuantTest = DataTest.select_dtypes(include=['float','integer']).copy()

QualTest = DataTest.select_dtypes(include=['object']).copy()
msno.matrix(df= QuantTrain, figsize=(20,14), color=(0.5,0,0))
QuantTrain = QuantTrain.fillna({"LotFrontage": QuantTrain['LotFrontage'].median()})

QuantTrain = QuantTrain.fillna({"GarageYrBlt": 0})

QuantTrain = QuantTrain.fillna({"MasVnrArea": QuantTrain['MasVnrArea'].median()})
QuantTrain.isnull().values.any()
msno.matrix(df= QuantTest, figsize=(20,14), color=(0.5,0,0))
QuantTest = QuantTest.fillna({"LotFrontage": QuantTest['LotFrontage'].median()})

QuantTest = QuantTest.fillna({"MasVnrArea": QuantTest['MasVnrArea'].median()})

QuantTest = QuantTest.fillna({"GarageYrBlt": 0})

QuantTest = QuantTest.fillna({"BsmtFinSF1": QuantTest['BsmtFinSF1'].median()})

QuantTest = QuantTest.fillna({"BsmtFinSF2": QuantTest['BsmtFinSF2'].median()})

QuantTest = QuantTest.fillna({"BsmtUnfSF": QuantTest['BsmtUnfSF'].median()})

QuantTest = QuantTest.fillna({"TotalBsmtSF": QuantTest['TotalBsmtSF'].median()})

QuantTest = QuantTest.fillna({"BsmtFullBath": QuantTest['BsmtFullBath'].median()})

QuantTest = QuantTest.fillna({"BsmtHalfBath": QuantTest['BsmtHalfBath'].median()})

QuantTest = QuantTest.fillna({"GarageCars": QuantTest['GarageCars'].median()})

QuantTest = QuantTest.fillna({"GarageArea": QuantTest['GarageArea'].median()})
QuantTest.isnull().values.any()
msno.matrix(df= QualTrain, figsize=(20,14), color=(0.5,0,0))
ToDrop=["PoolQC","Alley"]

QualTrain.drop(ToDrop, inplace=True, axis=1)

QualTest.drop(ToDrop, inplace=True, axis=1)
QualTrain = QualTrain.fillna({"BsmtQual": "NoB"})

QualTrain = QualTrain.fillna({"BsmtCond": "NoB"})

QualTrain = QualTrain.fillna({"BsmtExposure": "NoB"})

QualTrain = QualTrain.fillna({"BsmtFinType1": "NoB"})

QualTrain = QualTrain.fillna({"BsmtFinType2": "NoB"})



QualTrain = QualTrain.fillna({"FireplaceQu": "NoF"})



QualTrain = QualTrain.fillna({"GarageType": "NoG"})

QualTrain = QualTrain.fillna({"GarageFinish": "NoG"})

QualTrain = QualTrain.fillna({"GarageQual": "NoG"})

QualTrain = QualTrain.fillna({"GarageCond": "NoG"})



QualTrain = QualTrain.fillna({"MiscFeature": "NoGMisc"})



QualTrain = QualTrain.fillna({"Fence": "NoF"})



QualTrain = QualTrain.fillna({"Electrical": QualTrain['Electrical'].value_counts().idxmax()})
nan_rows = QualTrain[QualTrain.isnull().T.any().T]

nan_rows
for i in [234,529,650,936,973,977,1243,1278]:

    print(QuantTrain.loc[i,"MasVnrArea"])
QualTrain = QualTrain.fillna({"MasVnrType": "None"})
QualTrain.isnull().values.any()
msno.matrix(df= QualTest, figsize=(20,14), color=(0.5,0,0))
QualTest = QualTest.fillna({"BsmtQual": "NoB"})

QualTest = QualTest.fillna({"BsmtCond": "NoB"})

QualTest = QualTest.fillna({"BsmtExposure": "NoB"})

QualTest = QualTest.fillna({"BsmtFinType1": "NoB"})

QualTest = QualTest.fillna({"BsmtFinType2": "NoB"})



QualTest = QualTest.fillna({"FireplaceQu": "NoF"})



QualTest = QualTest.fillna({"GarageType": "NoG"})

QualTest = QualTest.fillna({"GarageFinish": "NoG"})

QualTest = QualTest.fillna({"GarageQual": "NoG"})

QualTest = QualTest.fillna({"GarageCond": "NoG"})



QualTest = QualTest.fillna({"MiscFeature": "NoGMisc"})



QualTest = QualTest.fillna({"Fence": "NoF"})
QualTest[QualTest["MasVnrType"].isnull()]
for i in [231,246,422,532,544,581,851,865,880,889,908,1132,1150,1197,1226,1402]:

    print(QuantTest.loc[i,"MasVnrArea"])
QualTest['MasVnrType'].value_counts().idxmax()
QualTest = QualTest.fillna({"MasVnrType": QualTest['MasVnrType'].value_counts().idxmax()})

#or QualTest = QualTest.fillna({"MasVnrType": "None"})
QualTest = QualTest.fillna({"KitchenQual": QualTest['KitchenQual'].value_counts().idxmax()})

QualTest = QualTest.fillna({"MSZoning": QualTest['MSZoning'].value_counts().idxmax()})

QualTest = QualTest.fillna({"Exterior1st": QualTest['Exterior1st'].value_counts().idxmax()})

QualTest = QualTest.fillna({"Functional": QualTest['Functional'].value_counts().idxmax()})

QualTest = QualTest.fillna({"Utilities": QualTest['Utilities'].value_counts().idxmax()})

QualTest = QualTest.fillna({"SaleType": QualTest['SaleType'].value_counts().idxmax()})

QualTest = QualTest.fillna({"Exterior2nd": QualTest['Exterior2nd'].value_counts().idxmax()})
nan_rows = QualTest[QualTest.isnull().T.any().T]

nan_rows
QualTest.isnull().values.any()
QualTrain = pd.get_dummies(QualTrain)

QualTest = pd.get_dummies(QualTest)
cols_to_scale = ['MSSubClass','LotFrontage','LotArea', 'OverallQual',

       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',

       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',

       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

       'MiscVal', 'MoSold', 'YrSold']

scaler = preprocessing.StandardScaler()

for col in cols_to_scale:

    QuantTrain[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(QuantTrain[col])),columns=[col])

    QuantTest[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(QuantTest[col])),columns=[col])
Xtrain = pd.concat([QuantTrain,QualTrain,Y], axis=1, sort=False)

Xtest = pd.concat([QuantTest,QualTest], axis=1, sort=False)
corrmat = Xtrain.corr()

corrmat['SalePrice'].sort_values(ascending = False)[:100]
corr = ["OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","1stFlrSF","FullBath","BsmtQual_Ex",

        "TotRmsAbvGrd","YearBuilt","YearRemodAdd","KitchenQual_Ex","Foundation_PConc","MasVnrArea","Fireplaces",

        "ExterQual_Gd","ExterQual_Ex","BsmtFinType1_GLQ","HeatingQC_Ex","GarageFinish_Fin","Neighborhood_NridgHt",

        "BsmtFinSF1","SaleType_New","SaleCondition_Partial","FireplaceQu_Gd","GarageType_Attchd",

        "LotFrontage","MasVnrType_Stone","Neighborhood_NoRidge","WoodDeckSF","KitchenQual_Gd","2ndFlrSF",

        "OpenPorchSF","BsmtExposure_Gd","Exterior2nd_VinylSd","Exterior1st_VinylSd","HalfBath","GarageCond_TA",

        "LotArea","GarageYrBlt","FireplaceQu_Ex","CentralAir_Y","GarageQual_TA","MSZoning_RL","HouseStyle_2Story",

        "Electrical_SBrkr","RoofStyle_Hip","GarageType_BuiltIn","BsmtQual_Gd","PavedDrive_Y","BsmtFullBath",

        "LotShape_IR1","Neighborhood_StoneBr","BsmtUnfSF","MasVnrType_BrkFace","Fence_NoF","GarageFinish_RFn",

        "RoofMatl_WdShngl","BedroomAbvGr","FireplaceQu_TA","LotConfig_CulDSac","Neighborhood_Somerst",

        "BldgType_1Fam","BsmtExposure_Av","Exterior1st_CemntBd","Exterior2nd_CmentBd","Neighborhood_Timber",

        "LotShape_IR2","LandContour_HLS","BsmtFinType2_Unf","Functional_Typ","Condition1_Norm","ScreenPorch",

        "ExterCond_TA","BsmtCond_TA","Heating_GasA","PoolArea","MSZoning_FV","BsmtCond_Gd","Exterior2nd_ImStucc",

        "Neighborhood_CollgCr","MiscFeature_NoGMisc","Neighborhood_Crawfor","Neighborhood_Veenker",

        "Neighborhood_ClearCr"]

       

XTrain = Xtrain[corr]

XTest = Xtest[corr]
corr1 = ["OverallQual","GrLivArea","GarageCars","GarageArea","TotalBsmtSF","1stFlrSF","FullBath","BsmtQual_Ex",

        "TotRmsAbvGrd","YearBuilt","YearRemodAdd","KitchenQual_Ex","Foundation_PConc","MasVnrArea","Fireplaces",

        "ExterQual_Gd","ExterQual_Ex","BsmtFinType1_GLQ","HeatingQC_Ex","GarageFinish_Fin","Neighborhood_NridgHt",

         "GarageYrBlt","SalePrice"]

Visualisation = Xtrain[corr1]

plt.figure(figsize =(15,8))

sns.heatmap(Visualisation.corr(),annot=True,cmap='coolwarm')

plt.show()
#ToDrop = ["GarageArea"]

#XTrain.drop(ToDrop, inplace=True, axis=1)

#XTest.drop(ToDrop, inplace=True, axis=1)
fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(16, 10))

axes = np.ravel(axes)

col_name = ["OverallQual","GrLivArea","GarageCars","TotalBsmtSF","1stFlrSF","FullBath"]

for i, c in zip(range(6), col_name):

    Visualisation.plot.scatter(ax=axes[i], x=c, y='SalePrice', sharey=True, colorbar=False, c='orange')
index= [(Visualisation[Visualisation['GrLivArea'] > 5]).index & (Visualisation[Visualisation['SalePrice'] < 300000]).index ,

         (Visualisation[Visualisation['TotalBsmtSF'] > 10]).index & (Visualisation[Visualisation['SalePrice'] < 300000]).index ,

         (Visualisation[Visualisation['1stFlrSF'] > 8]).index & (Visualisation[Visualisation['SalePrice'] < 300000]).index]

index
# Create linear regression object

LM = LinearRegression()

# Train the model using the training DataDrame

LM.fit(XTrain,Y)

#Prediction of the Sale Prices from the testing DataDrame

predictions = LM.predict(XTest)
submit = pd.DataFrame({'Id': Xtest.loc[:,"Id"], 'SalePrice': predictions})

submit.to_csv('submission.csv', index=False)