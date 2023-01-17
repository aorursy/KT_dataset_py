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
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
from collections import Counter
pd.set_option("display.max_columns",100)
train_df=pd.read_csv("../input/train.csv")
test_df=pd.read_csv("../input/test.csv")
train_df.info()
test_df.info()
#欠損値やデータ0が多いものを削除
del train_df["Alley"]
del train_df["PoolArea"]
del train_df["PoolQC"]
del train_df["Fence"]
del train_df["MiscFeature"]
del train_df["MiscVal"]
del train_df["Fireplaces"]
del train_df["FireplaceQu"]
del train_df["MasVnrType"]
del train_df["MasVnrArea"]
#欠損値の扱い（離散値は最頻値、連続値は中央値で補完）
train_df["BsmtQual"].fillna('TA',inplace=True)
train_df["BsmtCond"].fillna('TA',inplace=True)
train_df["BsmtExposure"].fillna('No',inplace=True)
train_df["BsmtFinType1"].fillna('Unf',inplace=True)
train_df["BsmtFinType2"].fillna('Unf',inplace=True)
train_df["Electrical"].fillna('SBrkr',inplace=True)
train_df["GarageType"].fillna('Attchd',inplace=True)
train_df["GarageFinish"].fillna('Unf',inplace=True)
train_df["GarageQual"].fillna('TA',inplace=True)
train_df["GarageCond"].fillna('TA',inplace=True)
train_df["LotFrontage"].fillna(train_df["LotFrontage"].median(),inplace=True)
train_df["GarageYrBlt"].fillna(train_df["GarageYrBlt"].median(),inplace=True)
#ラベルエンコーダ化
train_df["MSZoning"]=train_df["MSZoning"].replace("RL",1).replace("RM",2).replace("FV",3).replace(["RH","C (all)"],3)
train_df["LotShape"]=train_df["LotShape"].replace("Reg",1).replace("IR1",2).replace(["IR2","IR3"],3)
train_df["LandContour"]=train_df["LandContour"].replace("Lvl",1).replace("Bnk",2).replace("HLS",3).replace("Low",4)
train_df["Street"]=train_df["Street"].replace("Pave",1).replace("Grvl",2)
train_df["Utilities"]=train_df["Utilities"].replace("AllPub",1).replace("NoSeWa",2)
train_df["LotConfig"]=train_df["LotConfig"].replace("Inside",1).replace("Corner",2).replace("CulDSac",3).replace(["FR2","FR3"],4)
train_df["LandSlope"]=train_df["LandSlope"].replace("Gtl",1).replace(["Mod","Sev"],2)
train_df["Neighborhood"]=train_df["Neighborhood"].replace('NAmes',1).replace('CollgCr',2).replace('OldTown',3).replace('Edwards',4).replace('Somerst',5).replace('Gilbert',6).replace('NridgHt',7).replace('Sawyer',8).replace('NWAmes',9).replace('SawyerW',10).replace('BrkSide',11).replace('Crawfor',12).replace('Mitchel',13).replace('NoRidge',14).replace('Timber',15).replace('IDOTRR',16).replace('ClearCr',17).replace('StoneBr',18).replace('SWISU',19).replace('MeadowV',20).replace('Blmngtn',21).replace('BrDale',22).replace('Veenker',23).replace(['NPkVill','Blueste'],24)
train_df["Condition1"]=train_df["Condition1"].replace('Norm',1).replace('Feedr',2).replace('Artery',3).replace('RRAn',4).replace(['PosN','RRAe','PosA','RRNn','RRNe'],5)
train_df["Condition2"]=train_df["Condition2"].replace('Norm',1).replace(['Feedr','Artery','RRNn','PosN','PosA','RRAn','RRAe'],2)
train_df["BldgType"]=train_df["BldgType"].replace('1Fam',1).replace('TwnhsE',2).replace('Duplex',3).replace('Twnhs',4).replace('2fmCon',5)
train_df["HouseStyle"]=train_df["HouseStyle"].replace('1Story',1).replace('2Story',2).replace('1.5Fin',3).replace('SLvl', 4).replace('SFoyer',5).replace('1.5Unf',6).replace('2.5Unf',7).replace('2.5Fin',8)
train_df["RoofStyle"]=train_df["RoofStyle"].replace('Gable',1).replace('Hip',2).replace(['Flat','Gambrel','Mansard','Shed'],3)
train_df["RoofMatl"]=train_df["RoofMatl"].replace('CompShg',1).replace(['Tar&Grv','WdShngl','WdShake','Metal','Membran','Roll','ClyTile'],2)
train_df["Exterior1st"]=train_df["Exterior1st"].replace('VinylSd',1).replace('HdBoard',2).replace('MetalSd',3).replace('Wd Sdng',4).replace('Plywood',5).replace('CemntBd',6).replace('BrkFace',7).replace(['WdShing','Stucco','AsbShng','BrkComm','Stone','AsphShn','ImStucc','CBlock'],8)
train_df["Exterior2nd"]=train_df["Exterior2nd"].replace('VinylSd',1).replace('MetalSd',2).replace('HdBoard',3).replace('Wd Sdng',4).replace('Plywood',5).replace('CmentBd',6).replace(['Wd Shng','Stucco','BrkFace','AsbShng','ImStucc','Brk Cmn','Stone','AsphShn','Other','CBlock'],7)
train_df["ExterQual"]=train_df["ExterQual"].replace('TA',1).replace('Gd',2).replace(['Ex','Fa'],3)
train_df["ExterCond"]=train_df["ExterCond"].replace('TA',1).replace('Gd',2).replace(['Fa','Ex','Po'],3)
train_df["Foundation"]=train_df["Foundation"].replace('PConc',1).replace('CBlock',2).replace('BrkTil',3).replace(['Slab','Stone','Wood'],4)
train_df["BsmtQual"]=train_df["BsmtQual"].replace('TA',1).replace('Gd',2).replace('Ex',3).replace('Fa',4)
train_df["BsmtCond"]=train_df["BsmtCond"].replace('TA',1).replace('Gd',2).replace(['Fa','Po'],3)
train_df["BsmtExposure"]=train_df["BsmtExposure"].replace('No',1).replace('Av',2).replace('Gd',3).replace('Mn',4)
train_df["BsmtFinType1"]=train_df["BsmtFinType1"].replace('Unf',1).replace('GLQ',2).replace('ALQ',3).replace('BLQ',4).replace('Rec',5).replace('LwQ',6)
train_df["BsmtFinType2"]=train_df["BsmtFinType2"].replace('Unf',1).replace('GLQ',2).replace('ALQ',3).replace('BLQ',4).replace('Rec',5).replace('LwQ',6)
train_df["Heating"]=train_df["Heating"].replace('GasA',1).replace(['GasW','Grav','Wall','OthW','Floor'],2)
train_df["HeatingQC"]=train_df["HeatingQC"].replace('Ex',1).replace('TA',2).replace('Gd',3).replace(['Fa','Po'],4)
train_df["CentralAir"]=train_df["CentralAir"].replace('Y',1).replace('N',2)
train_df["Electrical"]=train_df["Electrical"].replace('SBrkr',1).replace('FuseA',2).replace(['FuseF','FuseP','Mix'],3)
train_df["KitchenQual"]=train_df["KitchenQual"].replace('TA',1).replace('Gd',2).replace('Ex',3).replace('Fa',4)
train_df["Functional"]=train_df["Functional"].replace('Typ',1).replace(['Min2','Min1','Mod','Maj1','Maj2','Sev'],2)
train_df["GarageType"]=train_df["GarageType"].replace('Attchd',1).replace('Detchd',2).replace('BuiltIn',3).replace(['Basment','CarPort','2Types'],4)
train_df["GarageFinish"]=train_df["GarageFinish"].replace('Unf',1).replace('RFn',2).replace('Fin',3)
train_df["GarageQual"]=train_df["GarageQual"].replace('TA',1).replace(['Fa','Gd','Ex','Po'],2)
train_df["GarageCond"]=train_df["GarageCond"].replace('TA',1).replace(['Fa','Gd','Po','Ex'],2)
train_df["PavedDrive"]=train_df["PavedDrive"].replace('Y',1).replace('N',2).replace('P',3)
train_df["SaleType"]=train_df["SaleType"].replace('WD',1).replace('New',2).replace('COD',3).replace(['ConLD','ConLI','ConLw','CWD','Oth','Con'],4)
train_df["SaleCondition"]=train_df["SaleCondition"].replace('Normal',1).replace('Partial',2).replace('Abnorml',3).replace(['Family','Alloca','AdjLand'],4)
#テストデータにも同様の操作を行う
del test_df["Alley"]
del test_df["PoolArea"]
del test_df["PoolQC"]
del test_df["Fence"]
del test_df["MiscFeature"]
del test_df["MiscVal"]
del test_df["Fireplaces"]
del test_df["FireplaceQu"]
del test_df["MasVnrType"]
del test_df["MasVnrArea"]
test_df["MSZoning"].fillna('RL',inplace=True)
test_df["Utilities"].fillna('AllPub',inplace=True)
test_df["Exterior1st"].fillna('VinylSd',inplace=True)
test_df["Exterior2nd"].fillna('VinylSd',inplace=True)
test_df["KitchenQual"].fillna('TA',inplace=True)
test_df["Functional"].fillna('Typ',inplace=True)
test_df["SaleType"].fillna('WD',inplace=True)
test_df["BsmtQual"].fillna('TA',inplace=True)
test_df["BsmtCond"].fillna('TA',inplace=True)
test_df["BsmtExposure"].fillna('No',inplace=True)
test_df["BsmtFinType1"].fillna('GLQ',inplace=True)
test_df["BsmtFinType2"].fillna('Unf',inplace=True)
test_df["GarageType"].fillna('Attchd',inplace=True)
test_df["GarageFinish"].fillna('Unf',inplace=True)
test_df["GarageQual"].fillna('TA',inplace=True)
test_df["GarageCond"].fillna('TA',inplace=True)
test_df["GarageCars"].fillna(2.0,inplace=True)
test_df["GarageArea"].fillna(0.0,inplace=True)
test_df["LotFrontage"].fillna(test_df["LotFrontage"].median(),inplace=True)
test_df["GarageYrBlt"].fillna(test_df["GarageYrBlt"].median(),inplace=True)
test_df["BsmtFinSF1"].fillna(test_df["BsmtFinSF1"].median(),inplace=True)
test_df["BsmtFinSF2"].fillna(test_df["BsmtFinSF2"].median(),inplace=True)
test_df["BsmtUnfSF"].fillna(test_df["BsmtUnfSF"].median(),inplace=True)
test_df["TotalBsmtSF"].fillna(test_df["TotalBsmtSF"].median(),inplace=True)
test_df["BsmtFullBath"].fillna(test_df["BsmtFullBath"].median(),inplace=True)
test_df["BsmtHalfBath"].fillna(test_df["BsmtHalfBath"].median(),inplace=True)
test_df["MSZoning"]=test_df["MSZoning"].replace("RL",1).replace("RM",2).replace("FV",3).replace(["RH","C (all)"],3)
test_df["LotShape"]=test_df["LotShape"].replace("Reg",1).replace("IR1",2).replace(["IR2","IR3"],3)
test_df["LandContour"]=test_df["LandContour"].replace("Lvl",1).replace("Bnk",2).replace("HLS",3).replace("Low",4)
test_df["Street"]=test_df["Street"].replace("Pave",1).replace("Grvl",2)
test_df["Utilities"]=test_df["Utilities"].replace("AllPub",1)
test_df["LotConfig"]=test_df["LotConfig"].replace("Inside",1).replace("Corner",2).replace("CulDSac",3).replace(["FR2","FR3"],4)
test_df["LandSlope"]=test_df["LandSlope"].replace("Gtl",1).replace(["Mod","Sev"],2)
test_df["Neighborhood"]=test_df["Neighborhood"].replace('NAmes',1).replace('CollgCr',2).replace('OldTown',3).replace('Edwards',4).replace('Somerst',5).replace('Gilbert',6).replace('NridgHt',7).replace('Sawyer',8).replace('NWAmes',9).replace('SawyerW',10).replace('BrkSide',11).replace('Crawfor',12).replace('Mitchel',13).replace('NoRidge',14).replace('Timber',15).replace('IDOTRR',16).replace('ClearCr',17).replace('StoneBr',18).replace('SWISU',19).replace('MeadowV',20).replace('Blmngtn',21).replace('BrDale',22).replace('Veenker',23).replace(['NPkVill','Blueste'],24)
test_df["Condition1"]=test_df["Condition1"].replace('Norm',1).replace('Feedr',2).replace('Artery',3).replace('RRAn',4).replace(['PosN','RRAe','PosA','RRNn','RRNe'],5)
test_df["Condition2"]=test_df["Condition2"].replace('Norm',1).replace(['Feedr','Artery','PosN','PosA'],2)
test_df["BldgType"]=test_df["BldgType"].replace('1Fam',1).replace('TwnhsE',2).replace('Duplex',3).replace('Twnhs',4).replace('2fmCon',5)
test_df["HouseStyle"]=test_df["HouseStyle"].replace('1Story',1).replace('2Story',2).replace('1.5Fin',3).replace('SLvl', 4).replace('SFoyer',5).replace('1.5Unf',6).replace('2.5Unf',7)
test_df["RoofStyle"]=test_df["RoofStyle"].replace('Gable',1).replace('Hip',2).replace(['Flat','Gambrel','Mansard','Shed'],3)
test_df["RoofMatl"]=test_df["RoofMatl"].replace('CompShg',1).replace(['Tar&Grv','WdShngl','WdShake'],2)
test_df["Exterior1st"]=test_df["Exterior1st"].replace('VinylSd',1).replace('HdBoard',2).replace('MetalSd',3).replace('Wd Sdng',4).replace('Plywood',5).replace('CemntBd',6).replace('BrkFace',7).replace(['WdShing','Stucco','AsbShng','BrkComm','AsphShn','CBlock'],8)
test_df["Exterior2nd"]=test_df["Exterior2nd"].replace('VinylSd',1).replace('MetalSd',2).replace('HdBoard',3).replace('Wd Sdng',4).replace('Plywood',5).replace('CmentBd',6).replace(['Wd Shng','Stucco','BrkFace','AsbShng','ImStucc','Brk Cmn','Stone','AsphShn','CBlock'],7)
test_df["ExterQual"]=test_df["ExterQual"].replace('TA',1).replace('Gd',2).replace(['Ex','Fa'],3)
test_df["ExterCond"]=test_df["ExterCond"].replace('TA',1).replace('Gd',2).replace(['Fa','Ex','Po'],3)
test_df["Foundation"]=test_df["Foundation"].replace('PConc',1).replace('CBlock',2).replace('BrkTil',3).replace(['Slab','Stone','Wood'],4)
test_df["BsmtQual"]=test_df["BsmtQual"].replace('TA',1).replace('Gd',2).replace('Ex',3).replace('Fa',4)
test_df["BsmtCond"]=test_df["BsmtCond"].replace('TA',1).replace('Gd',2).replace(['Fa','Po'],3)
test_df["BsmtExposure"]=test_df["BsmtExposure"].replace('No',1).replace('Av',2).replace('Gd',3).replace('Mn',4)
test_df["BsmtFinType1"]=test_df["BsmtFinType1"].replace('Unf',1).replace('GLQ',2).replace('ALQ',3).replace('BLQ',4).replace('Rec',5).replace('LwQ',6)
test_df["BsmtFinType2"]=test_df["BsmtFinType2"].replace('Unf',1).replace('GLQ',2).replace('ALQ',3).replace('BLQ',4).replace('Rec',5).replace('LwQ',6)
test_df["Heating"]=test_df["Heating"].replace('GasA',1).replace(['GasW','Grav','Wall'],2)
test_df["HeatingQC"]=test_df["HeatingQC"].replace('Ex',1).replace('TA',2).replace('Gd',3).replace(['Fa','Po'],4)
test_df["CentralAir"]=test_df["CentralAir"].replace('Y',1).replace('N',2)
test_df["Electrical"]=test_df["Electrical"].replace('SBrkr',1).replace('FuseA',2).replace(['FuseF','FuseP'],3)
test_df["KitchenQual"]=test_df["KitchenQual"].replace('TA',1).replace('Gd',2).replace('Ex',3).replace('Fa',4)
test_df["Functional"]=test_df["Functional"].replace('Typ',1).replace(['Min2','Min1','Mod','Maj1','Maj2','Sev'],2)
test_df["GarageType"]=test_df["GarageType"].replace('Attchd',1).replace('Detchd',2).replace('BuiltIn',3).replace(['Basment','CarPort','2Types'],4)
test_df["GarageFinish"]=test_df["GarageFinish"].replace('Unf',1).replace('RFn',2).replace('Fin',3)
test_df["GarageQual"]=test_df["GarageQual"].replace('TA',1).replace(['Fa','Gd','Po'],2)
test_df["GarageCond"]=test_df["GarageCond"].replace('TA',1).replace(['Fa','Gd','Po','Ex'],2)
test_df["PavedDrive"]=test_df["PavedDrive"].replace('Y',1).replace('N',2).replace('P',3)
test_df["SaleType"]=test_df["SaleType"].replace('WD',1).replace('New',2).replace('COD',3).replace(['ConLD','ConLI','ConLw','CWD','Oth','Con'],4)
test_df["SaleCondition"]=test_df["SaleCondition"].replace('Normal',1).replace('Partial',2).replace('Abnorml',3).replace(['Family','Alloca','AdjLand'],4)
#同項目を結合し結合前のデータを削除
train_df_sale=train_df["SalePrice"]
train_df = train_df.drop(['SalePrice'], axis=1)

train_df["FlrSF"]=train_df["1stFlrSF"]+train_df["2ndFlrSF"]
test_df["FlrSF"]=test_df["1stFlrSF"]+test_df["2ndFlrSF"]
train_df["Bath"]=train_df["BsmtFullBath"]+train_df["BsmtHalfBath"]+train_df["FullBath"]+train_df["HalfBath"]
test_df["Bath"]=test_df["BsmtFullBath"]+test_df["BsmtHalfBath"]+test_df["FullBath"]+test_df["HalfBath"]
train_df["Porch"]=train_df["OpenPorchSF"]+train_df["EnclosedPorch"]+train_df["3SsnPorch"]+train_df["ScreenPorch"]
test_df["Porch"]=test_df["OpenPorchSF"]+test_df["EnclosedPorch"]+test_df["3SsnPorch"]+test_df["ScreenPorch"]

del train_df["BsmtFullBath"]
del train_df["BsmtHalfBath"]
del train_df["FullBath"]
del train_df["HalfBath"]
del test_df["BsmtFullBath"]
del test_df["BsmtHalfBath"]
del test_df["FullBath"]
del test_df["HalfBath"]
del train_df["OpenPorchSF"]
del train_df["EnclosedPorch"]
del train_df["3SsnPorch"]
del train_df["ScreenPorch"]
del test_df["OpenPorchSF"]
del test_df["EnclosedPorch"]
del test_df["3SsnPorch"]
del test_df["ScreenPorch"]

train_df=pd.concat([train_df,train_df_sale],axis=1)
#トレーニングデータとテストデータに切り分け、y_trainの正規化
X_train=train_df.iloc[:,1:-1]
y_train=train_df.iloc[:,-1]
y_train=np.log(y_train)
X_test=test_df.iloc[:,1:]
X_train.shape,y_train.shape,X_test.shape
#表示
sns.distplot(y_train);
#項目の重要度の確認
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=200)
rfr.fit(X_train,y_train)
ranking = np.argsort(-rfr.feature_importances_)
f, ax = plt.subplots(figsize=(11, 9))
sns.barplot(x=rfr.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')
ax.set_xlabel("feature importance")
plt.tight_layout()
plt.show()
#数値化
for i, feat in enumerate(X_train.columns.values[ranking]):
    print('{0:10s} : {1:>.5f}'.format(feat,rfr.feature_importances_[ranking][i]))
#上位30位に絞り込み、top2の項目重要度をあげる
X_train = X_train.iloc[:,ranking[:30]]
X_test = X_test.iloc[:,ranking[:30]]
X_train["Interaction"] = X_train["FlrSF"]*X_train["OverallQual"]
X_test["Interaction"] = X_test["FlrSF"]*X_test["OverallQual"]
#y_trainとの関係性を図示
fig = plt.figure(figsize=(10,20))
for i in np.arange(30):
    ax = fig.add_subplot(10,3,i+1)
    sns.regplot(x=X_train.iloc[:,i], y=y_train)
plt.tight_layout()
plt.show();
#外れ値の削除
X_out=X_train
X_out['SalePrice'] = y_train
X_out=X_out.drop(X_out[(X_out['TotalBsmtSF']>5000) & (X_out['SalePrice']<12.5)].index)
X_out=X_out.drop(X_out[(X_out['GrLivArea']>4000) & (X_out['SalePrice']<13)].index)
X_out=X_out.drop(X_out[(X_out['FlrSF']>4000) & (X_out['SalePrice']<13)].index)
X_out=X_out.drop(X_out[(X_out['GarageArea']>1100) & (X_out['SalePrice']<12.5)].index)
X_out=X_out.drop(X_out[(X_out['BsmtFinSF1']>4000) & (X_out['SalePrice']<12.5)].index)
X_out=X_out.drop(X_out[(X_out['LotFrontage']>300) & (X_out['SalePrice']<13)].index)
X_out=X_out.drop(X_out[(X_out['Porch']>600) & (X_out['SalePrice']<11)].index)
X_out=X_out.drop(X_out[(X_out['1stFlrSF']>4000) & (X_out['SalePrice']<12.5)].index)

y_train = X_out['SalePrice']
X_train = X_out.drop(['SalePrice'], axis=1)
#実装と交差検証を用いた評価
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
gbr=GradientBoostingRegressor(random_state=0,n_estimators=245,learning_rate=0.15)
kfold=KFold(n_splits=10,random_state=0)
scores=cross_val_score(gbr,X_train,y_train,cv=kfold)#,scoring="accuracy")
gbr.fit(X_train,y_train)
gbr.score(X_train,y_train),scores.mean()
#予測値の指数化と表示
pred=np.exp(gbr.predict(X_test))
pred[:10]
#提出用のcsvファイルを作成
submit= pd.DataFrame()
imageid = []
for i in range(len(pred)):
    imageid.append(i+1461)
submit["Id"] = imageid
submit["SalePrice"] = pred
submit.to_csv("result25.csv", index=False)
