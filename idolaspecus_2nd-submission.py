import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

Y = train['SalePrice']

X = train[['MSSubClass','MSZoning', 'LotFrontage', 'LotArea','Street','Alley','LotShape',

           'LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1',

           'Condition2', 'BldgType','HouseStyle','OverallQual','OverallCond','YearBuilt',

           'YearRemodAdd','RoofStyle', 'RoofMatl','Exterior1st','Exterior2nd',

           'MasVnrType','ExterQual','ExterCond','Foundation',

           'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

           'MasVnrArea','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',

           'Heating','HeatingQC','CentralAir','Electrical',

           '1stFlrSF','2ndFlrSF',

           'LowQualFinSF',

           'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',

           'KitchenAbvGr','KitchenQual',

           'TotRmsAbvGrd',

           'MiscVal','Fireplaces','GarageYrBlt','GarageCars','GarageArea','PavedDrive',

           'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',

           'MiscVal','MoSold','YrSold'

           ]]



# 오브젝트 to int

X['MSZoning'] = X['MSZoning'].replace({"C (all)":1,"RM":2,"RH":3,"RL":4,"FV":5 })

X['Street'] = X['Street'].replace({"Grvl":1, "Pave":2})

X['Alley'] = X['Alley'].replace({"Grvl":2, "Pave":1})

X['LotShape'] = X['LotShape'].replace({"Reg":1, "IR1":2, "IR3":3, "IR2":4})

X['LandContour'] = X['LandContour'].replace({"Bnk":1, "Lvl":2, "Low":3, "HLS":4})

X['Utilities'] = X['Utilities'].replace({"AllPub":2, "NoSeWa":1})

X['LotConfig'] = X['LotConfig'].replace({"Inside":1, "FR2":2, "Corner":3, "FR3":4, 'CulDSac':5})

X['LandSlope'] = X['LandSlope'].replace({"Gtl":1, "Mod":2, "Sev":3})

X['BldgType'] = X['BldgType'].replace({"2fmCon":1, "Duplex":2, "Twnhs":3, "TwnhsE":4, "1Fam":4})

X['MasVnrType'] = X['MasVnrType'].replace({"BrkCmn":1, "None":2, "BrkFace":3, "Stone":4})

X['ExterQual'] = X['ExterQual'].replace({"Fa":1, "TA":2, "Gd":3, "Ex":4})

X['ExterCond'] = X['ExterCond'].replace({"Po":1, "Fa":1, "Gd":2, "TA":3, "Ex":4})

X['PavedDrive'] = X['PavedDrive'].replace({"N":1, "P":1, "Y":2})

X['CentralAir'] = X['CentralAir'].replace({"N":1, "Y":2})

X['Electrical'] = X['Electrical'].replace({"Mix":0,"FuseP":1,"FuseF":2,"FuseA":3,"SBrkr":4})

X['KitchenQual'] = X['KitchenQual'].replace({"Fa":1, "TA":2, "Gd":3, "Ex":4})

X['RoofStyle'] = X['RoofStyle'].replace({"Gambrel":1, "Gable":2, "Mansard":3, 

                                         "Flat":4, "Hip":4, 'Shed':5})

X['Foundation'] = X['Foundation'].replace({"Slab":1, "BrkTil":2, "CBlock":3, "Stone":3,

                                           "Wood":3, 'PConc':4})

X['BsmtQual'] = X['BsmtQual'].replace({"Fa":1, "TA":2, "Gd":3, "Ex":4})

X['BsmtCond'] = X['BsmtCond'].replace({"Po":1, "Fa":2, "TA":3, "Gd":4})

X['BsmtExposure'] = X['BsmtExposure'].replace({"No":1, "Mn":2, "Av":3, "Gd":4})

X['BsmtFinType1'] = X['BsmtFinType1'].replace({"Rec":1,"BLQ":1,"LwQ":1,"ALQ":1, "Unf":2, "GLQ":3})

X['BsmtFinType2'] = X['BsmtFinType2'].replace({"Rec":1,"BLQ":1,"LwQ":1,"ALQ":2, "Unf":2, "GLQ":3})

X['Heating'] = X['Heating'].replace({"Floor":1,"Grav":1,"OthW":1,"Wall":1, "GasW":2, "GasA":3})

X['HeatingQC'] = X['HeatingQC'].replace({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})



#결측 처리

X['LotFrontage'] = X['LotFrontage'].fillna(-1)

X['Alley'] = X['Alley'].fillna(-1)

X['Electrical'] = X['Electrical'].fillna(0)

X['MasVnrType'] = X['MasVnrType'].fillna(0)

X['MasVnrArea'] = X['MasVnrArea'].fillna(0)

X['BsmtQual'] = X['BsmtQual'].fillna(-1)

X['BsmtCond'] = X['BsmtCond'].fillna(-1)

X['BsmtExposure'] = X['BsmtExposure'].fillna(-1)

X['BsmtFinType1'] = X['BsmtFinType1'].fillna(-1)

X['BsmtFinType2'] = X['BsmtFinType2'].fillna(-1)

X['GarageYrBlt'] = X['GarageYrBlt'].fillna(-1)



      

       
testX = test[['MSSubClass','MSZoning', 'LotFrontage', 'LotArea','Street','Alley','LotShape',

           'LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1',

           'Condition2', 'BldgType','HouseStyle','OverallQual','OverallCond','YearBuilt',

           'YearRemodAdd','RoofStyle', 'RoofMatl','Exterior1st','Exterior2nd',

           'MasVnrType','ExterQual','ExterCond','Foundation',

           'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',

           'MasVnrArea','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',

           'Heating','HeatingQC','CentralAir','Electrical',

           '1stFlrSF','2ndFlrSF',

           'LowQualFinSF',

           'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr',

           'KitchenAbvGr','KitchenQual',

           'TotRmsAbvGrd',

           'MiscVal','Fireplaces','GarageYrBlt','GarageCars','GarageArea','PavedDrive',

           'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',

           'MiscVal','MoSold','YrSold'

           ]]



# 오브젝트 to int

testX['MSZoning'] = testX['MSZoning'].replace({"C (all)":1,"RM":2,"RH":3,"RL":4,"FV":5 })

testX['Street'] = testX['Street'].replace({"Grvl":1, "Pave":2})

testX['Alley'] = testX['Alley'].replace({"Grvl":2, "Pave":1})

testX['LotShape'] = testX['LotShape'].replace({"Reg":1, "IR1":2, "IR3":3, "IR2":4})

testX['LandContour'] = testX['LandContour'].replace({"Bnk":1, "Lvl":2, "Low":3, "HLS":4})

testX['Utilities'] = testX['Utilities'].replace({"AllPub":2, "NoSeWa":1})

testX['LotConfig'] = testX['LotConfig'].replace({"Inside":1, "FR2":2, "Corner":3, "FR3":4, 'CulDSac':5})

testX['LandSlope'] = testX['LandSlope'].replace({"Gtl":1, "Mod":2, "Sev":3})

testX['BldgType'] = testX['BldgType'].replace({"2fmCon":1, "Duplex":2, "Twnhs":3, "TwnhsE":4, "1Fam":4})

testX['MasVnrType'] = testX['MasVnrType'].replace({"BrkCmn":1, "None":2, "BrkFace":3, "Stone":4})

testX['ExterQual'] = testX['ExterQual'].replace({"Fa":1, "TA":2, "Gd":3, "Ex":4})

testX['ExterCond'] = testX['ExterCond'].replace({"Po":1, "Fa":1, "Gd":2, "TA":3, "Ex":4})

testX['PavedDrive'] = testX['PavedDrive'].replace({"N":1, "P":1, "Y":2})

testX['CentralAir'] = testX['CentralAir'].replace({"N":1, "Y":2})

testX['Electrical'] = testX['Electrical'].replace({"Mix":0,"FuseP":1,"FuseF":2,"FuseA":3,"SBrkr":4})

testX['KitchenQual'] = testX['KitchenQual'].replace({"Fa":1, "TA":2, "Gd":3, "Ex":4})

testX['RoofStyle'] = testX['RoofStyle'].replace({"Gambrel":1, "Gable":2, "Mansard":3,

                                                 "Flat":4, "Hip":4, 'Shed':5})

testX['Foundation'] = testX['Foundation'].replace({"Slab":1, "BrkTil":2, "CBlock":3, "Stone":3,

                                                   "Wood":3, 'PConc':4})

testX['BsmtQual'] = testX['BsmtQual'].replace({"Fa":1, "TA":2, "Gd":3, "Ex":4})

testX['BsmtCond'] = testX['BsmtCond'].replace({"Po":1, "Fa":2, "TA":3, "Gd":4})

testX['BsmtExposure'] = testX['BsmtExposure'].replace({"No":1, "Mn":2, "Av":3, "Gd":4})

testX['BsmtFinType1'] = testX['BsmtFinType1'].replace({"Rec":1,"BLQ":1,"LwQ":1,"ALQ":1, "Unf":2, "GLQ":3})

testX['BsmtFinType2'] = testX['BsmtFinType2'].replace({"Rec":1,"BLQ":1,"LwQ":1,"ALQ":2, "Unf":2, "GLQ":3})

testX['Heating'] = testX['Heating'].replace({"Floor":1,"Grav":1,"OthW":1,"Wall":1, "GasW":2, "GasA":3})

testX['HeatingQC'] = testX['HeatingQC'].replace({"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5})



#결측 처리

testX['LotFrontage'] = testX['LotFrontage'].fillna(-1)

testX['Alley'] = testX['Alley'].fillna(-1)

testX['Electrical'] = testX['Electrical'].fillna(0)

testX['MasVnrType'] = testX['MasVnrType'].fillna(0)

testX['MasVnrArea'] = testX['MasVnrArea'].fillna(0)

testX['Utilities'] = testX['Utilities'].fillna(0)

testX['BsmtQual'] = testX['BsmtQual'].fillna(-1)

testX['BsmtCond'] = testX['BsmtCond'].fillna(-1)

testX['BsmtExposure'] = testX['BsmtExposure'].fillna(-1)

testX['BsmtFinType1'] = testX['BsmtFinType1'].fillna(-1)

testX['BsmtFinType2'] = testX['BsmtFinType2'].fillna(-1)

testX['GarageYrBlt'] = testX['GarageYrBlt'].fillna(-1)



neigh=pd.DataFrame(Y.groupby(train['Neighborhood']).median()).sort_values('SalePrice')

neigh[0:6]=1

neigh[6:11]=2

neigh[11:22]=3

neigh[22:]=4

X['Neighborhood'] = X['Neighborhood'].replace(neigh.to_dict()['SalePrice'])

testX['Neighborhood'] = testX['Neighborhood'].replace(neigh.to_dict()['SalePrice'])



condition1 = pd.DataFrame(Y.groupby(train['Condition1']).median()).sort_values('SalePrice')

condition1[0:3] = 1

condition1[3:6] = 2

condition1[6:] = 3

X['Condition1'] = X['Condition1'].replace(condition1.to_dict()['SalePrice'])

testX['Condition1'] = testX['Condition1'].replace(condition1.to_dict()['SalePrice'])



condition2 = pd.DataFrame(Y.groupby(train['Condition2']).median()).sort_values('SalePrice')

condition1[0:3] = 1

condition1[3:4] = 2

condition1[4:5] = 3

condition1[5:] = 4

X['Condition2'] = X['Condition2'].replace(condition2.to_dict()['SalePrice'])

testX['Condition2'] = testX['Condition2'].replace(condition2.to_dict()['SalePrice'])



house = pd.DataFrame(Y.groupby(train['HouseStyle']).mean()).sort_values('SalePrice')

house[0:4] = 1

house[4:5] = 2

house[5:6] = 3

house[6:7] = 4

house[7:] = 5

X['HouseStyle'] = X['HouseStyle'].replace(house.to_dict()['SalePrice'])

testX['HouseStyle'] = testX['HouseStyle'].replace(house.to_dict()['SalePrice'])





roof = pd.DataFrame(Y.groupby(train['RoofMatl']).std()).sort_values('SalePrice')

roof[0:1] = 2

roof[1:3] = 1

roof[3:4] = 3

roof[4:] = 0

X['RoofMatl'] = X['RoofMatl'].replace(roof.to_dict()['SalePrice'])

testX['RoofMatl'] = testX['RoofMatl'].replace(roof.to_dict()['SalePrice'])





ext1 = pd.DataFrame(Y.groupby(train['Exterior1st']).mean()).sort_values('SalePrice')

ext1[0:1] = 0

ext1[1:4] = 1

ext1[4:8] = 2

ext1[8:11] = 3

ext1[11:] = 4

X['Exterior1st'] = X['Exterior1st'].replace(ext1.to_dict()['SalePrice'])

testX['Exterior1st'] = testX['Exterior1st'].replace(ext1.to_dict()['SalePrice'])





ext2 = pd.DataFrame(Y.groupby(train['Exterior2nd']).mean()).sort_values('SalePrice')

ext2[0:7] = 1

ext2[7:12] = 2

ext2[12:15] = 3

ext2[15:] = 0

X['Exterior2nd'] = X['Exterior2nd'].replace(ext2.to_dict()['SalePrice'])

testX['Exterior2nd'] = testX['Exterior2nd'].replace(ext2.to_dict()['SalePrice'])
testX['Exterior1st'] = testX['Exterior1st'].fillna(testX['Exterior1st'].median())

testX['Exterior2nd'] = testX['Exterior2nd'].fillna(testX['Exterior2nd'].median())

testX['MSZoning'] = testX['MSZoning'].fillna(testX['MSZoning'].median())



         

testX['BsmtFinSF1'] = testX['BsmtFinSF1'].fillna(testX['BsmtFinSF1'].median())

testX['BsmtFinSF2'] = testX['BsmtFinSF2'].fillna(testX['BsmtFinSF2'].median())

testX['BsmtUnfSF'] = testX['BsmtUnfSF'].fillna(testX['BsmtUnfSF'].median())

testX['TotalBsmtSF'] = testX['TotalBsmtSF'].fillna(testX['TotalBsmtSF'].median())

testX['BsmtFullBath'] = testX['BsmtFullBath'].fillna(testX['BsmtFullBath'].median())

testX['BsmtHalfBath'] = testX['BsmtHalfBath'].fillna(testX['BsmtHalfBath'].median())

testX['KitchenQual'] = testX['KitchenQual'].fillna(testX['KitchenQual'].median())

testX['GarageCars'] = testX['GarageCars'].fillna(testX['GarageCars'].median())

testX['GarageArea'] = testX['GarageArea'].fillna(testX['GarageArea'].median())
print(testX['Exterior1st'].value_counts())
print(pd.DataFrame(Y.groupby(X['Exterior1st']).mean()).sort_values('SalePrice'))

print(pd.DataFrame(Y.groupby(X['Exterior1st']).median()).sort_values('SalePrice'))

print(pd.DataFrame(Y.groupby(X['Exterior1st']).std()).sort_values('SalePrice'))

print(X['Exterior1st'].value_counts())
import seaborn as sns

sns.boxplot(Y, X['KitchenQual'])
testX.info()
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=500)

rf.fit(X,Y)

rf.score(X,Y)



rf.predict(testX)