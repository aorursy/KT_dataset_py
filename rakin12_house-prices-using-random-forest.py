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
dataset = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

X = dataset.copy()

y = dataset.iloc[:,-1].values.copy()
test_dataset = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

test_X = test_dataset.copy()
X = X.drop(columns = ["Id","Street","Neighborhood","Heating","SaleType","SalePrice"])

X['LotFrontage'] = X['LotFrontage'].fillna(0)

X['BsmtQual'] = X['BsmtQual'].fillna(0)

X['BsmtCond'] = X['BsmtCond'].fillna(0)

X['BsmtExposure'] = X['BsmtExposure'].fillna(0)

X['BsmtFinType1'] = X['BsmtFinType1'].fillna(0)

X['BsmtFinSF1'] = X['BsmtFinSF1'].fillna(0)

X['GarageFinish'] = X['GarageFinish'].fillna(0)

X['GarageQual'] = X['GarageQual'].fillna(0)

X['GarageCond'] = X['GarageCond'].fillna(0)

X['PoolQC'] = X['PoolQC'].fillna(0)

X['Fence'] = X['Fence'].fillna(0)

X['BsmtFinType2'] = X['BsmtFinType2'].fillna(0)

X['BldgType'] = X['BldgType'].fillna(0)

X['MasVnrType'] = X['MasVnrType'].fillna(0)

X['MasVnrArea'] = X['MasVnrArea'].fillna(0)

X = X.fillna(0)
test_X = test_X.drop(columns = ["Id","Street","Neighborhood","Heating","SaleType"])

test_X['LotFrontage'] = test_X['LotFrontage'].fillna(0)

test_X['BsmtQual'] = test_X['BsmtQual'].fillna(0)

test_X['BsmtCond'] = test_X['BsmtCond'].fillna(0)

test_X['BsmtExposure'] = test_X['BsmtExposure'].fillna(0)

test_X['BsmtFinType1'] = test_X['BsmtFinType1'].fillna(0)

test_X['BsmtFinSF1'] = test_X['BsmtFinSF1'].fillna(0)

test_X['GarageFinish'] = test_X['GarageFinish'].fillna(0)

test_X['GarageQual'] = test_X['GarageQual'].fillna(0)

test_X['GarageCond'] = test_X['GarageCond'].fillna(0)

test_X['PoolQC'] = test_X['PoolQC'].fillna(0)

test_X['Fence'] = test_X['Fence'].fillna(0)

test_X['BsmtFinType2'] = test_X['BsmtFinType2'].fillna(0)

test_X['BldgType'] = test_X['BldgType'].fillna(0)

test_X['MasVnrType'] = test_X['MasVnrType'].fillna(0)

test_X['MasVnrArea'] = test_X['MasVnrArea'].fillna(0)

test_X = test_X.fillna(0)
#mszoning

for i in range(len(X)):

    if(X.loc[i,"MSZoning"] == "A"):

        X.loc[i,"MSZoning"] = 11

    if(X.loc[i,"MSZoning"] == "C (all)"):

        X.loc[i,"MSZoning"] = 22

    if(X.loc[i,"MSZoning"] == "FV"):

        X.loc[i,"MSZoning"] = 33

    if(X.loc[i,"MSZoning"] == "I"):

        X.loc[i,"MSZoning"] = 44

    if(X.loc[i,"MSZoning"] == "RH"):

        X.loc[i,"MSZoning"] = 55

    if(X.loc[i,"MSZoning"] == "RL"):

        X.loc[i,"MSZoning"] = 66

    if(X.loc[i,"MSZoning"] == "RP"):

        X.loc[i,"MSZoning"] = 77

    if(X.loc[i,"MSZoning"] == "RM"):

        X.loc[i,"MSZoning"] = 88



        

for i in range(len(test_X)):

    if(test_X.loc[i,"MSZoning"] == "A"):

        test_X.loc[i,"MSZoning"] = 11

    if(test_X.loc[i,"MSZoning"] == "C (all)"):

        test_X.loc[i,"MSZoning"] = 22

    if(test_X.loc[i,"MSZoning"] == "FV"):

        test_X.loc[i,"MSZoning"] = 33

    if(test_X.loc[i,"MSZoning"] == "I"):

        test_X.loc[i,"MSZoning"] = 44

    if(test_X.loc[i,"MSZoning"] == "RH"):

        test_X.loc[i,"MSZoning"] = 55

    if(test_X.loc[i,"MSZoning"] == "RL"):

        test_X.loc[i,"MSZoning"] = 66

    if(test_X.loc[i,"MSZoning"] == "RP"):

        test_X.loc[i,"MSZoning"] = 77

    if(test_X.loc[i,"MSZoning"] == "RM"):

        test_X.loc[i,"MSZoning"] = 88

        

#lot shape        

for i in range(len(X)):

    if(X.loc[i,"LotShape"] == "IR3"):

        X.loc[i,"LotShape"] = 10

    if(X.loc[i,"LotShape"] == "IR2"):

        X.loc[i,"LotShape"] = 20

    if(X.loc[i,"LotShape"] == "IR1"):

        X.loc[i,"LotShape"] = 30

    if(X.loc[i,"LotShape"] == "Reg"):

        X.loc[i,"LotShape"] = 40

    

for i in range(len(test_X)):

    if(test_X.loc[i,"LotShape"] == "IR3"):

        test_X.loc[i,"LotShape"] = 10

    if(test_X.loc[i,"LotShape"] == "IR2"):

        test_X.loc[i,"LotShape"] = 20

    if(test_X.loc[i,"LotShape"] == "IR1"):

        test_X.loc[i,"LotShape"] = 30

    if(test_X.loc[i,"LotShape"] == "Reg"):

        test_X.loc[i,"LotShape"] = 40

    

    

    

#LandContour     

for i in range(len(X)):

    if(X.loc[i,"LandContour"] == "Lvl"):

        X.loc[i,"LandContour"] = 40

    if(X.loc[i,"LandContour"] == "Bnk"):

        X.loc[i,"LandContour"] = 30

    if(X.loc[i,"LandContour"] == "HLS"):

        X.loc[i,"LandContour"] = 20

    if(X.loc[i,"LandContour"] == "Low"):

        X.loc[i,"LandContour"] = 10

        

    

for i in range(len(test_X)):

    if(test_X.loc[i,"LandContour"] == "Lvl"):

        test_X.loc[i,"LandContour"] = 40

    if(test_X.loc[i,"LandContour"] == "Bnk"):

        test_X.loc[i,"LandContour"] = 30

    if(test_X.loc[i,"LandContour"] == "HLS"):

        test_X.loc[i,"LandContour"] = 20

    if(test_X.loc[i,"LandContour"] == "Low"):

        test_X.loc[i,"LandContour"] = 10

        

                

        



#Utilities     

for i in range(len(X)):

    if(X.loc[i,"Utilities"] == "AllPub"):

        X.loc[i,"Utilities"] = 40

    if(X.loc[i,"Utilities"] == "NoSewr"):

        X.loc[i,"Utilities"] = 30

    if(X.loc[i,"Utilities"] == "NoSeWa"):

        X.loc[i,"Utilities"] = 20

    if(X.loc[i,"Utilities"] == "ELO"):

        X.loc[i,"Utilities"] = 10

        

        

for i in range(len(test_X)):

    if(test_X.loc[i,"Utilities"] == "AllPub"):

        test_X.loc[i,"Utilities"] = 40

    if(test_X.loc[i,"Utilities"] == "NoSewr"):

        test_X.loc[i,"Utilities"] = 30

    if(test_X.loc[i,"Utilities"] == "NoSeWa"):

        test_X.loc[i,"Utilities"] = 20

    if(test_X.loc[i,"Utilities"] == "ELO"):

        test_X.loc[i,"Utilities"] = 10
#LotConfig     

for i in range(len(X)):

    if(X.loc[i,"LotConfig"] == "Inside"):

        X.loc[i,"LotConfig"] = 10

    if(X.loc[i,"LotConfig"] == "Corner"):

        X.loc[i,"LotConfig"] = 20

    if(X.loc[i,"LotConfig"] == "CulDSac"):

        X.loc[i,"LotConfig"] = 30

    if(X.loc[i,"LotConfig"] == "FR2"):

        X.loc[i,"LotConfig"] = 40

    if(X.loc[i,"LotConfig"] == "FR3"):

        X.loc[i,"LotConfig"] = 50



        

for i in range(len(test_X)):

    if(test_X.loc[i,"LotConfig"] == "Inside"):

        test_X.loc[i,"LotConfig"] = 10

    if(test_X.loc[i,"LotConfig"] == "Corner"):

        test_X.loc[i,"LotConfig"] = 20

    if(test_X.loc[i,"LotConfig"] == "CulDSac"):

        test_X.loc[i,"LotConfig"] = 30

    if(test_X.loc[i,"LotConfig"] == "FR2"):

        test_X.loc[i,"LotConfig"] = 40

    if(test_X.loc[i,"LotConfig"] == "FR3"):

        test_X.loc[i,"LotConfig"] = 50

        



#LandSlope     

for i in range(len(X)):

    if(X.loc[i,"LandSlope"] == "Sev"):

        X.loc[i,"LandSlope"] = 10

    if(X.loc[i,"LandSlope"] == "Mod"):

        X.loc[i,"LandSlope"] = 20

    if(X.loc[i,"LandSlope"] == "Gtl"):

        X.loc[i,"LandSlope"] = 30



        

for i in range(len(test_X)):

    if(test_X.loc[i,"LandSlope"] == "Sev"):

        test_X.loc[i,"LandSlope"] = 10

    if(test_X.loc[i,"LandSlope"] == "Mod"):

        test_X.loc[i,"LandSlope"] = 20

    if(test_X.loc[i,"LandSlope"] == "Gtl"):

        test_X.loc[i,"LandSlope"] = 30

        



#BldgType     

for i in range(len(X)):

    if(X.loc[i,"BldgType"] == "1Fam"):

        X.loc[i,"BldgType"] = 10

    if(X.loc[i,"BldgType"] == "2fmCon"):

        X.loc[i,"BldgType"] = 20

    if(X.loc[i,"BldgType"] == "Duplex"):

        X.loc[i,"BldgType"] = 30

    if(X.loc[i,"BldgType"] == "TwnhsE"):

        X.loc[i,"BldgType"] = 40

    if(X.loc[i,"BldgType"] == "Twnhs"):

        X.loc[i,"BldgType"] = 50

        

        

        

for i in range(len(test_X)):

    if(test_X.loc[i,"BldgType"] == "1Fam"):

        test_X.loc[i,"BldgType"] = 10

    if(test_X.loc[i,"BldgType"] == "2fmCon"):

        test_X.loc[i,"BldgType"] = 20

    if(test_X.loc[i,"BldgType"] == "Duplex"):

        test_X.loc[i,"BldgType"] = 30

    if(test_X.loc[i,"BldgType"] == "TwnhsE"):

        test_X.loc[i,"BldgType"] = 40

    if(test_X.loc[i,"BldgType"] == "Twnhs"):

        test_X.loc[i,"BldgType"] = 50

        

        

        

        



#HouseStyle     

for i in range(len(X)):

    if(X.loc[i,"HouseStyle"] == "1Story"):

        X.loc[i,"HouseStyle"] = 10

    if(X.loc[i,"HouseStyle"] == "1.5Fin"):

        X.loc[i,"HouseStyle"] = 20

    if(X.loc[i,"HouseStyle"] == "1.5Unf"):

        X.loc[i,"HouseStyle"] = 30

    if(X.loc[i,"HouseStyle"] == "2Story"):

        X.loc[i,"HouseStyle"] = 40

    if(X.loc[i,"HouseStyle"] == "2.5Fin"):

        X.loc[i,"HouseStyle"] = 50

    if(X.loc[i,"HouseStyle"] == "2.5Unf"):

        X.loc[i,"HouseStyle"] = 60

    if(X.loc[i,"HouseStyle"] == "SFoyer"):

        X.loc[i,"HouseStyle"] = 70

    if(X.loc[i,"HouseStyle"] == "SLvl"):

        X.loc[i,"HouseStyle"] = 80

        

        

        

        

for i in range(len(test_X)):

    if(test_X.loc[i,"HouseStyle"] == "1Story"):

        test_X.loc[i,"HouseStyle"] = 10

    if(test_X.loc[i,"HouseStyle"] == "1.5Fin"):

        test_X.loc[i,"HouseStyle"] = 20

    if(test_X.loc[i,"HouseStyle"] == "1.5Unf"):

        test_X.loc[i,"HouseStyle"] = 30

    if(test_X.loc[i,"HouseStyle"] == "2Story"):

        test_X.loc[i,"HouseStyle"] = 40

    if(test_X.loc[i,"HouseStyle"] == "2.5Fin"):

        test_X.loc[i,"HouseStyle"] = 50

    if(test_X.loc[i,"HouseStyle"] == "2.5Unf"):

        test_X.loc[i,"HouseStyle"] = 60

    if(test_X.loc[i,"HouseStyle"] == "SFoyer"):

        test_X.loc[i,"HouseStyle"] = 70

    if(test_X.loc[i,"HouseStyle"] == "SLvl"):

        test_X.loc[i,"HouseStyle"] = 80

  
#RoofMatl     

for i in range(len(X)):

    if(X.loc[i,"RoofMatl"] == "WdShngl"):

        X.loc[i,"RoofMatl"] = 10

    if(X.loc[i,"RoofMatl"] == "WdShake"):

        X.loc[i,"RoofMatl"] = 20

    if(X.loc[i,"RoofMatl"] == "Tar&Grv"):

        X.loc[i,"RoofMatl"] = 30

    if(X.loc[i,"RoofMatl"] == "Roll"):

        X.loc[i,"RoofMatl"] = 40

    if(X.loc[i,"RoofMatl"] == "Metal"):

        X.loc[i,"RoofMatl"] = 50

    if(X.loc[i,"RoofMatl"] == "Membran"):

        X.loc[i,"RoofMatl"] = 60

    if(X.loc[i,"RoofMatl"] == "CompShg"):

        X.loc[i,"RoofMatl"] = 70

    if(X.loc[i,"RoofMatl"] == "ClyTile"):

        X.loc[i,"RoofMatl"] = 80

        

        

        

for i in range(len(test_X)):

    if(test_X.loc[i,"RoofMatl"] == "WdShngl"):

        test_X.loc[i,"RoofMatl"] = 10

    if(test_X.loc[i,"RoofMatl"] == "WdShake"):

        test_X.loc[i,"RoofMatl"] = 20

    if(test_X.loc[i,"RoofMatl"] == "Tar&Grv"):

        test_X.loc[i,"RoofMatl"] = 30

    if(test_X.loc[i,"RoofMatl"] == "Roll"):

        test_X.loc[i,"RoofMatl"] = 40

    if(test_X.loc[i,"RoofMatl"] == "Metal"):

        test_X.loc[i,"RoofMatl"] = 50

    if(test_X.loc[i,"RoofMatl"] == "Membran"):

        test_X.loc[i,"RoofMatl"] = 60

    if(test_X.loc[i,"RoofMatl"] == "CompShg"):

        test_X.loc[i,"RoofMatl"] = 70

    if(test_X.loc[i,"RoofMatl"] == "ClyTile"):

        test_X.loc[i,"RoofMatl"] = 80

        

        

        

        

#Exterior1st     

for i in range(len(X)):

    if(X.loc[i,"Exterior1st"] == "AsbShng"):

        X.loc[i,"Exterior1st"] = 170

    if(X.loc[i,"Exterior1st"] == "AsphShn"):

        X.loc[i,"Exterior1st"] = 160

    if(X.loc[i,"Exterior1st"] == "BrkComm"):

        X.loc[i,"Exterior1st"] = 150

    if(X.loc[i,"Exterior1st"] == "BrkFace"):

        X.loc[i,"Exterior1st"] = 140

    if(X.loc[i,"Exterior1st"] == "CBlock"):

        X.loc[i,"Exterior1st"] = 130

    if(X.loc[i,"Exterior1st"] == "CemntBd"):

        X.loc[i,"Exterior1st"] = 120

    if(X.loc[i,"Exterior1st"] == "CmentBd"):

        X.loc[i,"Exterior1st"] = 120

    if(X.loc[i,"Exterior1st"] == "CmentBd"):

        X.loc[i,"Exterior1st"] = 120

    if(X.loc[i,"Exterior1st"] == "HdBoard"):

        X.loc[i,"Exterior1st"] = 110

    if(X.loc[i,"Exterior1st"] == "ImStucc"):

        X.loc[i,"Exterior1st"] = 100

    if(X.loc[i,"Exterior1st"] == "MetalSd"):

        X.loc[i,"Exterior1st"] = 90

    if(X.loc[i,"Exterior1st"] == "Other"):

        X.loc[i,"Exterior1st"] = 80

    if(X.loc[i,"Exterior1st"] == "Plywood"):

        X.loc[i,"Exterior1st"] = 70

    if(X.loc[i,"Exterior1st"] == "PreCast"):

        X.loc[i,"Exterior1st"] = 60

    if(X.loc[i,"Exterior1st"] == "Stone"):

        X.loc[i,"Exterior1st"] = 50

    if(X.loc[i,"Exterior1st"] == "Stucco"):

        X.loc[i,"Exterior1st"] = 40

    if(X.loc[i,"Exterior1st"] == "VinylSd"):

        X.loc[i,"Exterior1st"] = 30

    if(X.loc[i,"Exterior1st"] == "Wd Sdng"):

        X.loc[i,"Exterior1st"] = 20

    if(X.loc[i,"Exterior1st"] == "WdShing"):

        X.loc[i,"Exterior1st"] = 10

        

        

        

for i in range(len(test_X)):

    if(test_X.loc[i,"Exterior1st"] == "AsbShng"):

        test_X.loc[i,"Exterior1st"] = 170

    if(test_X.loc[i,"Exterior1st"] == "AsphShn"):

        test_X.loc[i,"Exterior1st"] = 160

    if(test_X.loc[i,"Exterior1st"] == "BrkComm"):

        test_X.loc[i,"Exterior1st"] = 150

    if(test_X.loc[i,"Exterior1st"] == "BrkFace"):

        test_X.loc[i,"Exterior1st"] = 140

    if(test_X.loc[i,"Exterior1st"] == "CBlock"):

        test_X.loc[i,"Exterior1st"] = 130

    if(test_X.loc[i,"Exterior1st"] == "CemntBd"):

        test_X.loc[i,"Exterior1st"] = 120

    if(test_X.loc[i,"Exterior1st"] == "CmentBd"):

        test_X.loc[i,"Exterior1st"] = 120

    if(test_X.loc[i,"Exterior1st"] == "CmentBd"):

        test_X.loc[i,"Exterior1st"] = 120

    if(test_X.loc[i,"Exterior1st"] == "HdBoard"):

        test_X.loc[i,"Exterior1st"] = 110

    if(test_X.loc[i,"Exterior1st"] == "ImStucc"):

        test_X.loc[i,"Exterior1st"] = 100

    if(test_X.loc[i,"Exterior1st"] == "MetalSd"):

        test_X.loc[i,"Exterior1st"] = 90

    if(test_X.loc[i,"Exterior1st"] == "Other"):

        test_X.loc[i,"Exterior1st"] = 80

    if(test_X.loc[i,"Exterior1st"] == "Plywood"):

        test_X.loc[i,"Exterior1st"] = 70

    if(test_X.loc[i,"Exterior1st"] == "PreCast"):

        test_X.loc[i,"Exterior1st"] = 60

    if(test_X.loc[i,"Exterior1st"] == "Stone"):

        test_X.loc[i,"Exterior1st"] = 50

    if(test_X.loc[i,"Exterior1st"] == "Stucco"):

        test_X.loc[i,"Exterior1st"] = 40

    if(test_X.loc[i,"Exterior1st"] == "VinylSd"):

        test_X.loc[i,"Exterior1st"] = 30

    if(test_X.loc[i,"Exterior1st"] == "Wd Sdng"):

        test_X.loc[i,"Exterior1st"] = 20

    if(test_X.loc[i,"Exterior1st"] == "WdShing"):

        test_X.loc[i,"Exterior1st"] = 10

        

        

        

#Exterior2nd     

for i in range(len(X)):

    if(X.loc[i,"Exterior2nd"] == "AsbShng"):

        X.loc[i,"Exterior2nd"] = 170

    if(X.loc[i,"Exterior2nd"] == "AsphShn"):

        X.loc[i,"Exterior2nd"] = 160

    if(X.loc[i,"Exterior2nd"] == "Brk Cmn"):

        X.loc[i,"Exterior2nd"] = 150

    if(X.loc[i,"Exterior2nd"] == "BrkFace"):

        X.loc[i,"Exterior2nd"] = 140

    if(X.loc[i,"Exterior2nd"] == "CBlock"):

        X.loc[i,"Exterior2nd"] = 130

    if(X.loc[i,"Exterior2nd"] == "CemntBd"):

        X.loc[i,"Exterior2nd"] = 120

    if(X.loc[i,"Exterior2nd"] == "CmentBd"):

        X.loc[i,"Exterior2nd"] = 120

    if(X.loc[i,"Exterior2nd"] == "HdBoard"):

        X.loc[i,"Exterior2nd"] = 110

    if(X.loc[i,"Exterior2nd"] == "ImStucc"):

        X.loc[i,"Exterior2nd"] = 100

    if(X.loc[i,"Exterior2nd"] == "MetalSd"):

        X.loc[i,"Exterior2nd"] = 90

    if(X.loc[i,"Exterior2nd"] == "Other"):

        X.loc[i,"Exterior2nd"] = 80

    if(X.loc[i,"Exterior2nd"] == "Plywood"):

        X.loc[i,"Exterior2nd"] = 70

    if(X.loc[i,"Exterior2nd"] == "PreCast"):

        X.loc[i,"Exterior2nd"] = 60

    if(X.loc[i,"Exterior2nd"] == "Stone"):

        X.loc[i,"Exterior2nd"] = 50

    if(X.loc[i,"Exterior2nd"] == "Stucco"):

        X.loc[i,"Exterior2nd"] = 40

    if(X.loc[i,"Exterior2nd"] == "VinylSd"):

        X.loc[i,"Exterior2nd"] = 30

    if(X.loc[i,"Exterior2nd"] == "Wd Shng"):

        X.loc[i,"Exterior2nd"] = 20

    if(X.loc[i,"Exterior2nd"] == "Wd Sdng"):

        X.loc[i,"Exterior2nd"] = 15

    if(X.loc[i,"Exterior2nd"] == "WdShing"):

        X.loc[i,"Exterior2nd"] = 10

        

        

for i in range(len(test_X)):

    if(test_X.loc[i,"Exterior2nd"] == "AsbShng"):

        test_X.loc[i,"Exterior2nd"] = 170

    if(test_X.loc[i,"Exterior2nd"] == "AsphShn"):

        test_X.loc[i,"Exterior2nd"] = 160

    if(test_X.loc[i,"Exterior2nd"] == "Brk Cmn"):

        test_X.loc[i,"Exterior2nd"] = 150

    if(test_X.loc[i,"Exterior2nd"] == "BrkFace"):

        test_X.loc[i,"Exterior2nd"] = 140

    if(test_X.loc[i,"Exterior2nd"] == "CBlock"):

        test_X.loc[i,"Exterior2nd"] = 130

    if(test_X.loc[i,"Exterior2nd"] == "CemntBd"):

        test_X.loc[i,"Exterior2nd"] = 120

    if(test_X.loc[i,"Exterior2nd"] == "CmentBd"):

        test_X.loc[i,"Exterior2nd"] = 120

    if(test_X.loc[i,"Exterior2nd"] == "HdBoard"):

        test_X.loc[i,"Exterior2nd"] = 110

    if(test_X.loc[i,"Exterior2nd"] == "ImStucc"):

        test_X.loc[i,"Exterior2nd"] = 100

    if(test_X.loc[i,"Exterior2nd"] == "MetalSd"):

        test_X.loc[i,"Exterior2nd"] = 90

    if(test_X.loc[i,"Exterior2nd"] == "Other"):

        test_X.loc[i,"Exterior2nd"] = 80

    if(test_X.loc[i,"Exterior2nd"] == "Plywood"):

        test_X.loc[i,"Exterior2nd"] = 70

    if(test_X.loc[i,"Exterior2nd"] == "PreCast"):

        test_X.loc[i,"Exterior2nd"] = 60

    if(test_X.loc[i,"Exterior2nd"] == "Stone"):

        test_X.loc[i,"Exterior2nd"] = 50

    if(test_X.loc[i,"Exterior2nd"] == "Stucco"):

        test_X.loc[i,"Exterior2nd"] = 40

    if(test_X.loc[i,"Exterior2nd"] == "VinylSd"):

        test_X.loc[i,"Exterior2nd"] = 30

    if(test_X.loc[i,"Exterior2nd"] == "Wd Shng"):

        test_X.loc[i,"Exterior2nd"] = 20

    if(test_X.loc[i,"Exterior2nd"] == "Wd Sdng"):

        test_X.loc[i,"Exterior2nd"] = 15

    if(test_X.loc[i,"Exterior2nd"] == "WdShing"):

        test_X.loc[i,"Exterior2nd"] = 10
#MasVnrType     

for i in range(len(X)):

    if(X.loc[i,"MasVnrType"] == "BrkCmn"):

        X.loc[i,"MasVnrType"] = 10

    if(X.loc[i,"MasVnrType"] == "BrkFace"):

        X.loc[i,"MasVnrType"] = 20

    if(X.loc[i,"MasVnrType"] == "CBlock"):

        X.loc[i,"MasVnrType"] = 30

    if(X.loc[i,"MasVnrType"] == "None"):

        X.loc[i,"MasVnrType"] = 40

    if(X.loc[i,"MasVnrType"] == "Stone"):

        X.loc[i,"MasVnrType"] = 50



        



for i in range(len(test_X)):

    if(test_X.loc[i,"MasVnrType"] == "BrkCmn"):

        test_X.loc[i,"MasVnrType"] = 10

    if(test_X.loc[i,"MasVnrType"] == "BrkFace"):

        test_X.loc[i,"MasVnrType"] = 20

    if(test_X.loc[i,"MasVnrType"] == "CBlock"):

        test_X.loc[i,"MasVnrType"] = 30

    if(test_X.loc[i,"MasVnrType"] == "None"):

        test_X.loc[i,"MasVnrType"] = 40

    if(test_X.loc[i,"MasVnrType"] == "Stone"):

        test_X.loc[i,"MasVnrType"] = 50

        

        

        

#ExterQual 

for i in range(len(X)):

    if(X.loc[i,"ExterQual"] == "Po"):

        X.loc[i,"ExterQual"] = 10

    if(X.loc[i,"ExterQual"] == "Fa"):

        X.loc[i,"ExterQual"] = 20

    if(X.loc[i,"ExterQual"] == "TA"):

        X.loc[i,"ExterQual"] = 30

    if(X.loc[i,"ExterQual"] == "Gd"):

        X.loc[i,"ExterQual"] = 40

    if(X.loc[i,"ExterQual"] == "Ex"):

        X.loc[i,"ExterQual"] = 50    

for i in range(len(test_X)):

    if(test_X.loc[i,"ExterQual"] == "Po"):

        test_X.loc[i,"ExterQual"] = 10

    if(test_X.loc[i,"ExterQual"] == "Fa"):

        test_X.loc[i,"ExterQual"] = 20

    if(test_X.loc[i,"ExterQual"] == "TA"):

        test_X.loc[i,"ExterQual"] = 30

    if(test_X.loc[i,"ExterQual"] == "Gd"):

        test_X.loc[i,"ExterQual"] = 40

    if(test_X.loc[i,"ExterQual"] == "Ex"):

        test_X.loc[i,"ExterQual"] = 50

        

        



    

    

    

#ExterCond     

for i in range(len(X)):

    if(X.loc[i,"ExterCond"] == "Po"):

        X.loc[i,"ExterCond"] = 10

    if(X.loc[i,"ExterCond"] == "Fa"):

        X.loc[i,"ExterCond"] = 20

    if(X.loc[i,"ExterCond"] == "TA"):

        X.loc[i,"ExterCond"] = 30

    if(X.loc[i,"ExterCond"] == "Gd"):

        X.loc[i,"ExterCond"] = 40

    if(X.loc[i,"ExterCond"] == "Ex"):

        X.loc[i,"ExterCond"] = 50

        

        

for i in range(len(test_X)):

    if(test_X.loc[i,"ExterCond"] == "Po"):

        test_X.loc[i,"ExterCond"] = 10

    if(test_X.loc[i,"ExterCond"] == "Fa"):

        test_X.loc[i,"ExterCond"] = 20

    if(test_X.loc[i,"ExterCond"] == "TA"):

        test_X.loc[i,"ExterCond"] = 30

    if(test_X.loc[i,"ExterCond"] == "Gd"):

        test_X.loc[i,"ExterCond"] = 40

    if(test_X.loc[i,"ExterCond"] == "Ex"):

        test_X.loc[i,"ExterCond"] = 50

        

        

        

#Foundation     

for i in range(len(X)):

    if(X.loc[i,"Foundation"] == "BrkTil"):

        X.loc[i,"Foundation"] = 60

    if(X.loc[i,"Foundation"] == "CBlock"):

        X.loc[i,"Foundation"] = 50

    if(X.loc[i,"Foundation"] == "PConc"):

        X.loc[i,"Foundation"] = 40

    if(X.loc[i,"Foundation"] == "Slab"):

        X.loc[i,"Foundation"] = 30

    if(X.loc[i,"Foundation"] == "Stone"):

        X.loc[i,"Foundation"] = 20

    if(X.loc[i,"Foundation"] == "Wood"):

        X.loc[i,"Foundation"] = 10

        

        

for i in range(len(test_X)):

    if(test_X.loc[i,"Foundation"] == "BrkTil"):

        test_X.loc[i,"Foundation"] = 60

    if(test_X.loc[i,"Foundation"] == "CBlock"):

        test_X.loc[i,"Foundation"] = 50

    if(test_X.loc[i,"Foundation"] == "PConc"):

        test_X.loc[i,"Foundation"] = 40

    if(test_X.loc[i,"Foundation"] == "Slab"):

        test_X.loc[i,"Foundation"] = 30

    if(test_X.loc[i,"Foundation"] == "Stone"):

        test_X.loc[i,"Foundation"] = 20

    if(test_X.loc[i,"Foundation"] == "Wood"):

        test_X.loc[i,"Foundation"] = 10

        

        

        

        

#BsmtQual     

for i in range(len(X)):

    if(X.loc[i,"BsmtQual"] == "Po"):

        X.loc[i,"BsmtQual"] = 10

    if(X.loc[i,"BsmtQual"] == "Fa"):

        X.loc[i,"BsmtQual"] = 20

    if(X.loc[i,"BsmtQual"] == "TA"):

        X.loc[i,"BsmtQual"] = 30

    if(X.loc[i,"BsmtQual"] == "Gd"):

        X.loc[i,"BsmtQual"] = 40

    if(X.loc[i,"BsmtQual"] == "Ex"):

        X.loc[i,"BsmtQual"] = 50

        

        

for i in range(len(test_X)):

    if(test_X.loc[i,"BsmtQual"] == "Po"):

        test_X.loc[i,"BsmtQual"] = 10

    if(test_X.loc[i,"BsmtQual"] == "Fa"):

        test_X.loc[i,"BsmtQual"] = 20

    if(test_X.loc[i,"BsmtQual"] == "TA"):

        test_X.loc[i,"BsmtQual"] = 30

    if(test_X.loc[i,"BsmtQual"] == "Gd"):

        test_X.loc[i,"BsmtQual"] = 40

    if(test_X.loc[i,"BsmtQual"] == "Ex"):

        test_X.loc[i,"BsmtQual"] = 50 
#BsmtCond     

for i in range(len(X)):

    if(X.loc[i,"BsmtCond"] == "Po"):

        X.loc[i,"BsmtCond"] = 10

    if(X.loc[i,"BsmtCond"] == "Fa"):

        X.loc[i,"BsmtCond"] = 20

    if(X.loc[i,"BsmtCond"] == "TA"):

        X.loc[i,"BsmtCond"] = 30

    if(X.loc[i,"BsmtCond"] == "Gd"):

        X.loc[i,"BsmtCond"] = 40

    if(X.loc[i,"BsmtCond"] == "Ex"):

        X.loc[i,"BsmtCond"] = 50

for i in range(len(test_X)):

    if(test_X.loc[i,"BsmtCond"] == "Po"):

        test_X.loc[i,"BsmtCond"] = 10

    if(test_X.loc[i,"BsmtCond"] == "Fa"):

        test_X.loc[i,"BsmtCond"] = 20

    if(test_X.loc[i,"BsmtCond"] == "TA"):

        test_X.loc[i,"BsmtCond"] = 30

    if(test_X.loc[i,"BsmtCond"] == "Gd"):

        test_X.loc[i,"BsmtCond"] = 40

    if(test_X.loc[i,"BsmtCond"] == "Ex"):

        test_X.loc[i,"BsmtCond"] = 50

        



        

#BsmtExposure     

for i in range(len(X)):

    if(X.loc[i,"BsmtExposure"] == "No"):

        X.loc[i,"BsmtExposure"] = 10

    if(X.loc[i,"BsmtExposure"] == "Mn"):

        X.loc[i,"BsmtExposure"] = 20

    if(X.loc[i,"BsmtExposure"] == "Av"):

        X.loc[i,"BsmtExposure"] = 30

    if(X.loc[i,"BsmtExposure"] == "Gd"):

        X.loc[i,"BsmtExposure"] = 40

for i in range(len(test_X)):

    if(test_X.loc[i,"BsmtExposure"] == "No"):

        test_X.loc[i,"BsmtExposure"] = 10

    if(test_X.loc[i,"BsmtExposure"] == "Mn"):

        test_X.loc[i,"BsmtExposure"] = 20

    if(test_X.loc[i,"BsmtExposure"] == "Av"):

        test_X.loc[i,"BsmtExposure"] = 30

    if(test_X.loc[i,"BsmtExposure"] == "Gd"):

        test_X.loc[i,"BsmtExposure"] = 40

        

        

        

        

#BsmtFinType1     

for i in range(len(X)):

    if(X.loc[i,"BsmtFinType1"] == "GLQ"):

        X.loc[i,"BsmtFinType1"] = 60

    if(X.loc[i,"BsmtFinType1"] == "ALQ"):

        X.loc[i,"BsmtFinType1"] = 50

    if(X.loc[i,"BsmtFinType1"] == "BLQ"):

        X.loc[i,"BsmtFinType1"] = 40

    if(X.loc[i,"BsmtFinType1"] == "Rec"):

        X.loc[i,"BsmtFinType1"] = 30

    if(X.loc[i,"BsmtFinType1"] == "LwQ"):

        X.loc[i,"BsmtFinType1"] = 20

    if(X.loc[i,"BsmtFinType1"] == "Unf"):

        X.loc[i,"BsmtFinType1"] = 10

for i in range(len(test_X)):

    if(test_X.loc[i,"BsmtFinType1"] == "GLQ"):

        test_X.loc[i,"BsmtFinType1"] = 60

    if(test_X.loc[i,"BsmtFinType1"] == "ALQ"):

        test_X.loc[i,"BsmtFinType1"] = 50

    if(test_X.loc[i,"BsmtFinType1"] == "BLQ"):

        test_X.loc[i,"BsmtFinType1"] = 40

    if(test_X.loc[i,"BsmtFinType1"] == "Rec"):

        test_X.loc[i,"BsmtFinType1"] = 30

    if(test_X.loc[i,"BsmtFinType1"] == "LwQ"):

        test_X.loc[i,"BsmtFinType1"] = 20

    if(test_X.loc[i,"BsmtFinType1"] == "Unf"):

        test_X.loc[i,"BsmtFinType1"] = 10

        

        

        

#BsmtFinType2     

for i in range(len(X)):

    if(X.loc[i,"BsmtFinType2"] == "GLQ"):

        X.loc[i,"BsmtFinType2"] = 60

    if(X.loc[i,"BsmtFinType2"] == "ALQ"):

        X.loc[i,"BsmtFinType2"] = 50

    if(X.loc[i,"BsmtFinType2"] == "BLQ"):

        X.loc[i,"BsmtFinType2"] = 40

    if(X.loc[i,"BsmtFinType2"] == "Rec"):

        X.loc[i,"BsmtFinType2"] = 30

    if(X.loc[i,"BsmtFinType2"] == "LwQ"):

        X.loc[i,"BsmtFinType2"] = 20

    if(X.loc[i,"BsmtFinType2"] == "Unf"):

        X.loc[i,"BsmtFinType2"] = 10

for i in range(len(test_X)):

    if(test_X.loc[i,"BsmtFinType2"] == "GLQ"):

        test_X.loc[i,"BsmtFinType2"] = 60

    if(test_X.loc[i,"BsmtFinType2"] == "ALQ"):

        test_X.loc[i,"BsmtFinType2"] = 50

    if(test_X.loc[i,"BsmtFinType2"] == "BLQ"):

        test_X.loc[i,"BsmtFinType2"] = 40

    if(test_X.loc[i,"BsmtFinType2"] == "Rec"):

        test_X.loc[i,"BsmtFinType2"] = 30

    if(test_X.loc[i,"BsmtFinType2"] == "LwQ"):

        test_X.loc[i,"BsmtFinType2"] = 20

    if(test_X.loc[i,"BsmtFinType2"] == "Unf"):

        test_X.loc[i,"BsmtFinType2"] = 10       
#BsmtUnfSF

for i in range(len(X)):

    X.loc[i,"BsmtUnfSF"] = 2500 - X.loc[i,"BsmtUnfSF"]

for i in range(len(test_X)):

    test_X.loc[i,"BsmtUnfSF"] = 2500 - test_X.loc[i,"BsmtUnfSF"]

    

        

#HeatingQC     

for i in range(len(X)):

    if(X.loc[i,"HeatingQC"] == "Po"):

        X.loc[i,"HeatingQC"] = 10

    if(X.loc[i,"HeatingQC"] == "Fa"):

        X.loc[i,"HeatingQC"] = 20

    if(X.loc[i,"HeatingQC"] == "TA"):

        X.loc[i,"HeatingQC"] = 30

    if(X.loc[i,"HeatingQC"] == "Gd"):

        X.loc[i,"HeatingQC"] = 40

    if(X.loc[i,"HeatingQC"] == "Ex"):

        X.loc[i,"HeatingQC"] = 50

for i in range(len(test_X)):

    if(test_X.loc[i,"HeatingQC"] == "Po"):

        test_X.loc[i,"HeatingQC"] = 10

    if(test_X.loc[i,"HeatingQC"] == "Fa"):

        test_X.loc[i,"HeatingQC"] = 20

    if(test_X.loc[i,"HeatingQC"] == "TA"):

        test_X.loc[i,"HeatingQC"] = 30

    if(test_X.loc[i,"HeatingQC"] == "Gd"):

        test_X.loc[i,"HeatingQC"] = 40

    if(test_X.loc[i,"HeatingQC"] == "Ex"):

        test_X.loc[i,"HeatingQC"] = 50

        

        

#CentralAir     

for i in range(len(X)):

    if(X.loc[i,"CentralAir"] == "N"):

        X.loc[i,"CentralAir"] = 0

    else:

        X.loc[i,"CentralAir"] = 1

for i in range(len(test_X)):

    if(test_X.loc[i,"CentralAir"] == "N"):

        test_X.loc[i,"CentralAir"] = 0

    else:

        test_X.loc[i,"CentralAir"] = 1

        

        

#Electrical     

for i in range(len(X)):

    if(X.loc[i,"Electrical"] == "Mix"):

        X.loc[i,"Electrical"] = 10

    if(X.loc[i,"Electrical"] == "FuseP"):

        X.loc[i,"Electrical"] = 20

    if(X.loc[i,"Electrical"] == "FuseF"):

        X.loc[i,"Electrical"] = 30

    if(X.loc[i,"Electrical"] == "FuseA"):

        X.loc[i,"Electrical"] = 40

    if(X.loc[i,"Electrical"] == "SBrkr"):

        X.loc[i,"Electrical"] = 50

for i in range(len(test_X)):

    if(test_X.loc[i,"Electrical"] == "Mix"):

        test_X.loc[i,"Electrical"] = 10

    if(test_X.loc[i,"Electrical"] == "FuseP"):

        test_X.loc[i,"Electrical"] = 20

    if(test_X.loc[i,"Electrical"] == "FuseF"):

        test_X.loc[i,"Electrical"] = 30

    if(test_X.loc[i,"Electrical"] == "FuseA"):

        test_X.loc[i,"Electrical"] = 40

    if(test_X.loc[i,"Electrical"] == "SBrkr"):

        test_X.loc[i,"Electrical"] = 50        

        
#KitchenQual     

for i in range(len(X)):

    if(X.loc[i,"KitchenQual"] == "Po"):

        X.loc[i,"KitchenQual"] = 10

    if(X.loc[i,"KitchenQual"] == "Fa"):

        X.loc[i,"KitchenQual"] = 20

    if(X.loc[i,"KitchenQual"] == "TA"):

        X.loc[i,"KitchenQual"] = 30

    if(X.loc[i,"KitchenQual"] == "Gd"):

        X.loc[i,"KitchenQual"] = 40

    if(X.loc[i,"KitchenQual"] == "Ex"):

        X.loc[i,"KitchenQual"] = 50

for i in range(len(test_X)):

    if(test_X.loc[i,"KitchenQual"] == "Po"):

        test_X.loc[i,"KitchenQual"] = 10

    if(test_X.loc[i,"KitchenQual"] == "Fa"):

        test_X.loc[i,"KitchenQual"] = 20

    if(test_X.loc[i,"KitchenQual"] == "TA"):

        test_X.loc[i,"KitchenQual"] = 30

    if(test_X.loc[i,"KitchenQual"] == "Gd"):

        test_X.loc[i,"KitchenQual"] = 40

    if(test_X.loc[i,"KitchenQual"] == "Ex"):

        test_X.loc[i,"KitchenQual"] = 50        

        

        

#Functional     

for i in range(len(X)):

    if(X.loc[i,"Functional"] == "Sal"):

        X.loc[i,"Functional"] = 10

    if(X.loc[i,"Functional"] == "Sev"):

        X.loc[i,"Functional"] = 20

    if(X.loc[i,"Functional"] == "Maj2"):

        X.loc[i,"Functional"] = 30

    if(X.loc[i,"Functional"] == "Maj1"):

        X.loc[i,"Functional"] = 40

    if(X.loc[i,"Functional"] == "Mod"):

        X.loc[i,"Functional"] = 50

    if(X.loc[i,"Functional"] == "Min2"):

        X.loc[i,"Functional"] = 60

    if(X.loc[i,"Functional"] == "Min1"):

        X.loc[i,"Functional"] = 70

    if(X.loc[i,"Functional"] == "Typ"):

        X.loc[i,"Functional"] = 80

for i in range(len(test_X)):

    if(test_X.loc[i,"Functional"] == "Sal"):

        test_X.loc[i,"Functional"] = 10

    if(test_X.loc[i,"Functional"] == "Sev"):

        test_X.loc[i,"Functional"] = 20

    if(test_X.loc[i,"Functional"] == "Maj2"):

        test_X.loc[i,"Functional"] = 30

    if(test_X.loc[i,"Functional"] == "Maj1"):

        test_X.loc[i,"Functional"] = 40

    if(test_X.loc[i,"Functional"] == "Mod"):

        test_X.loc[i,"Functional"] = 50

    if(test_X.loc[i,"Functional"] == "Min2"):

        test_X.loc[i,"Functional"] = 60

    if(test_X.loc[i,"Functional"] == "Min1"):

        test_X.loc[i,"Functional"] = 70

    if(test_X.loc[i,"Functional"] == "Typ"):

        test_X.loc[i,"Functional"] = 80
#FireplaceQu     

for i in range(len(X)):

    if(X.loc[i,"FireplaceQu"] == "Po"):

        X.loc[i,"FireplaceQu"] = 10

    if(X.loc[i,"FireplaceQu"] == "Fa"):

        X.loc[i,"FireplaceQu"] = 20

    if(X.loc[i,"FireplaceQu"] == "TA"):

        X.loc[i,"FireplaceQu"] = 30

    if(X.loc[i,"FireplaceQu"] == "Gd"):

        X.loc[i,"FireplaceQu"] = 40

    if(X.loc[i,"FireplaceQu"] == "Ex"):

        X.loc[i,"FireplaceQu"] = 50

for i in range(len(test_X)):

    if(test_X.loc[i,"FireplaceQu"] == "Po"):

        test_X.loc[i,"FireplaceQu"] = 10

    if(test_X.loc[i,"FireplaceQu"] == "Fa"):

        test_X.loc[i,"FireplaceQu"] = 20

    if(test_X.loc[i,"FireplaceQu"] == "TA"):

        test_X.loc[i,"FireplaceQu"] = 30

    if(test_X.loc[i,"FireplaceQu"] == "Gd"):

        test_X.loc[i,"FireplaceQu"] = 40

    if(test_X.loc[i,"FireplaceQu"] == "Ex"):

        test_X.loc[i,"FireplaceQu"] = 50        

        

        

        

        

#GarageType     

for i in range(len(X)):

    if(X.loc[i,"GarageType"] == "2Types"):

        X.loc[i,"GarageType"] = 10

    if(X.loc[i,"GarageType"] == "Attchd"):

        X.loc[i,"GarageType"] = 10

    if(X.loc[i,"GarageType"] == "Basment"):

        X.loc[i,"GarageType"] = 10

    if(X.loc[i,"GarageType"] == "BuiltIn"):

        X.loc[i,"GarageType"] = 10

    if(X.loc[i,"GarageType"] == "CarPort"):

        X.loc[i,"GarageType"] = 10

    if(X.loc[i,"GarageType"] == "Detchd"):

        X.loc[i,"GarageType"] = 10

for i in range(len(test_X)):

    if(test_X.loc[i,"GarageType"] == "2Types"):

        test_X.loc[i,"GarageType"] = 10

    if(test_X.loc[i,"GarageType"] == "Attchd"):

        test_X.loc[i,"GarageType"] = 10

    if(test_X.loc[i,"GarageType"] == "Basment"):

        test_X.loc[i,"GarageType"] = 10

    if(test_X.loc[i,"GarageType"] == "BuiltIn"):

        test_X.loc[i,"GarageType"] = 10

    if(test_X.loc[i,"GarageType"] == "CarPort"):

        test_X.loc[i,"GarageType"] = 10

    if(test_X.loc[i,"GarageType"] == "Detchd"):

        test_X.loc[i,"GarageType"] = 10        

        

        

        

#GarageFinish     

for i in range(len(X)):

    if(X.loc[i,"GarageFinish"] == "Unf"):

        X.loc[i,"GarageFinish"] = 10

    if(X.loc[i,"GarageFinish"] == "RFn"):

        X.loc[i,"GarageFinish"] = 20

    if(X.loc[i,"GarageFinish"] == "Fin"):

        X.loc[i,"GarageFinish"] = 30

for i in range(len(test_X)):

    if(test_X.loc[i,"GarageFinish"] == "Unf"):

        test_X.loc[i,"GarageFinish"] = 10

    if(test_X.loc[i,"GarageFinish"] == "RFn"):

        test_X.loc[i,"GarageFinish"] = 20

    if(test_X.loc[i,"GarageFinish"] == "Fin"):

        test_X.loc[i,"GarageFinish"] = 30        

        

        

        

#GarageQual     

for i in range(len(X)):

    if(X.loc[i,"GarageQual"] == "Po"):

        X.loc[i,"GarageQual"] = 10

    if(X.loc[i,"GarageQual"] == "Fa"):

        X.loc[i,"GarageQual"] = 20

    if(X.loc[i,"GarageQual"] == "TA"):

        X.loc[i,"GarageQual"] = 30

    if(X.loc[i,"GarageQual"] == "Gd"):

        X.loc[i,"GarageQual"] = 40

    if(X.loc[i,"GarageQual"] == "Ex"):

        X.loc[i,"GarageQual"] = 50

for i in range(len(test_X)):

    if(test_X.loc[i,"GarageQual"] == "Po"):

        test_X.loc[i,"GarageQual"] = 10

    if(test_X.loc[i,"GarageQual"] == "Fa"):

        test_X.loc[i,"GarageQual"] = 20

    if(test_X.loc[i,"GarageQual"] == "TA"):

        test_X.loc[i,"GarageQual"] = 30

    if(test_X.loc[i,"GarageQual"] == "Gd"):

        test_X.loc[i,"GarageQual"] = 40

    if(test_X.loc[i,"GarageQual"] == "Ex"):

        test_X.loc[i,"GarageQual"] = 50                

        

        

        

#GarageCond     

for i in range(len(X)):

    if(X.loc[i,"GarageCond"] == "Po"):

        X.loc[i,"GarageCond"] = 10

    if(X.loc[i,"GarageCond"] == "Fa"):

        X.loc[i,"GarageCond"] = 20

    if(X.loc[i,"GarageCond"] == "TA"):

        X.loc[i,"GarageCond"] = 30

    if(X.loc[i,"GarageCond"] == "Gd"):

        X.loc[i,"GarageCond"] = 40

    if(X.loc[i,"GarageCond"] == "Ex"):

        X.loc[i,"GarageCond"] = 50

for i in range(len(test_X)):

    if(test_X.loc[i,"GarageCond"] == "Po"):

        test_X.loc[i,"GarageCond"] = 10

    if(test_X.loc[i,"GarageCond"] == "Fa"):

        test_X.loc[i,"GarageCond"] = 20

    if(test_X.loc[i,"GarageCond"] == "TA"):

        test_X.loc[i,"GarageCond"] = 30

    if(test_X.loc[i,"GarageCond"] == "Gd"):

        test_X.loc[i,"GarageCond"] = 40

    if(test_X.loc[i,"GarageCond"] == "Ex"):

        test_X.loc[i,"GarageCond"] = 50                
#PavedDrive     

for i in range(len(X)):

    if(X.loc[i,"PavedDrive"] == "Y"):

        X.loc[i,"PavedDrive"] = 10

    if(X.loc[i,"PavedDrive"] == "P"):

        X.loc[i,"PavedDrive"] = 20

    if(X.loc[i,"PavedDrive"] == "N"):

        X.loc[i,"PavedDrive"] = 30

for i in range(len(test_X)):

    if(test_X.loc[i,"PavedDrive"] == "Y"):

        test_X.loc[i,"PavedDrive"] = 10

    if(test_X.loc[i,"PavedDrive"] == "P"):

        test_X.loc[i,"PavedDrive"] = 20

    if(test_X.loc[i,"PavedDrive"] == "N"):

        test_X.loc[i,"PavedDrive"] = 30







#PoolQC     

for i in range(len(X)):

    if(X.loc[i,"PoolQC"] == "Po"):

        X.loc[i,"PoolQC"] = 10

    if(X.loc[i,"PoolQC"] == "Fa"):

        X.loc[i,"PoolQC"] = 20

    if(X.loc[i,"PoolQC"] == "TA"):

        X.loc[i,"PoolQC"] = 30

    if(X.loc[i,"PoolQC"] == "Gd"):

        X.loc[i,"PoolQC"] = 40

    if(X.loc[i,"PoolQC"] == "Ex"):

        X.loc[i,"PoolQC"] = 50

for i in range(len(test_X)):

    if(test_X.loc[i,"PoolQC"] == "Po"):

        test_X.loc[i,"PoolQC"] = 10

    if(test_X.loc[i,"PoolQC"] == "Fa"):

        test_X.loc[i,"PoolQC"] = 20

    if(test_X.loc[i,"PoolQC"] == "TA"):

        test_X.loc[i,"PoolQC"] = 30

    if(test_X.loc[i,"PoolQC"] == "Gd"):

        test_X.loc[i,"PoolQC"] = 40

    if(test_X.loc[i,"PoolQC"] == "Ex"):

        test_X.loc[i,"PoolQC"] = 50





#Fence     

for i in range(len(X)):

    if(X.loc[i,"Fence"] == "MnWw"):

        X.loc[i,"Fence"] = 40

    if(X.loc[i,"Fence"] == "GdWo"):

        X.loc[i,"Fence"] = 30

    if(X.loc[i,"Fence"] == "MnPrv"):

        X.loc[i,"Fence"] = 20

    if(X.loc[i,"Fence"] == "GdPrv"):

        X.loc[i,"Fence"] = 10

for i in range(len(test_X)):

    if(test_X.loc[i,"Fence"] == "MnWw"):

        test_X.loc[i,"Fence"] = 40

    if(test_X.loc[i,"Fence"] == "GdWo"):

        test_X.loc[i,"Fence"] = 30

    if(test_X.loc[i,"Fence"] == "MnPrv"):

        test_X.loc[i,"Fence"] = 20

    if(test_X.loc[i,"Fence"] == "GdPrv"):

        test_X.loc[i,"Fence"] = 10        

        

        

#SaleCondition     

for i in range(len(X)):

    if(X.loc[i,"SaleCondition"] == "Partial"):

        X.loc[i,"SaleCondition"] = 10

    if(X.loc[i,"SaleCondition"] == "Family"):

        X.loc[i,"SaleCondition"] = 20

    if(X.loc[i,"SaleCondition"] == "Alloca"):

        X.loc[i,"SaleCondition"] = 30

    if(X.loc[i,"SaleCondition"] == "AdjLand"):

        X.loc[i,"SaleCondition"] = 40

    if(X.loc[i,"SaleCondition"] == "Abnorml"):

        X.loc[i,"SaleCondition"] = 50

    if(X.loc[i,"SaleCondition"] == "Normal"):

        X.loc[i,"SaleCondition"] = 60

for i in range(len(test_X)):

    if(test_X.loc[i,"SaleCondition"] == "Partial"):

        test_X.loc[i,"SaleCondition"] = 10

    if(test_X.loc[i,"SaleCondition"] == "Family"):

        test_X.loc[i,"SaleCondition"] = 20

    if(test_X.loc[i,"SaleCondition"] == "Alloca"):

        test_X.loc[i,"SaleCondition"] = 30

    if(test_X.loc[i,"SaleCondition"] == "AdjLand"):

        test_X.loc[i,"SaleCondition"] = 40

    if(test_X.loc[i,"SaleCondition"] == "Abnorml"):

        test_X.loc[i,"SaleCondition"] = 50

    if(test_X.loc[i,"SaleCondition"] == "Normal"):

        test_X.loc[i,"SaleCondition"] = 60

        

        

        

#YearBuilt     

for i in range(len(X)):

    X.loc[i,"YearBuilt"] = 2019 - X.loc[i,"YearBuilt"]

    X.loc[i,"YearBuilt"] = 120 - X.loc[i,"YearBuilt"]

for i in range(len(test_X)):

    test_X.loc[i,"YearBuilt"] = 2019 - test_X.loc[i,"YearBuilt"]

    test_X.loc[i,"YearBuilt"] = 120 - test_X.loc[i,"YearBuilt"]

    

        

#YearRemodAdd     

for i in range(len(X)):

    X.loc[i,"YearRemodAdd"] = 2019 - X.loc[i,"YearRemodAdd"]

    X.loc[i,"YearRemodAdd"] = 120 - X.loc[i,"YearRemodAdd"]

for i in range(len(test_X)):

    test_X.loc[i,"YearRemodAdd"] = 2019 - test_X.loc[i,"YearRemodAdd"]

    test_X.loc[i,"YearRemodAdd"] = 120 - test_X.loc[i,"YearRemodAdd"]
#Alley

dummies = pd.get_dummies(X.Alley)

X= pd.concat([X,dummies],axis='columns')

X = X.drop(columns = ["Alley"])

list(X.columns) 



dummies = pd.get_dummies(test_X.Alley)

test_X= pd.concat([test_X,dummies],axis='columns')

test_X = test_X.drop(columns = ["Alley"])

list(test_X.columns)



#Condition1

dummies = pd.get_dummies(X.Condition1)

X= pd.concat([X,dummies],axis='columns')

X = X.drop(columns = ["Condition1"])

list(X.columns) 



dummies = pd.get_dummies(test_X.Condition1)

test_X= pd.concat([test_X,dummies],axis='columns')

test_X = test_X.drop(columns = ["Condition1"])

list(test_X.columns)







#Condition2

dummies = pd.get_dummies(X.Condition2)

X= pd.concat([X,dummies],axis='columns')

X = X.drop(columns = ["Condition2"])

list(X.columns) 



dummies = pd.get_dummies(test_X.Condition2)

test_X= pd.concat([test_X,dummies],axis='columns')

test_X = test_X.drop(columns = ["Condition2"])

list(test_X.columns) 





#RoofStyle

dummies = pd.get_dummies(X.RoofStyle)

X= pd.concat([X,dummies],axis='columns')

X = X.drop(columns = ["RoofStyle"])

list(X.columns) 



dummies = pd.get_dummies(test_X.RoofStyle)

test_X= pd.concat([test_X,dummies],axis='columns')

test_X = test_X.drop(columns = ["RoofStyle"])

list(test_X.columns)







#MiscFeature

dummies = pd.get_dummies(X.MiscFeature)

X= pd.concat([X,dummies],axis='columns')

X = X.drop(columns = ["MiscFeature"])



dummies = pd.get_dummies(test_X.MiscFeature)

test_X= pd.concat([test_X,dummies],axis='columns')

test_X = test_X.drop(columns = ["MiscFeature"])
missing_cols = X.shape[1] - test_X.shape[1]

for c in range(missing_cols):

    test_X[str(c)+"A"] = 0
#standard feature scalling

from sklearn import preprocessing

normal = preprocessing.Normalizer().fit(X)

X = normal.transform(X)



normal = preprocessing.Normalizer().fit(test_X)

test_X = normal.transform(test_X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.1, random_state=42)

X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier
#Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

acc_log = logreg.score(X_train, y_train)

acc_log
# Support Vector Machines

svc = SVC()

svc.fit(X_train, y_train)

Y_pred = svc.predict(X_test)

acc_svc = svc.score(X_train, y_train)

acc_svc
# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

acc_random_forest = random_forest.score(X_train, y_train)

acc_random_forest
# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(test_X)

acc_random_forest = random_forest.score(X_train, y_train)

acc_random_forest
# Get Correlation Coefficient for each feature using Logistic Regression

coeff_df = pd.DataFrame(dataset.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])



# preview

coeff_df
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Support Vector Machines',

              'Random Forest'],

    'Score': [acc_log, acc_svc,

              acc_random_forest]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "Id": test_dataset["Id"],

        "SalePrice": y_pred

    })

submission.to_csv('sample_submission1.csv', index=False)