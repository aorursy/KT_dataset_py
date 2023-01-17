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
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

y_train = train_data.iloc[:, 80]

X_train = train_data.iloc[:, 1:80]
X_test = test_data.iloc[:, 1:80]

# Store our ID for easy access
Id = test_data.iloc[:, 0]

#Concatenate X_train & X_set for easier feature engineering, hope I remember to split again before fitting into regressor
#X = pd.concat([X_train, X_test])
#Now we need to emgineer our non-numeric-features one by one:

#MSZoning: Identifies the general zoning classification of the sale.
#          I don't know much about pricing according to zones,so it's neutral for me, will use hot encoder.

one_hot_encoded_MSZoning_train = pd.get_dummies(X_train.MSZoning)
X_train.MSZoning = one_hot_encoded_MSZoning_train
one_hot_encoded_MSZoning_test = pd.get_dummies(X_test.MSZoning)
X_test.MSZoning = one_hot_encoded_MSZoning_test
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
#imputer = imputer.fit(X_train[:, 1:2])
#X_train[:, 1:2] = imputer.transform(X_train[:, 1:2])
#Street: Type of road access to property. 
#        I think paved has higher price, so I will assign 1 to 'Grvl' and 2 to 'Pave'. 
X_train.Street.replace(to_replace=dict(Grvl=1, Pave=2), inplace=True)
X_test.Street.replace(to_replace=dict(Grvl=1, Pave=2), inplace=True)

#X.Street.replace({'Grvl': 1,'Pave': 2}, inplace=True)
#Alley: Type of alley access to property. 
#       1 for 'Grvl', and 2 for 'Pave'.
X_train.Alley.replace(to_replace=dict(Grvl=1, Pave=2), inplace=True)
X_train.Alley.fillna(value=0, inplace=True)
X_test.Alley.replace(to_replace=dict(Grvl=1, Pave=2), inplace=True)
X_test.Alley.fillna(value=0, inplace=True)
#LotShape: General shape of property.
#          The highest the regularity the higher the score:
#          1 for 'IR3', 2 for 'IR2', 3 for 'IR1', and 4 for 'Reg'
X_train.LotShape.replace(to_replace=dict(IR3=1, IR2=2, IR1=3, Reg=4), inplace=True)
X_test.LotShape.replace(to_replace=dict(IR3=1, IR2=2, IR1=3, Reg=4), inplace=True)
#LandContour: Flatness of the property.
#             The highest the flatness the higher the score:
#             1 for 'Low', 2 for 'HLS', 3 for 'Bnk', and 4 for 'Lvl'
X_train.LandContour.replace(to_replace=dict(Low=1, HLS=2, Bnk=3, Lvl=4), inplace=True)
X_test.LandContour.replace(to_replace=dict(Low=1, HLS=2, Bnk=3, Lvl=4), inplace=True)

#Utilities: Type of utilities available.
#           The more the available utilities, the higher the rating:
#           1 for 'ELO', 2 for 'NoSeWa', 3 for 'NoSewr', and 4 for 'AllPub'
X_train.Utilities.replace(to_replace=dict(ELO=1, NoSeWa=2, NoSewr=3, AllPub=4), inplace=True)
X_test.Utilities.replace(to_replace=dict(ELO=1, NoSeWa=2, NoSewr=3, AllPub=4), inplace=True)

#LotConfig: Lot configuration.
#           The more the frontage, the higher the rating:
#           1 for 'Inside', 2 for 'Corner', 3 for 'CulDSac', 4 for 'FR2', and 5 for 'FR3'
X_train.LotConfig.replace(to_replace=dict(Inside=1, Corner=2, CulDSac=3, FR2=4, FR3=5), inplace=True)
X_test.LotConfig.replace(to_replace=dict(Inside=1, Corner=2, CulDSac=3, FR2=4, FR3=5), inplace=True)
#LandSlope: Slope of property
#           Will give higher score to less slope
#           1 for 'Sev', 2 for 'Mod', and 3 for 'Gtl'.
X_train.LandSlope.replace(to_replace=dict(Sev=1, Mod=2, Gtl=3), inplace=True)
X_test.LandSlope.replace(to_replace=dict(Sev=1, Mod=2, Gtl=3), inplace=True)

#Neighborhood: Physical locations within Ames city limits.
#              I don't know much about pricing according to these locations,so it's neutral for me, will use hot encoder. 
one_hot_encoded_Neighborhood_train = pd.get_dummies(X_train.Neighborhood)
X_train.Neighborhood = one_hot_encoded_Neighborhood_train
one_hot_encoded_Neighborhood_test = pd.get_dummies(X_test.Neighborhood)
X_test.Neighborhood = one_hot_encoded_Neighborhood_test

#Condition1: Proximity to various conditions
#            I think the closer from main streets the better, also I think being so close from rail road is negative.
#            1 for 'RRAe', 2 for 'RRNe', 3 for 'PosA', 4 for 'PosN', 5 for 'RRAn', 6 for 'RRNn', 7 for 'Norm', 
#            8 for 'Feedr', and 9 for 'Artery'.
X_train.Condition1.replace(to_replace=dict(RRAe=1, RRNe=2, PosA=3, PosN=4, RRAn=5, RRNn=6, Norm=7, Feedr=8, Artery=9), inplace=True)
X_test.Condition1.replace(to_replace=dict(RRAe=1, RRNe=2, PosA=3, PosN=4, RRAn=5, RRNn=6, Norm=7, Feedr=8, Artery=9), inplace=True)
#Condition2: Proximity to various conditions (if more than one is present)
#            I think the closer from main streets the better, also I think being so close from rail road is negative.
#            1 for 'RRAe', 2 for 'RRNe', 3 for 'PosA', 4 for 'PosN', 5 for 'RRAn', 6 for 'RRNn', 7 for 'Norm', 
#            8 for 'Feedr', and 9 for 'Artery'.
X_train.Condition2.replace(to_replace=dict(RRAe=1, RRNe=2, PosA=3, PosN=4, RRAn=5, RRNn=6, Norm=7, Feedr=8, Artery=9), inplace=True)
X_test.Condition2.replace(to_replace=dict(RRAe=1, RRNe=2, PosA=3, PosN=4, RRAn=5, RRNn=6, Norm=7, Feedr=8, Artery=9), inplace=True)

#BldgType: Type of dwelling:
#          I don't know much about pricing according to type of dwelling,so it's neutral for me, will use hot encoder. 
one_hot_encoded_BldgType_train = pd.get_dummies(X_train.BldgType)
X_train.BldgType = one_hot_encoded_BldgType_train
one_hot_encoded_BldgType_test = pd.get_dummies(X_test.BldgType)
X_test.BldgType = one_hot_encoded_BldgType_test
#HouseStyle: Style of dwelling:
#          I don't know much about pricing according to style of dwelling,so it's neutral for me, will use hot encoder.
one_hot_encoded_HouseStyle_train = pd.get_dummies(X_train.HouseStyle)
X_train.HouseStyle = one_hot_encoded_HouseStyle_train
one_hot_encoded_HouseStyle_test = pd.get_dummies(X_test.HouseStyle)
X_test.HouseStyle = one_hot_encoded_HouseStyle_test

# I am thinking about combining 'YearBuilt' and 'YearRemodAdd' to reduce features redundancy.. but dunno how yet.

#RoofStyle: Type of roof:
#           I don't know much about pricing according to roof type,so it's neutral for me, will use hot encoder.
one_hot_encoded_RoofStyle_train = pd.get_dummies(X_train.RoofStyle)
X_train.RoofStyle = one_hot_encoded_RoofStyle_train
one_hot_encoded_RoofStyle_test = pd.get_dummies(X_test.RoofStyle)
X_test.RoofStyle = one_hot_encoded_RoofStyle_test

#RoofMatl: Roof material:
#          I don't know much about pricing according to roof material,so it's neutral for me, will use hot encoder.
one_hot_encoded_RoofMatl_train = pd.get_dummies(X_train.RoofMatl)
X_train.RoofMatl = one_hot_encoded_RoofMatl_train
one_hot_encoded_RoofMatl_test = pd.get_dummies(X_test.RoofMatl)
X_test.RoofMatl = one_hot_encoded_RoofMatl_test

#Exterior1st: Exterior covering on house
#             I don't know much about pricing according to this feature,so it's neutral for me, will use hot encoder.
one_hot_encoded_Exterior1st_train = pd.get_dummies(X_train.Exterior1st)
X_train.Exterior1st = one_hot_encoded_Exterior1st_train
one_hot_encoded_Exterior1st_test = pd.get_dummies(X_test.Exterior1st)
X_test.Exterior1st = one_hot_encoded_Exterior1st_test
#Exterior2nd: Exterior covering on house (if more than one material)
#             I don't know much about pricing according to this feature,so it's neutral for me, will use hot encoder.
one_hot_encoded_Exterior2nd_train = pd.get_dummies(X_train.Exterior2nd)
X_train.Exterior2nd = one_hot_encoded_Exterior2nd_train
one_hot_encoded_Exterior2nd_test = pd.get_dummies(X_test.Exterior2nd)
X_test.Exterior2nd = one_hot_encoded_Exterior2nd_test

#MasVnrType: Masonry veneer type:
#            I don't know much about pricing according to this feature,so it's neutral for me, will use hot encoder.
one_hot_encoded_MasVnrType_train = pd.get_dummies(X_train.MasVnrType)
X_train.MasVnrType = one_hot_encoded_MasVnrType_train
one_hot_encoded_MasVnrType_test = pd.get_dummies(X_test.MasVnrType)
X_test.MasVnrType = one_hot_encoded_MasVnrType_test

#ExterQual: Evaluates the quality of the material on the exterior: 
#           From poor to excellent 
#           1 for 'Po', 2 for 'Fa', 3 for 'TA', 4 for 'Gd', and 5 for 'Ex'.
X_train.ExterQual.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)
X_test.ExterQual.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)

#ExterCond: Evaluates the present condition of the material on the exterior
#           From poor to excellent 
#           1 for 'Po', 2 for 'Fa', 3 for 'TA', 4 for 'Gd', and 5 for 'Ex'.
X_train.ExterCond.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)
X_test.ExterCond.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)

#Foundation: Type of foundation:
#            I don't know much about pricing according to this feature,so it's neutral for me, will use hot encoder.
one_hot_encoded_Foundation_train = pd.get_dummies(X_train.Foundation)
X_train.Foundation = one_hot_encoded_Foundation_train
one_hot_encoded_Foundation_test = pd.get_dummies(X_test.Foundation)
X_test.Foundation = one_hot_encoded_Foundation_test

#BsmtQual: Evaluates the height of the basement
#           From poor to excellent 
#           1 for 'Po', 2 for 'Fa', 3 for 'TA', 4 for 'Gd', and 5 for 'Ex'.
X_train.BsmtQual.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)
X_train.BsmtQual.fillna(value=0, inplace=True)
X_test.BsmtQual.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)
X_test.BsmtQual.fillna(value=0, inplace=True)
#BsmtCond: Evaluates the general condition of the basement:
#          From poor to excellent 
#          1 for 'Po', 2 for 'Fa', 3 for 'TA', 4 for 'Gd', and 5 for 'Ex'.
X_train.BsmtCond.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)
X_train.BsmtCond.fillna(value=0, inplace=True)
X_test.BsmtCond.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)
X_test.BsmtCond.fillna(value=0, inplace=True)
#BsmtExposure: Refers to walkout or garden level walls:
#              From No exposure to Good Exposure 
#              1 for 'No', 2 for 'Mn', 3 for 'Av', and 4 for 'Gd'.
X_train.BsmtExposure.replace(to_replace=dict(No=1, Mn=2, Av=3, Gd=4), inplace=True)
X_train.BsmtExposure.fillna(value=0, inplace=True)
X_test.BsmtExposure.replace(to_replace=dict(No=1, Mn=2, Av=3, Gd=4), inplace=True)
X_test.BsmtExposure.fillna(value=0, inplace=True)
#BsmtFinType1: Rating of basement finished area
#              From unfinished to Good 
#              1 for 'Unf', 2 for 'LwQ', 3 for 'Rec', 4 for 'BLQ', 5 for 'ALQ', and 6 for 'GLQ'.
X_train.BsmtFinType1.replace(to_replace=dict(Unf=1, LwQ=2, Rec=3, BLQ=4, ALQ=5, GLQ=6), inplace=True)
X_train.BsmtFinType1.fillna(value=0, inplace=True)
X_test.BsmtFinType1.replace(to_replace=dict(Unf=1, LwQ=2, Rec=3, BLQ=4, ALQ=5, GLQ=6), inplace=True)
X_test.BsmtFinType1.fillna(value=0, inplace=True)
#BsmtFinType2: Rating of basement finished area (if multiple types):
#              From unfinished to Good 
#              1 for 'Unf', 2 for 'LwQ', 3 for 'Rec', 4 for 'BLQ', 5 for 'ALQ', and 6 for 'GLQ'.
X_train.BsmtFinType2.replace(to_replace=dict(Unf=1, LwQ=2, Rec=3, BLQ=4, ALQ=5, GLQ=6), inplace=True)
X_train.BsmtFinType2.fillna(value=0, inplace=True)
X_test.BsmtFinType2.replace(to_replace=dict(Unf=1, LwQ=2, Rec=3, BLQ=4, ALQ=5, GLQ=6), inplace=True)
X_test.BsmtFinType2.fillna(value=0, inplace=True)
#Heating: Type of heating:
#         I don't know much about pricing according to this feature,so it's neutral for me, will use hot encoder.
one_hot_encoded_Heating_train = pd.get_dummies(X_train.Heating)
X_train.Heating = one_hot_encoded_Heating_train
one_hot_encoded_Heating_test = pd.get_dummies(X_test.Heating)
X_test.Heating = one_hot_encoded_Heating_test

#HeatingQC: Heating quality and condition:
#           From poor to excellent 
#           1 for 'Po', 2 for 'Fa', 3 for 'TA', 4 for 'Gd', and 5 for 'Ex'.
X_train.HeatingQC.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)
X_test.HeatingQC.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)
#CentralAir: Central air conditioning:
#            We have only yes or no .. so will assign 1 or zero
#            0 for 'N', 1 for 'Y'
X_train.CentralAir.replace(to_replace=dict(N=0, Y=1), inplace=True)
X_test.CentralAir.replace(to_replace=dict(N=0, Y=1), inplace=True)

#Electrical: Electrical system:
#            Order from poor to standard, but I am confused about the 'Mix' value ... will put it in middle
#            1 for 'FuseP', 2 for 'FuseF', 3 for 'Mix', 4 for 'FuseA', and 5 for 'SBrkr'.
X_train.Electrical.replace(to_replace=dict(FuseP=1, FuseF=2, Mix=3, FuseA=4, SBrkr=5), inplace=True)
X_test.Electrical.replace(to_replace=dict(FuseP=1, FuseF=2, Mix=3, FuseA=4, SBrkr=5), inplace=True)

#KitchenQual: Kitchen quality:
#             From poor to excellent 
#             1 for 'Po', 2 for 'Fa', 3 for 'TA', 4 for 'Gd', and 5 for 'Ex'.
X_train.KitchenQual.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)
X_test.KitchenQual.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)
#Functional: Home functionality (Assume typical unless deductions are warranted)
#            In imputation will fill empty value with Typical values '8' (Not mean)
#            1 for 'Sal', 2 for 'Sev', 3 for 'Maj2', 4 for 'Maj1', 5 for 'Mod', 6 for 'Min2', 7 for 'Min1', and 8 for 'Typ'.
X_train.Functional.replace(to_replace=dict(Sal=1, Sev=2, Maj2=3, Maj1=4, Mod=5, Min2=6, Min1=7, Typ=8), inplace=True)
X_test.Functional.replace(to_replace=dict(Sal=1, Sev=2, Maj2=3, Maj1=4, Mod=5, Min2=6, Min1=7, Typ=8), inplace=True)

#FireplaceQu: Fireplace quality:
#             From poor to excellent 
#             1 for 'Po', 2 for 'Fa', 3 for 'TA', 4 for 'Gd', and 5 for 'Ex'.
X_train.FireplaceQu.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)
X_train.FireplaceQu.fillna(value=0, inplace=True)
X_test.FireplaceQu.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)
X_test.FireplaceQu.fillna(value=0, inplace=True)
#GarageType: Garage location:
#            I don't know much about pricing according to this feature,so it's neutral for me, will use hot encoder.
one_hot_encoded_GarageType_train = pd.get_dummies(X_train.GarageType)
X_train.GarageType = one_hot_encoded_GarageType_train
one_hot_encoded_GarageType_test = pd.get_dummies(X_test.GarageType)
X_test.GarageType = one_hot_encoded_GarageType_test
#GarageFinish: Interior finish of the garage
#              From unfinished to Finished
#              1 for 'Unf', 2 for 'RFn', and 3 for 'Fin'.
X_train.GarageFinish.replace(to_replace=dict(Unf=1, RFn=2, Fin=3), inplace=True)
X_train.GarageFinish.fillna(value=0, inplace=True)
X_test.GarageFinish.replace(to_replace=dict(Unf=1, RFn=2, Fin=3), inplace=True)
X_test.GarageFinish.fillna(value=0, inplace=True)
#GarageQual: Garage quality:
#            From poor to excellent 
#            1 for 'Po', 2 for 'Fa', 3 for 'TA', 4 for 'Gd', and 5 for 'Ex'.
X_train.GarageQual.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)
X_train.GarageQual.fillna(value=0, inplace=True)
X_test.GarageQual.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)
X_test.GarageQual.fillna(value=0, inplace=True)
#GarageCond: Garage condition:
#            From poor to excellent 
#            1 for 'Po', 2 for 'Fa', 3 for 'TA', 4 for 'Gd', and 5 for 'Ex'.
X_train.GarageCond.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)
X_train.GarageCond.fillna(value=0, inplace=True)
X_test.GarageCond.replace(to_replace=dict(Po=1, Fa=2, TA=3, Gd=4, Ex=5), inplace=True)
X_test.GarageCond.fillna(value=0, inplace=True)
#PavedDrive: Paved driveway
#            gravel < partially paved < paved
#            1 for 'N', 2 for 'P', and 3 for 'Y'.
X_train.PavedDrive.replace(to_replace=dict(N=1, P=2, Y=3), inplace=True)
X_test.PavedDrive.replace(to_replace=dict(N=1, P=2, Y=3), inplace=True)

#PoolQC: Pool quality:
#        From fair to excellent 
#        1 for 'Fa', 2 for 'TA', 3 for 'Gd', and 4 for 'Ex'.
X_train.PoolQC.replace(to_replace=dict(Fa=1, TA=2, Gd=3, Ex=4), inplace=True)
X_train.PoolQC.fillna(value=0, inplace=True)
X_test.PoolQC.replace(to_replace=dict(Fa=1, TA=2, Gd=3, Ex=4), inplace=True)
X_test.PoolQC.fillna(value=0, inplace=True)
#Fence: Fence quality:
#       From min to good 
#       1 for 'MnWw', 2 for 'GdWo', 3 for 'MnPrv', and 4 for 'GdPrv'.
X_train.Fence.replace(to_replace=dict(MnWw=1, GdWo=2, MnPrv=3, GdPrv=4), inplace=True)
X_train.Fence.fillna(value=0, inplace=True)
X_test.Fence.replace(to_replace=dict(MnWw=1, GdWo=2, MnPrv=3, GdPrv=4), inplace=True)
X_test.Fence.fillna(value=0, inplace=True)
#MiscFeature: Miscellaneous feature not covered in other categories:
#             I don't know much about pricing according to this feature,so it's neutral for me, will use hot encoder.

# first, fill NA
X_train.MiscFeature.fillna(value=0, inplace=True)
X_test.MiscFeature.fillna(value=0, inplace=True)

one_hot_encoded_MiscFeature_train = pd.get_dummies(X_train.MiscFeature)
X_train.MiscFeature = one_hot_encoded_MiscFeature_train
one_hot_encoded_MiscFeature_test = pd.get_dummies(X_test.MiscFeature)
X_test.MiscFeature = one_hot_encoded_MiscFeature_test

#SaleType: Type of sale:
#          I don't know much about pricing according to this feature,so it's neutral for me, will use hot encoder.
one_hot_encoded_SaleType_train = pd.get_dummies(X_train.SaleType)
X_train.SaleType = one_hot_encoded_SaleType_train
one_hot_encoded_SaleType_test = pd.get_dummies(X_test.SaleType)
X_test.SaleType = one_hot_encoded_SaleType_test

#SaleCondition: Condition of sale
#          I don't know much about pricing according to this feature,so it's neutral for me, will use hot encoder.
one_hot_encoded_SaleCondition_train = pd.get_dummies(X_train.SaleCondition)
X_train.SaleCondition = one_hot_encoded_SaleCondition_train
one_hot_encoded_SaleCondition_test = pd.get_dummies(X_test.SaleCondition)
X_test.SaleCondition = one_hot_encoded_SaleCondition_test
# Hot Encoding gives 0 in my screen and no new columns ... have to fix it
# handling missing values in column 'LotFrontage', 'GarageYrBlt'
X_train.LotFrontage = X_train.LotFrontage.fillna(X_train.LotFrontage.median())
X_test.LotFrontage = X_test.LotFrontage.fillna(X_test.LotFrontage.median())

X_train.GarageYrBlt = X_train.GarageYrBlt.fillna(X_train.GarageYrBlt.median())
X_test.GarageYrBlt = X_test.GarageYrBlt.fillna(X_test.GarageYrBlt.median())

#Ther's still NA values I cannot reach, will fill it all with zeros
X_train = X_train.fillna(value=0)
X_test = X_test.fillna(value=0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# import and fit regressor to train features and train target
from xgboost import XGBRegressor

regressor = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
regressor.fit(X_train, y_train, verbose=False)
# predict test target
y_pred = pd.DataFrame(regressor.predict(X_test))

#Creating PassengerID column
e =[]
for num in range(1461, 2920):
    e.append(num)
# adding the new column to y_pred DataFrame
y_pred['e'] = e

#adding headers after being deleted during imputation
y_pred.columns=['SalePrice', 'Id']

#swiching colums to the right order to match the needed output formula
y_pred = y_pred[['Id', 'SalePrice']]
y_pred.to_csv('House_Prices_XGBoost.csv', index=False)
