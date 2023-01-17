# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
#matplotlib inline
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

combine = [train, test]
train['GrLivAreaBand'] = pd.cut(dataset['GrLivArea'], 5)
train['TotalBsmtSFBand'] = pd.cut(dataset['TotalBsmtSF'], 5)
map_GarageCond = {'TA': 1, 'Fa': 2, 'Gd': 3, 'Po': 4, 'Ex': 5}

for dataset in combine:
    dataset['GarageCond'] = dataset['GarageCond'].map(map_GarageCond)
map_PavedDrive = {'Y': 1, 'P': 2, 'N': 3}

for dataset in combine:
    dataset['PavedDrive'] = dataset['PavedDrive'].map(map_PavedDrive)
train['WoodDeckSFBand'] = pd.cut(dataset['WoodDeckSF'], 10)

for dataset in combine:
    dataset.loc[dataset['WoodDeckSF'] <= 142, 'WoodDeckSF'] = 0
    dataset.loc[(dataset['WoodDeckSF'] > 142) & (dataset['WoodDeckSF'] <= 284), 'WoodDeckSF'] = 1
    dataset.loc[(dataset['WoodDeckSF'] > 284) & (dataset['WoodDeckSF'] <= 427), 'WoodDeckSF'] = 2
    dataset.loc[(dataset['WoodDeckSF'] > 427) & (dataset['WoodDeckSF'] <= 569), 'WoodDeckSF'] = 3
    dataset.loc[(dataset['WoodDeckSF'] > 569) & (dataset['WoodDeckSF'] <= 712), 'WoodDeckSF'] = 4
    dataset.loc[(dataset['WoodDeckSF'] > 712) & (dataset['WoodDeckSF'] <= 854), 'WoodDeckSF'] = 5
    dataset.loc[(dataset['WoodDeckSF'] > 854) & (dataset['WoodDeckSF'] <= 996), 'WoodDeckSF'] = 6
    dataset.loc[(dataset['WoodDeckSF'] > 996) & (dataset['WoodDeckSF'] <= 1139), 'WoodDeckSF'] = 7
    dataset.loc[(dataset['WoodDeckSF'] > 1139) & (dataset['WoodDeckSF'] <= 1281), 'WoodDeckSF'] = 8
    dataset.loc[(dataset['WoodDeckSF'] > 1281) & (dataset['WoodDeckSF'] <= 1424), 'WoodDeckSF'] = 11
    dataset.loc[dataset['WoodDeckSF'] > 1424, 'WoodDeckSF'] = 10

train['ScreenPorchBand'] = pd.cut(dataset['ScreenPorch'], 5)

for dataset in combine:
    dataset.loc[dataset['ScreenPorch'] <= 115, 'ScreenPorch'] = 0
    dataset.loc[(dataset['ScreenPorch'] > 115) & (dataset['ScreenPorch'] <= 230), 'ScreenPorch'] = 1
    dataset.loc[(dataset['ScreenPorch'] > 230) & (dataset['ScreenPorch'] <= 345), 'ScreenPorch'] = 2
    dataset.loc[(dataset['ScreenPorch'] > 345) & (dataset['ScreenPorch'] <= 460), 'ScreenPorch'] = 3
    dataset.loc[(dataset['ScreenPorch'] > 460) & (dataset['ScreenPorch'] <= 576), 'ScreenPorch'] = 4
    dataset.loc[dataset['ScreenPorch'] > 576, 'ScreenPorch'] = 10
map_SaleType = {'WD': 1, 'New': 2, 'COD': 3, 'ConLD': 4, 'ConLw': 5, 'ConLI': 6, 'CWD': 7, 'Oth': 8, 'Con': 9}

for dataset in combine:
    dataset['SaleType'] = dataset['SaleType'].map(map_SaleType)
for dataset in combine:
    dataset.loc[dataset['GrLivArea'] <= 402, 'GrLivArea'] = 0
    dataset.loc[(dataset['GrLivArea'] > 402) & (dataset['GrLivArea'] <= 1344), 'GrLivArea'] = 1
    dataset.loc[(dataset['GrLivArea'] > 1344) & (dataset['GrLivArea'] <= 2282), 'GrLivArea'] = 2
    dataset.loc[(dataset['GrLivArea'] > 2282) & (dataset['GrLivArea'] <= 3219), 'GrLivArea'] = 3
    dataset.loc[(dataset['GrLivArea'] > 3219) & (dataset['GrLivArea'] <= 4157), 'GrLivArea'] = 4
    dataset.loc[(dataset['GrLivArea'] > 4157) & (dataset['GrLivArea'] <= 5095), 'GrLivArea'] = 5
    dataset.loc[dataset['GrLivArea'] > 5095, 'GrLivArea'] = 10
for dataset in combine:
    dataset.loc[dataset['TotalBsmtSF'] <= 1019, 'TotalBsmtSF'] = 0
    dataset.loc[(dataset['TotalBsmtSF'] > 1019) & (dataset['TotalBsmtSF'] <= 2038), 'TotalBsmtSF'] = 1
    dataset.loc[(dataset['TotalBsmtSF'] > 2038) & (dataset['TotalBsmtSF'] <= 3057), 'TotalBsmtSF'] = 2
    dataset.loc[(dataset['TotalBsmtSF'] > 3057) & (dataset['TotalBsmtSF'] <= 4076), 'TotalBsmtSF'] = 3
    dataset.loc[(dataset['TotalBsmtSF'] > 4076) & (dataset['TotalBsmtSF'] <= 5095), 'TotalBsmtSF'] = 4
    dataset.loc[dataset['TotalBsmtSF'] > 5095, 'TotalBsmtSF'] = 10
train['LotAreaBand'] = pd.cut(dataset['LotArea'], 5)
map_Functional = {'Typ': 1, 'Min2': 2, 'Min1': 3, 'Mod': 4, 'Maj1': 5, 'Maj2': 6, 'Sev': 7}

for dataset in combine:
    dataset['Functional'] = dataset['Functional'].map(map_Functional)
map_GarageFinish = {'Unf': 1, 'RFn': 2, 'Fin': 3}

for dataset in combine:
    dataset['GarageFinish'] = dataset['GarageFinish'].map(map_GarageFinish)
for dataset in combine:
    dataset.loc[dataset['LotArea'] <= 1414, 'LotArea'] = 0
    dataset.loc[(dataset['LotArea'] > 1414) & (dataset['LotArea'] <= 12496), 'LotArea'] = 1
    dataset.loc[(dataset['LotArea'] > 12496) & (dataset['LotArea'] <= 23522), 'LotArea'] = 2
    dataset.loc[(dataset['LotArea'] > 23522) & (dataset['LotArea'] <= 34548), 'LotArea'] = 3
    dataset.loc[(dataset['LotArea'] > 34548) & (dataset['LotArea'] <= 45574), 'LotArea'] = 4
    dataset.loc[(dataset['LotArea'] > 45574) & (dataset['LotArea'] <= 56600), 'LotArea'] = 5
    dataset.loc[dataset['LotArea'] > 56600, 'LotArea'] = 10
train['1stFlrSFBand'] = pd.cut(dataset['1stFlrSF'], 20)
train['1stFlrSF'].value_counts()

map_GarageType = {'Attchd': 1, 'Detchd': 2, 'BuiltIn': 3, 'Basment': 4, 'CarPort': 5, '2Types': 6}

for dataset in combine:
    dataset['GarageType'] = dataset['GarageType'].map(map_GarageType)
train['OpenPorchSFBand'] = pd.cut(dataset['OpenPorchSF'], 10)

for dataset in combine:
    dataset.loc[dataset['OpenPorchSF'] <= 74, 'OpenPorchSF'] = 0
    dataset.loc[(dataset['OpenPorchSF'] > 74) & (dataset['OpenPorchSF'] <= 148), 'OpenPorchSF'] = 1
    dataset.loc[(dataset['OpenPorchSF'] > 148) & (dataset['OpenPorchSF'] <= 222), 'OpenPorchSF'] = 2
    dataset.loc[(dataset['OpenPorchSF'] > 222) & (dataset['OpenPorchSF'] <= 296), 'OpenPorchSF'] = 3
    dataset.loc[(dataset['OpenPorchSF'] > 296) & (dataset['OpenPorchSF'] <= 371), 'OpenPorchSF'] = 4
    dataset.loc[(dataset['OpenPorchSF'] > 371) & (dataset['OpenPorchSF'] <= 445), 'OpenPorchSF'] = 5
    dataset.loc[(dataset['OpenPorchSF'] > 445) & (dataset['OpenPorchSF'] <= 519), 'OpenPorchSF'] = 6
    dataset.loc[(dataset['OpenPorchSF'] > 519) & (dataset['OpenPorchSF'] <= 593), 'OpenPorchSF'] = 7
    dataset.loc[(dataset['OpenPorchSF'] > 593) & (dataset['OpenPorchSF'] <= 667), 'OpenPorchSF'] = 8
    dataset.loc[(dataset['OpenPorchSF'] > 667) & (dataset['OpenPorchSF'] <= 742), 'OpenPorchSF'] = 9
    dataset.loc[dataset['OpenPorchSF'] > 742, 'OpenPorchSF'] = 19
map_HeatingQC = {'Ex': 1, 'TA': 2, 'Gd': 3, 'Fa': 4, 'Po': 5}

for dataset in combine:
    dataset['HeatingQC'] = dataset['HeatingQC'].map(map_HeatingQC)
train['EnclosedPorchBand'] = pd.cut(dataset['EnclosedPorch'], 10)

for dataset in combine:
    dataset.loc[dataset['EnclosedPorch'] <= 101, 'EnclosedPorch'] = 0
    dataset.loc[(dataset['EnclosedPorch'] > 101) & (dataset['EnclosedPorch'] <= 202), 'EnclosedPorch'] = 1
    dataset.loc[(dataset['EnclosedPorch'] > 202) & (dataset['EnclosedPorch'] <= 303), 'EnclosedPorch'] = 2
    dataset.loc[(dataset['EnclosedPorch'] > 303) & (dataset['EnclosedPorch'] <= 404), 'EnclosedPorch'] = 3
    dataset.loc[(dataset['EnclosedPorch'] > 404) & (dataset['EnclosedPorch'] <= 506), 'EnclosedPorch'] = 4
    dataset.loc[(dataset['EnclosedPorch'] > 506) & (dataset['EnclosedPorch'] <= 607), 'EnclosedPorch'] = 5
    dataset.loc[(dataset['EnclosedPorch'] > 607) & (dataset['EnclosedPorch'] <= 708), 'EnclosedPorch'] = 6
    dataset.loc[(dataset['EnclosedPorch'] > 708) & (dataset['EnclosedPorch'] <= 809), 'EnclosedPorch'] = 7
    dataset.loc[(dataset['EnclosedPorch'] > 809) & (dataset['EnclosedPorch'] <= 910), 'EnclosedPorch'] = 8
    dataset.loc[(dataset['EnclosedPorch'] > 910) & (dataset['EnclosedPorch'] <= 1012), 'EnclosedPorch'] = 9
    dataset.loc[dataset['EnclosedPorch'] > 1012, 'EnclosedPorch'] = 19
map_KitchenQual = {'Ex': 1, 'TA': 2, 'Gd': 3, 'Fa': 4}

for dataset in combine:
    dataset['KitchenQual'] = dataset['KitchenQual'].map(map_KitchenQual)
for dataset in combine:
    dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].fillna(0)
    dataset['BsmtFinSF2'] = dataset['BsmtFinSF2'].fillna(0)
    dataset['BsmtUnfSF'] = dataset['BsmtUnfSF'].fillna(0)
    dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(0)
map_Heating = {'GasA': 1, 'GasW': 2, 'Grav': 3, 'Wall': 4, 'OthW': 5, 'Floor': 6}

for dataset in combine:
    dataset['Heating'] = dataset['Heating'].map(map_Heating)


train['GarageAreaBand'] = pd.cut(dataset['GarageArea'], 10)

for dataset in combine:
    dataset.loc[dataset['GarageArea'] <= 148, 'GarageArea'] = 0
    dataset.loc[(dataset['GarageArea'] > 148) & (dataset['GarageArea'] <= 297), 'GarageArea'] = 1
    dataset.loc[(dataset['GarageArea'] > 297) & (dataset['GarageArea'] <= 446), 'GarageArea'] = 2
    dataset.loc[(dataset['GarageArea'] > 446) & (dataset['GarageArea'] <= 595), 'GarageArea'] = 3
    dataset.loc[(dataset['GarageArea'] > 595) & (dataset['GarageArea'] <= 645), 'GarageArea'] = 4
    dataset.loc[(dataset['GarageArea'] > 645) & (dataset['GarageArea'] <= 744), 'GarageArea'] = 5
    dataset.loc[(dataset['GarageArea'] > 744) & (dataset['GarageArea'] <= 892), 'GarageArea'] = 6
    dataset.loc[(dataset['GarageArea'] > 892) & (dataset['GarageArea'] <= 1041), 'GarageArea'] = 7
    dataset.loc[(dataset['GarageArea'] > 1041) & (dataset['GarageArea'] <= 1190), 'GarageArea'] = 8
    dataset.loc[(dataset['GarageArea'] > 1190) & (dataset['GarageArea'] <= 1339), 'GarageArea'] = 11
    dataset.loc[(dataset['GarageArea'] > 1339) & (dataset['GarageArea'] <= 1488), 'GarageArea'] = 9
    dataset.loc[dataset['GarageArea'] > 1488, 'GarageArea'] = 10
map_Electrical = {'SBrkr': 1, 'FuseA': 2, 'FuseF': 3, 'FuseP': 4, 'Mix': 5}

for dataset in combine:
    dataset['Electrical'] = dataset['Electrical'].map(map_Electrical)
map_GarageQual = {'TA': 1, 'Fa': 2, 'Gd': 3, 'Po': 4, 'Ex': 5}

for dataset in combine:
    dataset['GarageQual'] = dataset['GarageQual'].map(map_GarageQual)
map_BsmtFinType2 = {'Unf': 1, 'GLQ': 2, 'ALQ': 3, 'BLQ': 4, 'Rec': 5, 'LwQ': 6}

for dataset in combine:
    dataset['BsmtFinType2'] = dataset['BsmtFinType2'].map(map_BsmtFinType2)
    dataset['BsmtFinType2'] = dataset['BsmtFinType2'].fillna(1)
train['BsmtFinSF1Band'] = pd.cut(dataset['BsmtFinSF1'], 20)


train['2ndFlrSFBand'] = pd.cut(dataset['2ndFlrSF'], 10)

for dataset in combine:
    dataset.loc[dataset['2ndFlrSF'] <= 186, '2ndFlrSF'] = 0
    dataset.loc[(dataset['2ndFlrSF'] > 186) & (dataset['2ndFlrSF'] <= 372), '2ndFlrSF'] = 1
    dataset.loc[(dataset['2ndFlrSF'] > 372) & (dataset['2ndFlrSF'] <= 558), '2ndFlrSF'] = 2
    dataset.loc[(dataset['2ndFlrSF'] > 558) & (dataset['2ndFlrSF'] <= 744), '2ndFlrSF'] = 3
    dataset.loc[(dataset['2ndFlrSF'] > 744) & (dataset['2ndFlrSF'] <= 931), '2ndFlrSF'] = 4
    dataset.loc[(dataset['2ndFlrSF'] > 931) & (dataset['2ndFlrSF'] <= 1117), '2ndFlrSF'] = 5
    dataset.loc[(dataset['2ndFlrSF'] > 1117) & (dataset['2ndFlrSF'] <= 1303), '2ndFlrSF'] = 6
    dataset.loc[(dataset['2ndFlrSF'] > 1303) & (dataset['2ndFlrSF'] <= 1489), '2ndFlrSF'] = 7
    dataset.loc[(dataset['2ndFlrSF'] > 1489) & (dataset['2ndFlrSF'] <= 1675), '2ndFlrSF'] = 8
    dataset.loc[(dataset['2ndFlrSF'] > 1675) & (dataset['2ndFlrSF'] <= 1862), '2ndFlrSF'] = 9
    dataset.loc[dataset['2ndFlrSF'] > 1862, '2ndFlrSF'] = 19
train['BsmtUnfSFBand'] = pd.cut(dataset['BsmtUnfSF'], 20)

map_BsmtFinType1 = {'Unf': 1, 'GLQ': 2, 'ALQ': 3, 'BLQ': 4, 'Rec': 5, 'LwQ': 6}

for dataset in combine:
    dataset['BsmtFinType1'] = dataset['BsmtFinType1'].map(map_BsmtFinType1)
    dataset['BsmtFinType1'] = dataset['BsmtFinType1'].fillna(7)
map_BsmtExposure = {'No': 1, 'Av': 2, 'Gd': 3, 'Mn': 4}

for dataset in combine:
    dataset['BsmtExposure'] = dataset['BsmtExposure'].map(map_BsmtExposure)
    dataset['BsmtExposure'] = dataset['BsmtExposure'].fillna(5)
map_BsmtCond = {'TA': 1, 'Gd': 2, 'Po': 3, 'Fa': 4}

for dataset in combine:
    dataset['BsmtCond'] = dataset['BsmtCond'].map(map_BsmtCond)
    dataset['BsmtCond'] = dataset['BsmtCond'].fillna(5)
for dataset in combine:
    dataset.loc[dataset['BsmtUnfSF'] <= 107, 'BsmtUnfSF'] = 0
    dataset.loc[(dataset['BsmtUnfSF'] > 107) & (dataset['BsmtUnfSF'] <= 214), 'BsmtUnfSF'] = 1
    dataset.loc[(dataset['BsmtUnfSF'] > 214) & (dataset['BsmtUnfSF'] <= 321), 'BsmtUnfSF'] = 2
    dataset.loc[(dataset['BsmtUnfSF'] > 321) & (dataset['BsmtUnfSF'] <= 428), 'BsmtUnfSF'] = 3
    dataset.loc[(dataset['BsmtUnfSF'] > 428) & (dataset['BsmtUnfSF'] <= 535), 'BsmtUnfSF'] = 4
    dataset.loc[(dataset['BsmtUnfSF'] > 535) & (dataset['BsmtUnfSF'] <= 642), 'BsmtUnfSF'] = 5
    dataset.loc[(dataset['BsmtUnfSF'] > 642) & (dataset['BsmtUnfSF'] <= 749), 'BsmtUnfSF'] = 6
    dataset.loc[(dataset['BsmtUnfSF'] > 749) & (dataset['BsmtUnfSF'] <= 856), 'BsmtUnfSF'] = 7
    dataset.loc[(dataset['BsmtUnfSF'] > 856) & (dataset['BsmtUnfSF'] <= 963), 'BsmtUnfSF'] = 8
    dataset.loc[(dataset['BsmtUnfSF'] > 963) & (dataset['BsmtUnfSF'] <= 1070), 'BsmtUnfSF'] = 9
    dataset.loc[(dataset['BsmtUnfSF'] > 1070) & (dataset['BsmtUnfSF'] <= 1177), 'BsmtUnfSF'] = 10
    dataset.loc[(dataset['BsmtUnfSF'] > 1177) & (dataset['BsmtUnfSF'] <= 1284), 'BsmtUnfSF'] = 11
    dataset.loc[(dataset['BsmtUnfSF'] > 1284) & (dataset['BsmtUnfSF'] <= 1391), 'BsmtUnfSF'] = 12
    dataset.loc[(dataset['BsmtUnfSF'] > 1391) & (dataset['BsmtUnfSF'] <= 1498), 'BsmtUnfSF'] = 13
    dataset.loc[(dataset['BsmtUnfSF'] > 1498) & (dataset['BsmtUnfSF'] <= 1498), 'BsmtUnfSF'] = 14
    dataset.loc[(dataset['BsmtUnfSF'] > 1498) & (dataset['BsmtUnfSF'] <= 1605), 'BsmtUnfSF'] = 15
    dataset.loc[(dataset['BsmtUnfSF'] > 1605) & (dataset['BsmtUnfSF'] <= 1712), 'BsmtUnfSF'] = 16
    dataset.loc[(dataset['BsmtUnfSF'] > 1712) & (dataset['BsmtUnfSF'] <= 1819), 'BsmtUnfSF'] = 17
    dataset.loc[(dataset['BsmtUnfSF'] > 1819) & (dataset['BsmtUnfSF'] <= 1926), 'BsmtUnfSF'] = 18
    dataset.loc[(dataset['BsmtUnfSF'] > 1926) & (dataset['BsmtUnfSF'] <= 2033), 'BsmtUnfSF'] = 20
    dataset.loc[dataset['BsmtUnfSF'] > 2033, 'BsmtUnfSF'] = 19
map_BsmtQual = {'TA': 1, 'Gd': 2, 'Ex': 3, 'Fa': 4}

for dataset in combine:
    dataset['BsmtQual'] = dataset['BsmtQual'].map(map_BsmtQual)
    dataset['BsmtQual'] = dataset['BsmtQual'].fillna(5)
map_CentralAir = {'Y': 1, 'N': 2}

for dataset in combine:
    dataset['CentralAir'] = dataset['CentralAir'].map(map_CentralAir)

for dataset in combine:
    dataset.loc[dataset['1stFlrSF'] <= 641, '1stFlrSF'] = 0
    dataset.loc[(dataset['1stFlrSF'] > 641) & (dataset['1stFlrSF'] <= 875), '1stFlrSF'] = 1
    dataset.loc[(dataset['1stFlrSF'] > 875) & (dataset['1stFlrSF'] <= 1110), '1stFlrSF'] = 2
    dataset.loc[(dataset['1stFlrSF'] > 1110) & (dataset['1stFlrSF'] <= 1344), '1stFlrSF'] = 3
    dataset.loc[(dataset['1stFlrSF'] > 1344) & (dataset['1stFlrSF'] <= 1579), '1stFlrSF'] = 4
    dataset.loc[(dataset['1stFlrSF'] > 1579) & (dataset['1stFlrSF'] <= 1813), '1stFlrSF'] = 5
    dataset.loc[(dataset['1stFlrSF'] > 1813) & (dataset['1stFlrSF'] <= 2047), '1stFlrSF'] = 6
    dataset.loc[(dataset['1stFlrSF'] > 2047) & (dataset['1stFlrSF'] <= 2282), '1stFlrSF'] = 7
    dataset.loc[(dataset['1stFlrSF'] > 2282) & (dataset['1stFlrSF'] <= 2516), '1stFlrSF'] = 8
    dataset.loc[(dataset['1stFlrSF'] > 2516) & (dataset['1stFlrSF'] <= 2751), '1stFlrSF'] = 9
    dataset.loc[(dataset['1stFlrSF'] > 2751) & (dataset['1stFlrSF'] <= 2985), '1stFlrSF'] = 10
    dataset.loc[(dataset['1stFlrSF'] > 2985) & (dataset['1stFlrSF'] <= 3219), '1stFlrSF'] = 11
    dataset.loc[(dataset['1stFlrSF'] > 3219) & (dataset['1stFlrSF'] <= 3454), '1stFlrSF'] = 12
    dataset.loc[(dataset['1stFlrSF'] > 3454) & (dataset['1stFlrSF'] <= 3688), '1stFlrSF'] = 13
    dataset.loc[(dataset['1stFlrSF'] > 3688) & (dataset['1stFlrSF'] <= 3923), '1stFlrSF'] = 14
    dataset.loc[(dataset['1stFlrSF'] > 3923) & (dataset['1stFlrSF'] <= 4157), '1stFlrSF'] = 15
    dataset.loc[(dataset['1stFlrSF'] > 4157) & (dataset['1stFlrSF'] <= 4391), '1stFlrSF'] = 16
    dataset.loc[(dataset['1stFlrSF'] > 4391) & (dataset['1stFlrSF'] <= 4626), '1stFlrSF'] = 17
    dataset.loc[(dataset['1stFlrSF'] > 4626) & (dataset['1stFlrSF'] <= 4860), '1stFlrSF'] = 18
    dataset.loc[(dataset['1stFlrSF'] > 4860) & (dataset['1stFlrSF'] <= 5095), '1stFlrSF'] = 20
    dataset.loc[dataset['1stFlrSF'] > 5095, '1stFlrSF'] = 19
map_Foundation = {'PConc': 1, 'CBlock': 2, 'BrkTil': 3, 'Slab': 4, 'Stone': 5, 'Wood': 6}

for dataset in combine:
    dataset['Foundation'] = dataset['Foundation'].map(map_Foundation)
map_ExterCond = {'TA': 1, 'Gd': 2, 'Ex': 3, 'Fa': 4, 'Po': 5}

for dataset in combine:
    dataset['ExterCond'] = dataset['ExterCond'].map(map_ExterCond)


for dataset in combine:
    dataset.loc[dataset['BsmtFinSF1'] <= 200, 'BsmtFinSF1'] = 0
    dataset.loc[(dataset['BsmtFinSF1'] > 200) & (dataset['BsmtFinSF1'] <= 401), 'BsmtFinSF1'] = 1
    dataset.loc[(dataset['BsmtFinSF1'] > 401) & (dataset['BsmtFinSF1'] <= 601), 'BsmtFinSF1'] = 2
    dataset.loc[(dataset['BsmtFinSF1'] > 601) & (dataset['BsmtFinSF1'] <= 802), 'BsmtFinSF1'] = 3
    dataset.loc[(dataset['BsmtFinSF1'] > 802) & (dataset['BsmtFinSF1'] <= 1002), 'BsmtFinSF1'] = 4
    dataset.loc[(dataset['BsmtFinSF1'] > 1002) & (dataset['BsmtFinSF1'] <= 1203), 'BsmtFinSF1'] = 5
    dataset.loc[(dataset['BsmtFinSF1'] > 1203) & (dataset['BsmtFinSF1'] <= 1403), 'BsmtFinSF1'] = 6
    dataset.loc[(dataset['BsmtFinSF1'] > 1403) & (dataset['BsmtFinSF1'] <= 1604), 'BsmtFinSF1'] = 7
    dataset.loc[(dataset['BsmtFinSF1'] > 1604) & (dataset['BsmtFinSF1'] <= 1804), 'BsmtFinSF1'] = 8
    dataset.loc[(dataset['BsmtFinSF1'] > 1804) & (dataset['BsmtFinSF1'] <= 2005), 'BsmtFinSF1'] = 9
    dataset.loc[(dataset['BsmtFinSF1'] > 2005) & (dataset['BsmtFinSF1'] <= 2406), 'BsmtFinSF1'] = 10
    dataset.loc[(dataset['BsmtFinSF1'] > 2406) & (dataset['BsmtFinSF1'] <= 2606), 'BsmtFinSF1'] = 11
    dataset.loc[(dataset['BsmtFinSF1'] > 2606) & (dataset['BsmtFinSF1'] <= 2807), 'BsmtFinSF1'] = 12
    dataset.loc[(dataset['BsmtFinSF1'] > 2807) & (dataset['BsmtFinSF1'] <= 3007), 'BsmtFinSF1'] = 13
    dataset.loc[(dataset['BsmtFinSF1'] > 3007) & (dataset['BsmtFinSF1'] <= 3208), 'BsmtFinSF1'] = 14
    dataset.loc[(dataset['BsmtFinSF1'] > 3208) & (dataset['BsmtFinSF1'] <= 3408), 'BsmtFinSF1'] = 15
    dataset.loc[(dataset['BsmtFinSF1'] > 3408) & (dataset['BsmtFinSF1'] <= 3609), 'BsmtFinSF1'] = 16
    dataset.loc[(dataset['BsmtFinSF1'] > 3609) & (dataset['BsmtFinSF1'] <= 3809), 'BsmtFinSF1'] = 17
    dataset.loc[(dataset['BsmtFinSF1'] > 3809) & (dataset['BsmtFinSF1'] <= 4010), 'BsmtFinSF1'] = 18
    dataset.loc[dataset['BsmtFinSF1'] > 4010, 'BsmtFinSF1'] = 19
map_ExterQual = {'TA': 1, 'Gd': 2, 'Ex': 3, 'Fa': 4}

for dataset in combine:
    dataset['ExterQual'] = dataset['ExterQual'].map(map_ExterQual)

train['MasVnrAreaBand'] = pd.cut(dataset['MasVnrArea'], 10)


for dataset in combine:
    dataset.loc[dataset['MasVnrArea'] <= 129, 'MasVnrArea'] = 0
    dataset.loc[(dataset['MasVnrArea'] > 129) & (dataset['MasVnrArea'] <= 258), 'MasVnrArea'] = 1
    dataset.loc[(dataset['MasVnrArea'] > 258) & (dataset['MasVnrArea'] <= 387), 'MasVnrArea'] = 2
    dataset.loc[(dataset['MasVnrArea'] > 387) & (dataset['MasVnrArea'] <= 516), 'MasVnrArea'] = 3
    dataset.loc[(dataset['MasVnrArea'] > 516) & (dataset['MasVnrArea'] <= 645), 'MasVnrArea'] = 4
    dataset.loc[(dataset['MasVnrArea'] > 645) & (dataset['MasVnrArea'] <= 774), 'MasVnrArea'] = 5
    dataset.loc[(dataset['MasVnrArea'] > 774) & (dataset['MasVnrArea'] <= 903), 'MasVnrArea'] = 6
    dataset.loc[(dataset['MasVnrArea'] > 903) & (dataset['MasVnrArea'] <= 1032), 'MasVnrArea'] = 7
    dataset.loc[(dataset['MasVnrArea'] > 1032) & (dataset['MasVnrArea'] <= 1161), 'MasVnrArea'] = 8
    dataset.loc[(dataset['MasVnrArea'] > 1161) & (dataset['MasVnrArea'] <= 1290), 'MasVnrArea'] = 9
    dataset.loc[dataset['MasVnrArea'] > 1290, 'MasVnrArea'] = 10
    dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(0)
map_MasVnrType = {'None': 1, 'BrkFace': 2, 'Stone': 3, 'BrkCmn': 4}

for dataset in combine:
    dataset['MasVnrType'] = dataset['MasVnrType'].map(map_MasVnrType)
    dataset['MasVnrType'] = dataset['MasVnrType'].fillna(1)
train['GarageYrBltBand'] = pd.cut(dataset['GarageYrBlt'], 10)


for dataset in combine:
    dataset.loc[dataset['GarageYrBlt'] <= 1894, 'GarageYrBlt'] = 0
    dataset.loc[(dataset['GarageYrBlt'] > 1894) & (dataset['GarageYrBlt'] <= 1926), 'GarageYrBlt'] = 1
    dataset.loc[(dataset['GarageYrBlt'] > 1926) & (dataset['GarageYrBlt'] <= 1957), 'GarageYrBlt'] = 2
    dataset.loc[(dataset['GarageYrBlt'] > 1957) & (dataset['GarageYrBlt'] <= 1988), 'GarageYrBlt'] = 3
    dataset.loc[(dataset['GarageYrBlt'] > 1988) & (dataset['GarageYrBlt'] <= 2019), 'GarageYrBlt'] = 4
    dataset.loc[(dataset['GarageYrBlt'] > 2019) & (dataset['GarageYrBlt'] <= 2051), 'GarageYrBlt'] = 5
    dataset.loc[(dataset['GarageYrBlt'] > 2051) & (dataset['GarageYrBlt'] <= 2082), 'GarageYrBlt'] = 6
    dataset.loc[(dataset['GarageYrBlt'] > 2082) & (dataset['GarageYrBlt'] <= 2113), 'GarageYrBlt'] = 7
    dataset.loc[(dataset['GarageYrBlt'] > 2113) & (dataset['GarageYrBlt'] <= 2144), 'GarageYrBlt'] = 8
    dataset.loc[(dataset['GarageYrBlt'] > 2144) & (dataset['GarageYrBlt'] <= 2175), 'GarageYrBlt'] = 9
    dataset.loc[(dataset['GarageYrBlt'] > 2175) & (dataset['GarageYrBlt'] <= 2207), 'GarageYrBlt'] = 10
    dataset.loc[dataset['GarageYrBlt'] > 2207, 'GarageYrBlt'] = 19
map_Exterior2nd = {'VinylSd': 1, 'HdBoard': 2, 'MetalSd': 3, 'Wd Sdng': 4, 'Plywood': 5, 'CemntBd': 6, 'BrkFace': 7, 'WdShing': 8,
                  'Stucco': 9, 'AsbShng': 10, 'BrkComm': 11, 'Stone': 12, 'CBlock': 13, 'AsphShn': 14, 'ImStucc': 15}

for dataset in combine:
    dataset['Exterior2nd'] = dataset['Exterior2nd'].map(map_Exterior2nd)
    dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(1)
map_Exterior1st = {'VinylSd': 1, 'HdBoard': 2, 'MetalSd': 3, 'Wd Sdng': 4, 'Plywood': 5, 'CemntBd': 6, 'BrkFace': 7, 'WdShing': 8,
                  'Stucco': 9, 'AsbShng': 10, 'BrkComm': 11, 'Stone': 12, 'CBlock': 13, 'AsphShn': 14, 'ImStucc': 15}

for dataset in combine:
    dataset['Exterior1st'] = dataset['Exterior1st'].map(map_Exterior1st)
    dataset['Exterior1st'] = dataset['Exterior1st'].fillna(1)
map_RoofMatl = {'CompShg': 1, 'Tar&Grv': 2, 'WdShngl': 3, 'WdShake': 4, 'Membran': 5, 'Roll': 6, 'ClyTile': 7, 'Metal': 8}

for dataset in combine:
    dataset['RoofMatl'] = dataset['RoofMatl'].map(map_RoofMatl)

map_RoofStyle = {'Gable': 1, 'Hip': 2, 'Flat': 3, 'Gambrel': 4, 'Mansard': 5, 'Shed': 6}

for dataset in combine:
    dataset['RoofStyle'] = dataset['RoofStyle'].map(map_RoofStyle)
train['YearBuiltBand'] = pd.cut(dataset['YearBuilt'], 20)
for dataset in combine:
    dataset.loc[dataset['YearBuilt'] <= 1878, 'YearBuilt'] = 0
    dataset.loc[(dataset['YearBuilt'] > 1878) & (dataset['YearBuilt'] <= 1885), 'YearBuilt'] = 1
    dataset.loc[(dataset['YearBuilt'] > 1885) & (dataset['YearBuilt'] <= 1892), 'YearBuilt'] = 2
    dataset.loc[(dataset['YearBuilt'] > 1892) & (dataset['YearBuilt'] <= 1898), 'YearBuilt'] = 3
    dataset.loc[(dataset['YearBuilt'] > 1898) & (dataset['YearBuilt'] <= 1905), 'YearBuilt'] = 4
    dataset.loc[(dataset['YearBuilt'] > 1905) & (dataset['YearBuilt'] <= 1911), 'YearBuilt'] = 5
    dataset.loc[(dataset['YearBuilt'] > 1911) & (dataset['YearBuilt'] <= 1918), 'YearBuilt'] = 6
    dataset.loc[(dataset['YearBuilt'] > 1918) & (dataset['YearBuilt'] <= 1924), 'YearBuilt'] = 7
    dataset.loc[(dataset['YearBuilt'] > 1924) & (dataset['YearBuilt'] <= 1931), 'YearBuilt'] = 8
    dataset.loc[(dataset['YearBuilt'] > 1931) & (dataset['YearBuilt'] <= 1937), 'YearBuilt'] = 9
    dataset.loc[(dataset['YearBuilt'] > 1937) & (dataset['YearBuilt'] <= 1944), 'YearBuilt'] = 10
    dataset.loc[(dataset['YearBuilt'] > 1944) & (dataset['YearBuilt'] <= 1951), 'YearBuilt'] = 11
    dataset.loc[(dataset['YearBuilt'] > 1951) & (dataset['YearBuilt'] <= 1957), 'YearBuilt'] = 12
    dataset.loc[(dataset['YearBuilt'] > 1957) & (dataset['YearBuilt'] <= 1964), 'YearBuilt'] = 13
    dataset.loc[(dataset['YearBuilt'] > 1964) & (dataset['YearBuilt'] <= 1970), 'YearBuilt'] = 14
    dataset.loc[(dataset['YearBuilt'] > 1970) & (dataset['YearBuilt'] <= 1977), 'YearBuilt'] = 15
    dataset.loc[(dataset['YearBuilt'] > 1977) & (dataset['YearBuilt'] <= 1983), 'YearBuilt'] = 16
    dataset.loc[(dataset['YearBuilt'] > 1983) & (dataset['YearBuilt'] <= 1990), 'YearBuilt'] = 17
    dataset.loc[(dataset['YearBuilt'] > 1990) & (dataset['YearBuilt'] <= 1996), 'YearBuilt'] = 18
    dataset.loc[(dataset['YearBuilt'] > 1996) & (dataset['YearBuilt'] <= 2003), 'YearBuilt'] = 19
    dataset.loc[(dataset['YearBuilt'] > 2003) & (dataset['YearBuilt'] <= 2010), 'YearBuilt'] = 20
    dataset.loc[dataset['YearBuilt'] > 2010, 'YearBuilt'] = 21
train['YearRemodAddBand'] = pd.cut(dataset['YearRemodAdd'], 10)
for dataset in combine:
    dataset.loc[dataset['YearRemodAdd'] <= 1949, 'YearRemodAdd'] = 0
    dataset.loc[(dataset['YearRemodAdd'] > 1949) & (dataset['YearRemodAdd'] <= 1956), 'YearRemodAdd'] = 1
    dataset.loc[(dataset['YearRemodAdd'] > 1956) & (dataset['YearRemodAdd'] <= 1962), 'YearRemodAdd'] = 2
    dataset.loc[(dataset['YearRemodAdd'] > 1962) & (dataset['YearRemodAdd'] <= 1968), 'YearRemodAdd'] = 3
    dataset.loc[(dataset['YearRemodAdd'] > 1968) & (dataset['YearRemodAdd'] <= 1974), 'YearRemodAdd'] = 4
    dataset.loc[(dataset['YearRemodAdd'] > 1974) & (dataset['YearRemodAdd'] <= 1980), 'YearRemodAdd'] = 5
    dataset.loc[(dataset['YearRemodAdd'] > 1980) & (dataset['YearRemodAdd'] <= 1986), 'YearRemodAdd'] = 6
    dataset.loc[(dataset['YearRemodAdd'] > 1986) & (dataset['YearRemodAdd'] <= 1992), 'YearRemodAdd'] = 7
    dataset.loc[(dataset['YearRemodAdd'] > 1992) & (dataset['YearRemodAdd'] <= 1998), 'YearRemodAdd'] = 8
    dataset.loc[(dataset['YearRemodAdd'] > 1998) & (dataset['YearRemodAdd'] <= 2004), 'YearRemodAdd'] = 9
    dataset.loc[(dataset['YearRemodAdd'] > 2004) & (dataset['YearRemodAdd'] <= 2010), 'YearRemodAdd'] = 10
    dataset.loc[dataset['YearRemodAdd'] > 2010, 'YearRemodAdd'] = 21
map_HouseStyle = {'1Story': 1, '2Story': 2, '1.5Fin': 3, 'SLvl': 4, 'SFoyer': 5, '2.5Unf': 6, '1.5Unf': 7}

for dataset in combine:
    dataset['HouseStyle'] = dataset['HouseStyle'].map(map_HouseStyle)
    dataset['HouseStyle'] = dataset['HouseStyle'].fillna(2)
map_BldgType = {'1Fam': 1, 'TwnhsE': 2, 'Duplex': 3, 'Twnhs': 4, '2fmCon': 5}

for dataset in combine:
    dataset['BldgType'] = dataset['BldgType'].map(map_BldgType)
map_cond1 = {'Norm': 1, 'Feedr': 2, 'Artery': 3, 'RRAn': 4, 'PosN': 5, 'RRAe': 6, 'PosA': 7, 'RRNn': 8, 'RRNe': 9}

for dataset in combine:
    dataset['Condition1'] = dataset['Condition1'].map(map_cond1)
map_cond2 = {'Norm': 1, 'Feedr': 2, 'Artery': 3, 'RRAn': 4, 'PosN': 5, 'RRAe': 6, 'PosA': 7, 'RRNn': 8, 'RRNe': 9}

for dataset in combine:
    dataset['Condition2'] = dataset['Condition2'].map(map_cond2)
map_neigh = {'NAmes': 1, 'OldTown': 2, 'CollgCr': 3, 'Somerst': 4, 'Edwards': 5, 'NridgHt': 6, 'Gilbert': 7, 'Sawyer': 8,
            'Mitchel': 9, 'NWAmes': 0, 'IDOTRR': 10, 'Crawfor': 11, 'BrkSide': 12, 'Timber': 13, 'NoRidge': 14, 'StoneBr': 15,
            'SWISU': 16, 'MeadowV': 17, 'ClearCr': 18, 'NPkVill': 19, 'BrDale': 20, 'Veenker': 21, 'Blmngtn': 22, 'Blueste': 23}

for dataset in combine:
    dataset['Neighborhood'] = dataset['Neighborhood'].map(map_neigh)
    dataset['Neighborhood'] = dataset['Neighborhood'].fillna(24)
map_slope = {'Gtl': 1, 'Mod': 2, 'Sev': 3}

for dataset in combine:
    dataset['LandSlope'] = dataset['LandSlope'].map(map_slope)
map_oth = {'AllPub': 1, 'NoSeWa': 2}

for dataset in combine:
    dataset['Utilities'] = dataset['Utilities'].map(map_oth)
    dataset['Utilities'] = dataset['Utilities'].fillna(1)
map_St = {'Lvl': 1, 'HLS': 2, 'Bnk': 3, 'Low': 4}

for dataset in combine:
    dataset['LandContour'] = dataset['LandContour'].map(map_St)
map_Str = {'Reg': 1, 'IR1': 2, 'IR2': 3, 'IR3': 4}

for dataset in combine:
    dataset['LotShape'] = dataset['LotShape'].map(map_Str)
map_Street = {'Pave': 1, 'Grvl': 2}

for dataset in combine:
    dataset['Street'] = dataset['Street'].map(map_Street)
    
    sns.barplot(x="Street", y="SalePrice", data=train)


map_con = {'Inside': 1, 'Corner': 2, 'CulDSac': 3, 'FR2': 4, 'FR3': 4}

for dataset in combine:
    dataset['LotConfig'] = dataset['LotConfig'].map(map_con)
for dataset in combine:
    train['BandLot'] = pd.cut(dataset['LotFrontage'], 5)
for dataset in combine:
    dataset.loc[dataset['LotFrontage'] <= 20, 'LotFrontage'] = 0
    dataset.loc[(dataset['LotFrontage'] > 20) & (dataset['LotFrontage'] <= 56), 'LotFrontage'] = 1
    dataset.loc[(dataset['LotFrontage'] > 56) & (dataset['LotFrontage'] <= 92), 'LotFrontage'] = 2
    dataset.loc[(dataset['LotFrontage'] > 92) & (dataset['LotFrontage'] <= 128), 'LotFrontage'] = 3
    dataset.loc[(dataset['LotFrontage'] > 128) & (dataset['LotFrontage'] <= 164), 'LotFrontage'] = 4
    dataset.loc[(dataset['LotFrontage'] > 164) & (dataset['LotFrontage'] <= 200), 'LotFrontage'] = 5
    dataset.loc[dataset['LotFrontage'] > 200, 'LotFrontage'] = 7
    dataset['LotFrontage'] = dataset['LotFrontage'].fillna(6)
train['LotFrontage'].value_counts()

for dataset in combine:
    dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(0)
    dataset['BsmtHalfBath'] = dataset['BsmtHalfBath'].fillna(0)
    dataset['KitchenQual'] = dataset['KitchenQual'].fillna(2)
    dataset['Functional'] = dataset['Functional'].fillna(1)
    dataset['GarageType'] = dataset['GarageType'].fillna(100)
    dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(100)
    dataset['GarageFinish'] = dataset['GarageFinish'].fillna(100)
    dataset['GarageCars'] = dataset['GarageCars'].fillna(0)
    dataset['GarageArea'] = dataset['GarageArea'].fillna(3)
    dataset['GarageQual'] = dataset['GarageQual'].fillna(0)
    dataset['GarageCond'] = dataset['GarageCond'].fillna(100)
    dataset['SaleType'] = dataset['SaleType'].fillna(1)
map_MSZoning = {'RL': 0, 'RM': 1, 'FV': 2, 'RH': 3, 'C (all)': 4 }

for dataset in combine:
    dataset['MSZoning'] = dataset['MSZoning'].map(map_MSZoning)
    dataset['MSZoning'] = dataset['MSZoning'].fillna(5)

map_SaleCondition = {'Normal': 0, 'Abnormal': 1, 'Partial': 2, 'AdjLand': 3, 'Alloca': 4, 'Family': 5 }

for dataset in combine:
    dataset['SaleCondition'] = dataset['SaleCondition'].map(map_SaleCondition)
train = train.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
test = test.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

train['Electrical'] = train['Electrical'].fillna(1)
test['SaleCondition'] = test['SaleCondition'].fillna(50)
train['SaleCondition'] = train['SaleCondition'].fillna(50)
train = train.drop(['GrLivAreaBand', 'TotalBsmtSFBand', 'WoodDeckSFBand', 'ScreenPorchBand', 'LotAreaBand',
                   '1stFlrSFBand', 'OpenPorchSFBand', 'EnclosedPorchBand', 'GarageAreaBand', 'BsmtFinSF1Band',
                   'MasVnrAreaBand', 'GarageYrBltBand', 'YearBuiltBand', 'YearRemodAddBand', 'BandLot',
                    '2ndFlrSFBand', 'BsmtUnfSFBand'], axis=1)
train.info()
sns.barplot(x="MSSubClass", y="SalePrice", data=train)
train = train.drop(['Id'], axis=1)
x_train = train.drop(['SalePrice'], axis=1)
y_train = train['SalePrice']
x_test = test.drop(['Id'], axis=1).copy()

x_train.shape, y_train.shape, x_test.shape

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc = round(clf.score(x_train, y_train)*100, 2)
print(acc)
clf = SVC()
clf.fit(x_train, y_train)
y_pred_svc = clf.predict(x_test)
acc_svc = round(clf.score(x_train, y_train) * 100, 2)
print (acc_svc)
clf = LinearSVC()
clf.fit(x_train, y_train)
y_pred_linear_svc = clf.predict(x_test)
acc_linear_svc = round(clf.score(x_train, y_train) * 100, 2)
print (acc_linear_svc)
clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(x_train, y_train)
y_pred_knn = clf.predict(x_test)
acc_knn = round(clf.score(x_train, y_train) * 100, 2)
print (acc_knn)
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_pred_decision_tree = clf.predict(x_test)
acc_decision_tree = round(clf.score(x_train, y_train) * 100, 2)
print (acc_decision_tree)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(x_train, y_train)
y_pred_random_forest = clf.predict(x_test)
acc_random_forest = round(clf.score(x_train, y_train) * 100, 2)
print (acc_random_forest)
clf = GaussianNB()
clf.fit(x_train, y_train)
y_pred_gnb = clf.predict(x_test)
acc_gnb = round(clf.score(x_train, y_train) * 100, 2)
print (acc_gnb)
clf = Perceptron(max_iter=5, tol=None)
clf.fit(x_train, y_train)
y_pred_perceptron = clf.predict(x_test)
acc_perceptron = round(clf.score(x_train, y_train) * 100, 2)
print (acc_perceptron)
clf = SGDClassifier(max_iter=5, tol=None)
clf.fit(x_train, y_train)
y_pred_sgd = clf.predict(x_test)
acc_sgd = round(clf.score(x_train, y_train) * 100, 2)
print (acc_sgd)
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 
              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 
              'Perceptron', 'Stochastic Gradient Decent'],
    
    'Score': [acc, acc_svc, acc_linear_svc, 
              acc_knn,  acc_decision_tree, acc_random_forest, acc_gnb, 
              acc_perceptron, acc_sgd]
    })

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({
        "Id": test["Id"],
        "SalePrice": y_pred_random_forest
    })


submission.to_csv('HousePrice.csv', index=False)