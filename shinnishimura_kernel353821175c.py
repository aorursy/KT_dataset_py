# imports of machine learning

from sklearn.model_selection import learning_curve as LC

from sklearn.linear_model import LogisticRegression as LR

from sklearn.linear_model import RidgeCV as RCV

from sklearn.svm import SVC

from sklearn.svm import LinearSVC as LSVC

from sklearn.ensemble import ExtraTreesRegressor as ETR

from sklearn.ensemble import RandomForestRegressor as RFR

from sklearn.ensemble import AdaBoostRegressor as ABR

from sklearn.ensemble import BaggingRegressor as BR

from sklearn.ensemble import GradientBoostingRegressor as GBR

from sklearn.ensemble import StackingRegressor as SR

from sklearn.neural_network import MLPRegressor as MLPR

from sklearn.tree import DecisionTreeRegressor as DTR

from sklearn.impute import SimpleImputer as SI

from sklearn.model_selection import train_test_split as TTS

from sklearn.metrics import mean_squared_log_error as MSLE



# imports of everything else

import math

import random

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



# print file name of data

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load data

train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")



# data definition and dictionary

exterQualDict = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}

poolQualDict = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1}

garageQualDict = {'Po': '1to2', 'Fa': '1to2', 'TA': '3', 'Gd': '4to5', 'Ex': '4to5'}

garageCondDict = {'Po': '1to2', 'Fa': '1to2', 'TA': '3', 'Gd': '4to5', 'Ex': '4to5'}

garageFinDict = {'Fin': 3, 'RFn': 2, 'Unf': 1}

fireplaceQualDict = {'Ex': 'Good', 'Gd': 'Good', 'TA': 'Good', 'Fa': 'Bad', 'Po': 'Bad'}

functionalDict = {'Typ': 1, 'Min1': 0, 'Min2': 0, 'Mod': 0, 'Maj1': 0, 'Maj2': 0, 'Sev': 0, 'Sal': 0}

HeatingQCDict = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0}

kitchenQualDict = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'Invalid': 0}

bsmtFinishDict = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'Invalid': 0}

bsmtQualDict = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'Invalid': 0}

bsmtCondDict = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'Invalid': 0}

bsmtExpDict = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'Invalid': 0}

landContourDict = {'Lvl': 1, 'Bnk': 2, 'HLS': 3, 'Low': 4}

landSlopeDict = {'Gtl': 1, 'Mod': 2, 'Sev': 3}

condition = ('Artery', 'Feedr', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe')

exterior = ('AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc',

            'MetalSd', 'Other', 'Plywood', 'PreCast', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing')

roofStyleDict = {'Hip': 'Hip', 'Gable': 'Gable', 'Flat': 'Rare', 'Gambrel': 'Rare', 'Mansard': 'Rare', 'Shed': 'Rare'}

roofMatlDict = {'Metal': 'Rare', 'Membran': 'Rare', 'Roll': 'Rare', 'ClyTile': 'Rare', 

                'CompShg': 'CompShg', 'Tar&Grv': 'Tar&Grv', 'WdShake': 'WdShake', 'WdShngl': 'WdShngl'}

conditionDict = {1: '1to4', 2: '1to4', 3: '1to4', 4: '1to4', 5: '5', 6: '6to9', 7: '6to9', 8: '6to9', 9: '6to9'}

exterConditionDict = {'Po': '1to2', 'Fa': '1to2', 'TA': '3', 'Gd': '4to5', 'Ex': '4to5'}



# data fix

test_data.loc[1132, 'GarageYrBlt'] = 2007



# data

train_data['SalePrice'] = train_data['SalePrice'].apply(lambda x: math.log(x))

for data in train_data, test_data:

    data['LandContNum'] = data['LandContour'].apply(lambda x: landContourDict[x])

    data['LandSlopNum'] = data['LandSlope'].apply(lambda x: landSlopeDict[x])

    data['LotFrontage'] = data['LotFrontage'].apply(lambda x: 0 if math.isnan(x) else x)

    data['G/PStreet'] = data['Street'].apply(lambda x: 0 if x=='Grvl' else 1)

    data['LotRoot'] = data['LotArea']**(1/2)

    data['ConditionClassify'] = data['OverallCond'].apply(lambda x: conditionDict[x])

    data['RoofStyleClassify'] = data['RoofStyle'].apply(lambda x: roofStyleDict[x])

    data['RoofMatlClassify'] = data['RoofMatl'].apply(lambda x: roofMatlDict[x])

    data['MasVnrType'] = data['MasVnrType'].apply(lambda x: 'Invalid' if type(x) == float else x)

    data['MasVnrArea'] = data['MasVnrArea'].apply(lambda x: 0 if math.isnan(x) == float else x)

    data['ExterQual'] = data['ExterQual'].apply(lambda x: exterQualDict[x])

    data['ExterCond'] = data['ExterCond'].apply(lambda x: exterConditionDict[x])

    data['MasVnrType'] = data['MasVnrType'].apply(lambda x: 'Invalid' if type(x) == float else x)

    data['MasVnrArea'] = data['MasVnrArea'].apply(lambda x: 0 if math.isnan(x) else x)

    data['BsmtQual'] = data['BsmtQual'].apply(lambda x: bsmtQualDict['Invalid'] if type(x) == float else bsmtQualDict[x])

    data['BsmtCond'] = data['BsmtCond'].apply(lambda x: bsmtCondDict['Invalid'] if type(x) == float else bsmtCondDict[x])

    data['BsmtExposure'] = data['BsmtExposure'].apply(lambda x: bsmtExpDict['Invalid'] if type(x) == float else bsmtExpDict[x])

    data['BsmtFinType1'] = data['BsmtFinType1'].apply(lambda x: bsmtFinishDict['Invalid'] if type(x) == float else bsmtFinishDict[x])

    data['BsmtFinSF1'] = data['BsmtFinSF1'].apply(lambda x: 0 if math.isnan(x) else x)

    data['Bsmt1*'] = data['BsmtFinSF1'] * data['BsmtFinType1']

    data['BsmtFinType2'] = data['BsmtFinType2'].apply(lambda x: bsmtFinishDict['Invalid'] if type(x) == float else bsmtFinishDict[x])

    data['BsmtFinSF2'] = data['BsmtFinSF2'].apply(lambda x: 0 if math.isnan(x) else x)

    data['Bsmt2*'] = data['BsmtFinSF2'] * data['BsmtFinType2']

    data['TotalBsmtSF'] = data['TotalBsmtSF'].apply(lambda x: 0 if math.isnan(x) else x)

    data['BsmtSF*'] = data['TotalBsmtSF'].apply(lambda x: x**2)

    data['BsmtUnfSF'] = data['BsmtUnfSF'].apply(lambda x: 0 if math.isnan(x) else x)

    data['BsmtFullBath'] = data['BsmtFullBath'].apply(lambda x: 0 if math.isnan(x) else x)

    data['BsmtHalfBath'] = data['BsmtHalfBath'].apply(lambda x: 0 if math.isnan(x) else x)

    data['HeatingQC'] = data['HeatingQC'].apply(lambda x: HeatingQCDict[x])

    data['Electrical'] = data['Electrical'].apply(lambda x: 'Invalid' if type(x) == float else x)

    data['SBrkr'] = data['Electrical'].apply(lambda x: 1 if x=='SBrkr' else 0)

    data['KitchenQual'] = data['KitchenQual'].apply(lambda x: kitchenQualDict['Invalid'] if type(x) == float else kitchenQualDict[x])

    data['Functional'] = data['Functional'].apply(lambda x: 1 if type(x) == float else functionalDict[x])

    data['FireplaceQu'] = data['FireplaceQu'].apply(lambda x: 'Invalid' if type(x) == float else fireplaceQualDict[x])

    data['GarageCars'] = data['GarageCars'].apply(lambda x: 0 if math.isnan(x) else x)

    data['GarageArea'] = data['GarageArea'].apply(lambda x: 0 if math.isnan(x) else x)

    data['GarageType'] = data['GarageType'].apply(lambda x: 'Invalid' if type(x) == float else x)

    data['GarageFinish'] = data['GarageFinish'].apply(lambda x: 0 if type(x) == float else garageFinDict[x])

    data['GarageYrBlt'] = data['GarageYrBlt'].apply(lambda x: 0 if math.isnan(x) else x)

    data['GarageQual'] = data['GarageQual'].apply(lambda x: 'Invalid' if type(x) == float else garageQualDict[x])

    data['GarageCond'] = data['GarageCond'].apply(lambda x: 'Invalid' if type(x) == float else garageCondDict[x])

    data['PoolQC'] = data['PoolQC'].apply(lambda x: 0 if type(x) == float else poolQualDict[x])

    data['Fence'] = data['Fence'].apply(lambda x: 'Invalid' if type(x) == float else x)

    data['MiscFeature'] = data['MiscFeature'].apply(lambda x: 'Invalid' if type(x) == float else x)

    data['CentralAir'] = data['CentralAir'].apply(lambda x: 1 if x=='Y' else 0)

    data['YearBuilt'] = data['YearBuilt'].apply(lambda x: str(x))

    data['OverallQual'] = data['OverallQual'].apply(lambda x: str(x))

    data['GrLivAreaRoot'] = data['GrLivArea']**(1/2)

    data['MSSubClass'] = data['MSSubClass'].apply(lambda x: str(x))

    data['MoSold'] = data['MoSold'].astype(str)

    

    for prox in condition:

        data['temp1'] = data['Condition1'].apply(lambda x: 1 if x==prox else 0)

        data['temp2'] = data['Condition1'].apply(lambda x: 1 if x==prox else 0)

        data[prox] = data['temp1'] + data['temp2']

        data[prox] = data[prox].apply(lambda x: 1 if x>0 else 0)

    

    for material in exterior:

        data['temp1'] = data['Condition1'].apply(lambda x: 1 if x==material else 0)

        data['temp2'] = data['Condition1'].apply(lambda x: 1 if x==material else 0)

        data[material] = data['temp1'] + data['temp2']

        data[material] = data[material].apply(lambda x: 1 if x>0 else 0)



# drop irrelevant

irr = ['Utilities', 'RoofStyle', 'RoofMatl', 'Electrical', 'KitchenAbvGr', 'Street', 'Alley', 

       'Condition1', 'Condition2', 'Exterior1st', 'Exterior2nd', 'temp1', 'temp2'

      ]

train_data = train_data.drop(irr, axis = 1)

test_data = test_data.drop(irr, axis = 1)



# plotting

# print(train_data)

# features = [

#     'SalePrice',

#     'SaleType', 'SaleCondition'

# ]

# data = pd.get_dummies(train_data).astype(float).corr()



# plt.figure(figsize = (720, 10))

# sns.heatmap(data=data, annot=True, fmt = ".2f", cmap = "seismic", square = True, vmax = 1.0, vmin = -1.0)



# sns.catplot(x='YrSold' , y='SalePrice' , data=full_data)

# sns.catplot(x='MoSold' , y='SalePrice' , data=full_data, col='YrSold', kind='swarm')

# sns.catplot(x='MoSold' , y='SalePrice' , data=full_data, col='YrSold', kind='bar')

# sns.relplot(x='MoSold' , y='SalePrice' , data=full_data, col='YrSold', kind='line')

# sns.lmplot(x='TotalBsmtSF' , y='SalePrice' , data=train_data)

# sns.lmplot(x='GrLivAreaRoot' , y='SalePrice' , data=train_data)

# sns.lmplot(x='MiscVal' , y='SalePrice' , data=full_data, hue='MiscFeature')



# train_data[features].groupby([features[0]], as_index=False).mean().sort_values(by=features[1], ascending=False)



# g = sns.FacetGrid(train_data, col='LotFrontageValidity')

# g.map(plt.hist, 'SalePrice', bins=20)
seed = 2

price = 'SalePrice'

full_data = pd.concat([train_data, test_data], sort = True, axis=0)

dropPostDummy = [

#     '3SsnPorch', 'AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'BsmtHalfBath', 'CBlock', 'CemntBd', 'Feedr', 'G/PStreet', 'HdBoard', 'ImStucc', 'LandSlopNum', 'LowQualFinSF', 'MetalSd', 'MiscVal', 'Other', 'Plywood', 'PosA', 'PosN', 'PreCast', 'RRAe', 'RRAn', 'RRNe', 'RRNn', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BsmtQual_0', 'ConditionClassify_1to4', 'ExterCond_1to2', 'ExterCond_3', 'ExterCond_4to5', 'Fence_GdPrv', 'Fence_GdWo', 'Fence_MnWw', 'FireplaceQu_Bad', 'Foundation_BrkTil', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood', 'GarageCond_1to2', 'GarageCond_3', 'GarageCond_4to5', 'GarageCond_Invalid', 'GarageQual_1to2', 'GarageQual_4to5', 'GarageQual_Invalid', 'GarageType_2Types', 'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Invalid', 'Heating_Floor', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'HouseStyle_1.5Fin', 'HouseStyle_1.5Unf', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'KitchenQual_0', 'KitchenQual_2', 'LandContour_Bnk', 'LandContour_Low', 'LandSlope_Mod', 'LandSlope_Sev', 'LotConfig_FR2', 'LotConfig_FR3', 'LotShape_IR3', 'MSZoning_C (all)', 'MSZoning_FV', 'MSZoning_RH', 'MasVnrType_BrkCmn', 'MasVnrType_Stone', 'MiscFeature_Gar2', 'MiscFeature_Invalid', 'MiscFeature_Othr', 'MiscFeature_Shed', 'MiscFeature_TenC', 'Neighborhood_Blmngtn', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_Timber', 'OverallQual_1', 'OverallQual_2', 'OverallQual_3', 'PavedDrive_N', 'PavedDrive_P', 'PavedDrive_Y', 'RoofMatlClassify_CompShg', 'RoofMatlClassify_Rare', 'RoofMatlClassify_Tar&Grv', 'RoofMatlClassify_WdShake', 'RoofMatlClassify_WdShngl', 'RoofStyleClassify_Rare', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleType_COD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_Oth', 'YearBuilt_1872', 'YearBuilt_1875', 'YearBuilt_1879', 'YearBuilt_1880', 'YearBuilt_1882', 'YearBuilt_1885', 'YearBuilt_1890', 'YearBuilt_1892', 'YearBuilt_1893', 'YearBuilt_1895', 'YearBuilt_1896', 'YearBuilt_1898', 'YearBuilt_1900', 'YearBuilt_1901', 'YearBuilt_1902', 'YearBuilt_1904', 'YearBuilt_1905', 'YearBuilt_1906', 'YearBuilt_1907', 'YearBuilt_1908', 'YearBuilt_1910', 'YearBuilt_1911', 'YearBuilt_1912', 'YearBuilt_1913', 'YearBuilt_1914', 'YearBuilt_1915', 'YearBuilt_1916', 'YearBuilt_1917', 'YearBuilt_1918', 'YearBuilt_1919', 'YearBuilt_1920', 'YearBuilt_1921', 'YearBuilt_1922', 'YearBuilt_1923', 'YearBuilt_1924', 'YearBuilt_1925', 'YearBuilt_1926', 'YearBuilt_1927', 'YearBuilt_1928', 'YearBuilt_1930', 'YearBuilt_1931', 'YearBuilt_1932', 'YearBuilt_1934', 'YearBuilt_1935', 'YearBuilt_1936', 'YearBuilt_1937', 'YearBuilt_1938', 'YearBuilt_1939', 'YearBuilt_1940', 'YearBuilt_1941', 'YearBuilt_1942', 'YearBuilt_1945', 'YearBuilt_1946', 'YearBuilt_1947', 'YearBuilt_1948', 'YearBuilt_1949', 'YearBuilt_1951', 'YearBuilt_1952', 'YearBuilt_1953', 'YearBuilt_1954', 'YearBuilt_1955', 'YearBuilt_1956', 'YearBuilt_1957', 'YearBuilt_1958', 'YearBuilt_1959', 'YearBuilt_1960', 'YearBuilt_1961', 'YearBuilt_1962', 'YearBuilt_1963', 'YearBuilt_1964', 'YearBuilt_1965', 'YearBuilt_1966', 'YearBuilt_1967', 'YearBuilt_1968', 'YearBuilt_1970', 'YearBuilt_1971', 'YearBuilt_1972', 'YearBuilt_1973', 'YearBuilt_1974', 'YearBuilt_1975', 'YearBuilt_1976', 'YearBuilt_1978', 'YearBuilt_1979', 'YearBuilt_1980', 'YearBuilt_1981', 'YearBuilt_1982', 'YearBuilt_1983', 'YearBuilt_1984', 'YearBuilt_1985', 'YearBuilt_1986', 'YearBuilt_1987', 'YearBuilt_1988', 'YearBuilt_1989', 'YearBuilt_1990', 'YearBuilt_1991', 'YearBuilt_1992', 'YearBuilt_1994', 'YearBuilt_1995', 'YearBuilt_1997', 'YearBuilt_1998', 'YearBuilt_1999', 'YearBuilt_2001', 'YearBuilt_2002', 'YearBuilt_2008', 'YearBuilt_2010'

]

dropPreDummy = [

    

]



less = [0, math.inf, []]

while True:

    y = train_data[price]

    X = pd.get_dummies(full_data.drop(dropPreDummy, axis=1)).dropna(axis=0).drop([price, 'Id']+dropPostDummy, axis=1)

    train_X, val_X, train_y, val_y = TTS(X, y, random_state = seed, test_size = 0.25)



    models = [

        RFR(random_state = seed)

    ]

    for modelnum in range(len(models)):

        models[modelnum].fit(train_X, train_y)

        prediction = models[modelnum].predict(val_X)

        mean = np.sqrt(MSLE(np.expm1(val_y), np.expm1(prediction)))

        show = pd.DataFrame({'Prediction': np.expm1(prediction), price: np.expm1(val_y)})

        show['Difference'] = (show['Prediction']-show[price]).apply(lambda x: math.fabs(x))

        print(show.head())

        print('-'*50)

        models[modelnum].fit(val_X, val_y)

        

        importance, values = models[less[0]].feature_importances_, X.columns.values

        for item in range(len(values)):

            if importance[item] > 0.05:

                print('Importance:', importance[item].round(5), '\t', 'Feature:', values[item])

        print('-'*50)

        ease = []



        for item in range(len(values)):

            if importance[item] <= 0.00004:

                ease.append(values[item])

#                 print('Importance:', importance[item].round(5), '\t', 'Feature:', values[item])

#         print('-'*50)

        print('Dropping', len(ease))

        dropPostDummy += ease

        

        if mean <= less[1]:

            less = [modelnum, mean, dropPostDummy]

            

    print('Score:', less[1])

    print('-'*50)

    if ease == []:

        break

print('Total dropped parameters:', len(less[2]))

print('Final score:', less[1])

print('-'*50)

            



full_data[price] = full_data[price].apply(lambda x: 1 if math.isnan(x) else math.nan)

X_test = pd.get_dummies(full_data.drop(dropPreDummy, axis=1)).dropna(axis=0).drop([price, 'Id']+less[2], axis=1)



importance, values = models[less[0]].feature_importances_, X.columns.values

plt.figure(figsize = (70, 70))

plot = pd.DataFrame({'Importance': importance, 'Value': values})

plot = plot.sort_values(by=['Importance'])

sns.barplot(x = 'Importance', y = 'Value', data = plot)



predictions = models[less[0]].predict(X_test)
output = pd.DataFrame({'Id': test_data.Id, price: np.expm1(predictions)})

output.to_csv('submission.csv', index=False)

print(output.head(10))

print('etc')

print('--------------------------')

print("success")