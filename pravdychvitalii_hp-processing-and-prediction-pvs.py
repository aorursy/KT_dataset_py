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
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv').set_index('Id')
train.head()
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv').set_index('Id')
test.head()
ssbm = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv').set_index('Id')
ssbm.head()
# check out data
train.columns
print(train.shape, test.shape)
train.info()
correlations_data = train.corr()['SalePrice'].sort_values()
correlations_data 
# for the best result "SalePrice" logarithm
y = np.log(train["SalePrice"] + 1)
train = train.drop(["SalePrice"], axis = 1)
# data preprocessing

def preprocessing(X):
   
    # gruoping columns
    X['YrBltAndRemod']=X['YearBuilt']+X['YearRemodAdd']

    X['TotalSF']= X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
 

    X['Total_sqr_footage'] = X['BsmtFinSF1'] + X['BsmtFinSF2'] + X['1stFlrSF'] + X['2ndFlrSF']

    X['Total_Bathrooms'] = X['FullBath'] + (0.5 * X['HalfBath']) + X['BsmtFullBath'] + (0.5 * X['BsmtHalfBath'])

    X['Total_porch_sf'] = X['OpenPorchSF'] + X['3SsnPorch'] + X['EnclosedPorch'] + X['ScreenPorch'] + X['WoodDeckSF']
    
    #fillna    
    X['Exterior1st'] = X['Exterior1st'].fillna(X['Exterior1st'].mode()[0])
    X['Exterior2nd'] = X['Exterior2nd'].fillna(X['Exterior2nd'].mode()[0])
    X['SaleType'] = X['SaleType'].fillna(X['SaleType'].mode()[0])
    X['Electrical'] = X['Electrical'].fillna(X['Electrical'].mode()[0])
    X['KitchenQual'] = X['KitchenQual'].fillna(X['KitchenQual'].mode()[0])
        
    # drop columns
    drop_elements = ['YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1',
                             'BsmtFinSF2', 'FullBath', 'HalfBath', 'BsmtFullBath',
                             'BsmtHalfBath', 'OpenPorchSF', '3SsnPorch', 'EnclosedPorch',
                             'ScreenPorch', 'WoodDeckSF', 'Utilities',
                             'LotFrontage', 'LotShape','MiscFeature', 'MSSubClass', 
                             'Street', 'PoolQC', 'YrSold', 'LowQualFinSF' ,'MiscVal', 'MoSold']
          
    X = X.drop(drop_elements, axis=1)
        
    # replace name
    X['MSZoning'] = X['MSZoning'].replace(['RL', 'RM', 'C (all)', 'FV', 'RH'],[1, 2, 3, 4, 5])
    
    X['Alley'] = X['Alley'].replace(['Pave', 'Grvl'],[1, 2])
    
    X['LandContour'] = X['LandContour'].replace(['Lvl', 'Bnk', 'Low', 'HLS'],[1, 2, 3, 4])
    
    X['LotConfig'] = X['LotConfig'].replace(['Inside', 'FR2', 'Corner', 'CulDSac', 'FR3'],[1, 2, 3, 4, 5])
    
    X['LandSlope'] = X['LandSlope'].replace(['Gtl', 'Mod', 'Sev'],[1, 2, 3])
    
    X['Neighborhood'] = X['Neighborhood'].replace(['CollgCr', 'Veenker',
                                                   'Crawfor', 'NoRidge', 'Mitchel', 'Somerst',
                                                   'NWAmes', 'OldTown', 'BrkSide', 'Sawyer',
                                                   'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR',
                                                   'MeadowV', 'Edwards', 'Timber', 'Gilbert', 
                                                   'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn',
                                                   'BrDale', 'SWISU', 'Blueste'],
                                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
                                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 
                                                   23, 24, 25])
    
    X['Condition1'] = X['Condition1'].replace(['Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA',
           'RRNe'],[1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    X['Condition2'] = X['Condition2'].replace(['Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA',
           'RRNe'],[1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    X['BldgType'] = X['BldgType'].replace(['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'],[1, 2, 3, 4, 5])
    
    X['HouseStyle'] = X['HouseStyle'].replace(['2Story', '1Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf',
                                               '2.5Fin'],[8, 7, 6, 5, 4, 3, 2, 1])
    
    X['RoofStyle'] = X['RoofStyle'].replace(['Gable', 'Hip', 'Gambrel', 'Mansard', 'Flat', 'Shed'],[6, 5, 4, 3, 2, 1])
    
    X['RoofMatl'] = X['RoofMatl'].replace(['CompShg', 'WdShngl', 'Metal', 'WdShake', 'Membran', 'Tar&Grv',
           'Roll', 'ClyTile'],[8, 7, 6, 5, 4, 3, 2, 1])
    
    X['Exterior1st'] = X['Exterior1st'].replace(['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing',
           'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn',
           'Stone', 'ImStucc', 'CBlock', 'Other', 'Brk Cmn', 'Wd Shng', 'CmentBd'],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
            13, 14, 15, 16, 17, 18, 19])
    
    X['Exterior2nd'] = X['Exterior2nd'].replace(['VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing',
           'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn',
           'Stone', 'ImStucc', 'CBlock', 'Other', 'Brk Cmn', 'Wd Shng', 'CmentBd'],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
            13, 14, 15, 16, 17, 18, 19])
    
    X['MasVnrType'] = X['MasVnrType'].replace(['BrkFace', 'Stone', 'BrkCmn', 'None'],[1, 2, 3, 0])
    
    X['ExterQual'] = X['ExterQual'].replace(['Gd', 'TA', 'Ex', 'Fa', 'Po'],[4, 3, 5, 2, 1])
    
    X['ExterCond'] = X['ExterCond'].replace(['Gd', 'TA', 'Ex', 'Fa', 'Po'],[4, 3, 5, 2, 1])
    
    X['Foundation'] = X['Foundation'].replace(['PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone'],[1, 2, 3, 4, 5, 6])
    
    X['BsmtQual'] = X['BsmtQual'].replace(['Gd', 'TA', 'Ex', 'Fa', 'Po'],[4, 3, 5, 2, 1])
    
    X['BsmtCond'] = X['BsmtCond'].replace(['Gd', 'TA', 'Ex', 'Fa', 'Po'],[4, 3, 5, 2, 1])
    
    X['BsmtExposure'] = X['BsmtExposure'].replace(['No', 'Gd', 'Mn', 'Av'],[0, 3, 2, 1])
    
    X['BsmtFinType1'] = X['BsmtFinType1'].replace(['GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', 'LwQ'], [6, 5, 4, 3, 2, 1])
    
    X['BsmtFinType2'] = X['BsmtFinType2'].replace(['GLQ', 'ALQ', 'Unf', 'Rec', 'BLQ', 'LwQ'], [6, 5, 4, 3, 2, 1])
    
    X['Functional'] = X['Functional'].replace(['Typ', 'Min1', 'Maj1', 'Min2', 'Mod', 'Maj2', 'Sev'], [1, 2, 3, 4, 5, 6, 7])
    
    X['Heating'] = X['Heating'].replace(['GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'], [1, 2, 3, 4, 5, 6])
    
    X['HeatingQC'] = X['HeatingQC'].replace(['Gd', 'TA', 'Ex', 'Fa', 'Po'], [4, 3, 5, 2, 1])
    
    X['CentralAir'] = X['CentralAir'].replace(['Y', 'N'], [1, 0])
    
    X['Electrical'] = X['Electrical'].replace(['SBrkr', 'FuseF', 'FuseA', 'FuseP', 'Mix'], [5, 4, 3, 2, 1])
    
    X['KitchenQual'] = X['KitchenQual'].replace(['Gd', 'TA', 'Ex', 'Fa', 'Po'], [4, 3, 5, 2, 1])
    
    X['FireplaceQu'] = X['FireplaceQu'].replace(['Gd', 'TA', 'Ex', 'Fa', 'Po'], [4, 3, 5, 2, 1])
    
    X['GarageType'] = X['GarageType'].replace(['Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'Basment', '2Types'], [1, 2, 3, 4, 5, 6])
    
    X['GarageFinish'] = X['GarageFinish'].replace(['RFn', 'Unf', 'Fin'], [1, 2, 3])
    
    X['GarageQual'] = X['GarageQual'].replace(['Gd', 'TA', 'Ex', 'Fa', 'Po'], [4, 3, 5, 2, 1])
    
    X['GarageCond'] = X['GarageCond'].replace(['Gd', 'TA', 'Ex', 'Fa', 'Po'], [4, 3, 5, 2, 1])
    
    X['PavedDrive'] = X['PavedDrive'].replace(['Y', 'N', 'P'], [2, 0, 1])
    
    X['Fence'] = X['Fence'].replace(['MnPrv', 'GdWo', 'GdPrv', 'MnWw'], [1, 2, 3, 4])
    
    X['SaleType'] = X['SaleType'].replace(['WD', 'New', 'COD', 'ConLD', 'ConLI', 'CWD', 'ConLw', 'Con', 'Oth'], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    
    X['SaleCondition'] = X['SaleCondition'].replace(['Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family'], [1, 2, 3, 4, 5, 6])

    #fillna 
    X['Alley'] = X['Alley'].fillna(0)
    X['MasVnrType'] = X['MasVnrType'].fillna(0)
    X['MasVnrArea'] = X['MasVnrArea'].fillna(0)
    X['BsmtQual'] = X['BsmtQual'].fillna(0)
    X['BsmtCond'] = X['BsmtCond'].fillna(0)
    X['BsmtExposure'] = X['BsmtExposure'].fillna(0)
    X['BsmtFinType1'] = X['BsmtFinType1'].fillna(0)
    X['BsmtFinType2'] = X['BsmtFinType2'].fillna(0)
    X['FireplaceQu'] = X['FireplaceQu'].fillna(0)
    X['GarageType'] = X['GarageType'].fillna(0)
    X['GarageYrBlt'] = X['GarageYrBlt'].fillna(0)
    X['GarageFinish'] = X['GarageFinish'].fillna(0)
    X['GarageQual'] = X['GarageQual'].fillna(0)
    X['GarageCond'] = X['GarageCond'].fillna(0)
    X['Fence'] = X['Fence'].fillna(0)
    X['MSZoning'] = X['MSZoning'].fillna(0)
    X['Functional'] = X['Functional'].fillna(0)
    X['GarageCars'] = X['GarageCars'].fillna(0)
    X['GarageArea'] = X['GarageArea'].fillna(0)
    X['TotalSF'] = X['TotalSF'].fillna(0)
    X['Total_sqr_footage'] = X['Total_sqr_footage'].fillna(0)
    X['Total_Bathrooms'] = X['Total_Bathrooms'].fillna(0)
    X['BsmtUnfSF'] = X['BsmtUnfSF'].fillna(0)
        
    return X
X = preprocessing(train)
X_for_pred = preprocessing(test)
# let's check some models for the best result
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

classifiers = [
    AdaBoostRegressor(),
    BaggingRegressor(),
    ExtraTreesRegressor(),
    GradientBoostingRegressor(),
    RandomForestRegressor(),
    BayesianRidge(),
    KNeighborsRegressor(),
    DecisionTreeRegressor(),
    SVR()
]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_cols = ["Regressor", "Accuracy"]
log = pd.DataFrame(columns=log_cols)


mape_dict = {}

for clf in classifiers:
    name = clf.__class__.__name__
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, train_predictions)
        
    if name in mape_dict:
        mape_dict[name] += mape
    else:
        mape_dict[name] = mape

for clf in mape_dict:
    log_entry = pd.DataFrame([[clf, mape_dict[clf]]], columns = log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Regressor MAPE')

sns.set_color_codes("muted")
sns.barplot(x = 'Accuracy', y = 'Regressor', data = log, color = "b")
mape_dict
#best result 'GradientBoostingRegressor' using for predict
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.03,
                                   random_state =42)
gbr.fit(X, y)
pred = gbr.predict(X_for_pred)
ssbm['SalePrice'] = np.exp(pred)
ssbm.to_csv('submission.csv', index='id')
#Score 0.12858