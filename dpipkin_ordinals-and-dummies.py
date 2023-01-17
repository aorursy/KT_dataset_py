import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
test = pd.read_csv('../input/test.csv')
lot_mapper = {'Reg': 4 , 'IR1': 3, 'IR2': 2, 'IR3': 1}
util_mapper = {'AllPub': 4 , 'NoSewr': 3, 'NoSeWa': 2, 'ELO': 1}
slope_mapper = {'Gtl': 3, 'Mod': 2, 'Sev': 1}
expo_mapper = {'Ex': 5, 'Gd':4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
bsmt_exposure_mapper = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
bsmt_fin_mapper = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
electrical_mapper = {'SBrkr': 5, 'FuseA': 4, 'FuseF': 3, 'FuseP': 2, 'Mix': 1}
functional_mapper = {'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1}
finish_mapper = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0}
paved_mapper = {'Y': 2, 'P': 1, 'N': 0}
fence_mapper = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0}
df = pd.read_csv('../input/train.csv')
df.columns
ordinal_df = df.replace({'LotShape': lot_mapper,
                         'Utilities': util_mapper,
                         'LandSlope': slope_mapper,
                         'ExterQual': expo_mapper,
                         'ExterCond': expo_mapper,
                         'BsmtQual': expo_mapper,
                         'BsmtCond': expo_mapper,
                         'BsmtExposure': bsmt_exposure_mapper,
                         'BsmtFinType1': bsmt_fin_mapper,
                         'BsmtFinType2': bsmt_fin_mapper,
                         'HeatingQC': expo_mapper,
                         'Electrical': electrical_mapper,
                         'KitchenQual': expo_mapper,
                         'Functional': functional_mapper,
                         'FireplaceQu': expo_mapper,
                         'GarageFinish': finish_mapper,
                         'GarageQual': expo_mapper,
                         'GarageCond': expo_mapper,
                         'PavedDrive': paved_mapper,
                         'PoolQC': expo_mapper,
                         'Fence': fence_mapper})
ordinal_df[['LotShape', 'Utilities', 'LandSlope', 'ExterQual',
            'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
            'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical',
            'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish',
            'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']].head(20)