import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from lightgbm import LGBMRegressor

from sklearn.model_selection import (StratifiedKFold, KFold, cross_validate)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

pd.options.display.max_columns= None

pd.options.display.max_rows= None

np.set_printoptions(suppress=True)

print('All libraries imported.')
train_df= pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col= 'Id')

test_df= pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col= 'Id')

df= pd.concat([train_df, test_df], axis= 0)

print(len(df.columns), ' columns:')

print(df.columns) # lots of columns present
print(df.isna().sum().sum(), ' total missing values.')

print(len(df), ' is the number of rows.')
print('Column\t\t\tDtype\t\t\tMissing\t\tMissing%')

l= len(df)

for i in df.columns:

    n= df[i].isna().sum()

    print(i, '\t\t', df[i].dtype, '\t\t', n, '\t\t', (n*100)/l)
df[['Neighborhood', 'SalePrice']].groupby(['Neighborhood'], as_index=True).mean().sort_values(by='SalePrice', ascending=False)
df.replace({

    'BsmtCond': {'Ex':3, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1},

    'BsmtQual': {'Ex':3, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1},

    'CentralAir': {'N': 0, 'Y': 1},

    'ExterQual': {'Ex':3, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1},

    'ExterCond': {'Ex':3, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1},

    'Functional': {'Typ':3, 'Min1':2, 'Min2':2, 'Mod':2, 'Maj1':1, 'Maj2':1, 'Sev':0, 'Sal':0},

    'HeatingQC': {'Ex':3, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1},

    'HouseStyle': {'1Story':1, '1.5Fin':1, '1.5Unf':1, '2Story':2, '2.5Fin':2, '2.5Unf':2, 'SFoyer':0, 'SLvl':0},

    'KitchenQual': {'Ex':3, 'Gd':3, 'TA':2, 'Fa':1, 'Po':1},

    'Street': {'Pave':1, 'Grvl':0},

    'PavedDrive': {'Y':1, 'P':1, 'N':0}

}, inplace= True)
df.replace({

    'Neighborhood': {'NoRidge':3, 'NridgHt':3, 'StoneBr':3, 'Timber':2, 'Veenker':2, 'Somerst':2, 'ClearCr':2,

                     'Crawfor':2, 'CollgCr':2, 'Blmngtn':2, 'Gilbert':2, 'NWAmes':2, 'SawyerW':2, 'Mitchel':1,

                     'NAmes':1, 'NPkVill':1, 'SWISU':1, 'Blueste':1, 'Sawyer':1, 'OldTown':1, 'Edwards':1,

                     'BrkSide':1, 'BrDale':1, 'IDOTRR':1, 'MeadowV':1}

}, inplace= True)
df['Age']= ' '

df['Age']= df['YrSold']-df['YearRemodAdd']
df['AlleyAccess']= ' '

df['AlleyAccess'][df['Alley'].isna()]= 0 

df['AlleyAccess'][df['Alley'].notna()]= 1
df['Bathrooms']= ' '

df['Bathrooms']= (2*df['BsmtFullBath'])+df['BsmtHalfBath']+(2*df['FullBath'])+df['HalfBath']
df['BsmtRating']= ' '

df['BsmtRating']= df['BsmtQual']*df['BsmtCond']

df['BsmtRating'][df['BsmtRating'].isna()]= 0
df['BsmtYN']= ' '

df['BsmtYN'][df['BsmtQual'].isna()]= 0

df['BsmtYN'][df['BsmtQual'].notna()]= 1

df['BsmtFinSF1'][df['BsmtQual'].isna()]= 0

df['BsmtFinSF1'][df['BsmtQual'].isna()]= 0

df['BsmtUnfSF'][df['BsmtQual'].isna()]= 0

df['TotalBsmtSF'][df['BsmtQual'].isna()]= 0
df['TotalSqFt']= ' '

df['TotalSqFt']= df['1stFlrSF']+df['2ndFlrSF']+df['LowQualFinSF']+df['TotalBsmtSF']+df['BsmtFinSF2']+df['BsmtFinSF1']+df['GrLivArea']+df['BsmtUnfSF']
df['BsmtCond'][df['BsmtYN']==0]= 0

df['BsmtCond'][df['BsmtCond'].isna()]= 0

df['BsmtQual'][df['BsmtYN']==0]= 0

df['Functional'][df['Functional'].isna()]= 0

df['GarageArea'][df['GarageArea'].isna()]= df['GarageArea'].mean()

df['KitchenQual'][df['KitchenQual'].isna()]= 2 # imputing with mode value

df['TotalSqFt'][df['TotalSqFt'].isna()]= df['TotalSqFt'].mean()

df['Bathrooms'][df['Bathrooms'].isna()]= 4 # imputing with mode value
df['Cond1']= ' '

df['Cond1'][df['Condition1']=='Norm']= 0

df['Cond1'][df['Condition1']!='Norm']= 1

df['Cond2']= ' '

df['Cond2'][df['Condition2']=='Norm']= 0

df['Cond2'][df['Condition2']!='Norm']= 1

df['Condition']= df['Cond1']+df['Cond2']
df['FenceYN']= ' '

df['FenceYN'][df['Fence'].isna()]= 0 

df['FenceYN'][df['Fence'].notna()]= 1
df['FireplaceYN']= ' '

df['FireplaceYN'][df['FireplaceQu'].isna()]= 0 

df['FireplaceYN'][df['FireplaceQu'].notna()]= 1

df['Fireplaces'][df['FireplaceQu'].isna()]= 0 
df['GarageYN']= ' '

df['GarageYN'][df['GarageType'].isna()]= 0

df['GarageYN'][df['GarageType'].notna()]= 1

df['GarageArea'][df['GarageType'].isna()]= 0
df['MasVnrYN']= ' '

df['MasVnrYN'][df['MasVnrType'].isna()]= 0

df['MasVnrYN'][df['MasVnrType'].notna()]= 1

df['MasVnrArea'][df['MasVnrType'].isna()]= 0
df['Amenities']= ' '

df['Amenities'][df['MiscFeature'].isna()]= 0

df['Amenities'][df['MiscFeature'].notna()]= 1

df['Amenities'][df['MiscFeature'].isna()]= 0
df['PavedYN']= ' '

df['PavedYN']= (2*df['Street'])+df['PavedDrive']
df['PoolYN']= ' '

df['PoolYN'][df['PoolQC'].isna()]= 0

df['PoolYN'][df['PoolQC'].notna()]= 1

df['PoolArea'][df['PoolQC'].isna()]= 0
df['PorchArea']= ' '

df['PorchArea']= df['OpenPorchSF']+df['EnclosedPorch']+df['3SsnPorch']+df['ScreenPorch']
df['Rating']= ' '

df['Rating']= df['OverallQual']*df['OverallCond']
df['Utilities'][df['Utilities'].isna()]= 'AllPub' # imputing with mode value



df['Gas']= ' '

df['Water']= ' '

df['Septic']= ' '



df['Gas'][df['Utilities']=='AllPub']= 1

df['Water'][df['Utilities']=='AllPub']= 1

df['Septic'][df['Utilities']=='AllPub']= 1



df['Gas'][df['Utilities']=='NoSewr']= 1

df['Water'][df['Utilities']=='NoSewr']= 1

df['Septic'][df['Utilities']=='NoSewr']= 0



df['Gas'][df['Utilities']=='NoSeWa']= 1

df['Water'][df['Utilities']=='NoSeWa']= 0

df['Septic'][df['Utilities']=='NoSeWa']= 0



df['Gas'][df['Utilities']=='ELO']= 0

df['Water'][df['Utilities']=='ELO']= 0

df['Septic'][df['Utilities']=='ELO']= 0
df.drop(['1stFlrSF','2ndFlrSF','Alley','BsmtFinSF1', 'BsmtFinSF2','BsmtFinType1','BsmtFinType2','BsmtFullBath','BsmtHalfBath',

         'BsmtUnfSF','Fence','FireplaceQu','FullBath','GarageCars','GarageCond','GarageFinish','GarageQual','GarageType',

         'GarageYrBlt','GrLivArea','HalfBath','PoolQC', 'Condition1', 'Condition2', 'BedroomAbvGr', 'YrSold', 'YearBuilt', 

        'Cond1', 'Cond2', 'Electrical', 'Street', 'PavedDrive', 'SaleType', 'SaleCondition', 'OverallQual'], axis= 1, inplace= True)
df.drop(['LandContour','LandSlope', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'RoofStyle', 'RoofMatl',

        'MoSold','TotalBsmtSF', 'BldgType', 'BsmtExposure', 'Exterior1st', 'Exterior2nd', 'Foundation', 'Heating',

        'KitchenAbvGr','LotFrontage', 'LotShape', 'LotConfig', 'LowQualFinSF', 'MSZoning', 'MasVnrType', 'MiscFeature',

         'OverallCond', 'YearRemodAdd', 'Utilities'], axis= 1, inplace= True)
train_df= df[df['SalePrice'].notna()]

test_df= df[df['SalePrice'].isna()]
train_df.drop(train_df[train_df['TotalSqFt']>15000].index, inplace= True)

sns.scatterplot(x= train_df['TotalSqFt'], y= train_df['SalePrice'])
lr= LinearRegression()

skf= StratifiedKFold(n_splits= 10, shuffle= True)

result= cross_validate(lr, train_df[['Neighborhood', 'TotalSqFt']], train_df['SalePrice'], cv= skf, scoring= ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'])

print('Model 0b:\tMAE: %.8f\t\tMSE: %.6f\t\t\tR2: %.6f'%(result['test_neg_mean_absolute_error'].mean(), result['test_neg_mean_squared_error'].mean(), result['test_r2'].mean()))
lr= LinearRegression()

skf= StratifiedKFold(n_splits= 10, shuffle= True)

result= cross_validate(lr, train_df.drop(['SalePrice'], axis= 1), train_df['SalePrice'], cv= skf, scoring= ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error'])

print('Model 0b:\tMAE: %.8f\t\tMSE: %.6f\t\t\tR2: %.6f'%(result['test_neg_mean_absolute_error'].mean(), result['test_neg_mean_squared_error'].mean(), result['test_r2'].mean()))
train_df= train_df.astype(np.float64)

test_df= test_df.astype(np.float64)

lgbm = LGBMRegressor(objective='regression', 

       num_leaves=5, #was 3

       learning_rate=0.01, 

       n_estimators=11000, #8000

       max_bin=200, 

       bagging_fraction=0.75,

       bagging_freq=5, 

       bagging_seed=7,

       feature_fraction=0.4, # 'was 0.2'

)

lgbm.fit(train_df.drop(['SalePrice'], axis= 1), train_df['SalePrice'], eval_metric='rmse')

pred= lgbm.predict(test_df.drop(['SalePrice'], axis= 1))
lr= LinearRegression()

skf= StratifiedKFold(n_splits= 10, shuffle= True)

lr.fit(train_df.drop(['SalePrice'], axis= 1), train_df['SalePrice'])

pred= lr.predict(test_df.drop(['SalePrice'], axis= 1))

pred
sub= pd.DataFrame({

    "Id": test_df.index,

    "SalePrice": pred

})

sub.to_csv('houseprice1.csv', index= False)