import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import seaborn as sns

from scipy.stats import skew



% matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
df = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'], test.loc[:, 'MSSubClass':'SaleCondition']))
print(train.shape)

print(test.shape)

print(df.shape)
df.dtypes.value_counts()
object_feats = list(df.dtypes[df.dtypes == "object"].index)

print(len(object_feats))

print(object_feats)
int_feats = list(df.dtypes[df.dtypes == "int64"].index)

print(len(int_feats))

print(int_feats)
float_feats = list(df.dtypes[df.dtypes == "float64"].index)

print(len(float_feats))

print(float_feats)
na_counts = train.isnull().sum(axis=0)

print('number of features containing NA')

print('train: ', na_counts.loc[na_counts > 0].shape[0])

na_counts = test.isnull().sum(axis=0)

print('test: ', na_counts.loc[na_counts > 0].shape[0])

na_counts = df.isnull().sum(axis=0)

print('all: ', na_counts.loc[na_counts > 0].shape[0])

print(na_counts.loc[na_counts > 0])
num_feats = int_feats + float_feats

NA_num = [x for x in list(na_counts.loc[na_counts > 0].index) if x in num_feats]

NA_obj = [x for x in list(na_counts.loc[na_counts > 0].index) if x in object_feats]

print('number of numerical features with NA: ', len(NA_num), '\nnumber of object features with NA: ', len(NA_obj))
print(NA_num)
# Let's deal with numerical features first

# We might process some object features in the mean time, if they are about the same properties

# And we use a list to store the "btw processed" object features, if their NAs are completely eliminated

NA_obj_btw = []

# Also, we use a list to store the object features that are ordinal categorical

ordianl_obj = []
# LotFrontage

mean_LotFrontage = df.groupby('BldgType').LotFrontage.mean()

for x in list(mean_LotFrontage.index):

    df.loc[(df.LotFrontage.isnull()) & (df.BldgType==x), 'LotFrontage'] = mean_LotFrontage[x]
# MasVnrArea

df.loc[df.MasVnrType.isnull(), ['MasVnrType', 'MasVnrArea']]
df.MasVnrType.value_counts(dropna=False)
# We have to fill na correspondingly for both of them

df.loc[df.MasVnrArea.isnull(), 'MasVnrType'] = 'None'

df.loc[df.MasVnrArea.isnull(), 'MasVnrArea'] = 0
# and there is one REAL NaN in MasVnrType, row 1150 shown above

df.loc[df.MasVnrType.isnull(), 'MasVnrType'] = 'BrkFace'
NA_obj_btw.append('MasVnrType')
# features related to basement

base_feats_num = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

for x in base_feats_num:

    print(x, df.loc[df[x].isnull(), :].shape[0])
# maybe it's better to do with all basement features together

# features related to basement

base_feats_obj = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

for x in base_feats_obj:

    print(x, df.loc[df[x].isnull(), :].shape[0])
df.loc[df.BsmtFullBath.isnull(), base_feats_num+base_feats_obj]
# it's obvious that these two houses have no basement

df.loc[df.BsmtFullBath.isnull(), base_feats_num] = 0
df.loc[df.BsmtFinType1.isnull(), base_feats_num+base_feats_obj].head()
# as in the data description, most NaN means no basement

df.loc[df.BsmtCond.isnull(), base_feats_obj] = 'Without'
# Maybe there are some REAL NaNs

base_feats_obj = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

for x in base_feats_obj:

    print(x, df.loc[df[x].isnull(), :].shape[0])
df.loc[df.BsmtQual.isnull(), base_feats_num+base_feats_obj]
df.BsmtQual.mode()
# yes the BsmtQual measures the height of the basement, and there are two NaNs

df.loc[df.BsmtQual.isnull(), 'BsmtQual'] = 'TA'
df.loc[df.BsmtExposure.isnull(), base_feats_num+base_feats_obj].head()
df['BsmtUnfinishRatio'] = df.BsmtUnfSF / df.TotalBsmtSF

grouped = df.groupby('BsmtExposure')

grouped['BsmtUnfinishRatio'].mean().sort_values()
df.loc[df.BsmtExposure.isnull(), 'BsmtExposure'] = 'No'
df.loc[df.BsmtFinType2.isnull(), base_feats_num+base_feats_obj]
grouped = df.groupby('BsmtFinType2')

grouped['BsmtFinSF2'].mean()
df.loc[df.BsmtFinType2.isnull(), 'BsmtFinType2'] = 'ALQ'
# Maybe there are some real NaNs

base_feats = base_feats_num + base_feats_obj

for x in base_feats:

    print(x, df.loc[df[x].isnull(), :].shape[0])
base_feats_obj
NA_obj_btw = NA_obj_btw + ['BsmtQual', 'BsmtCond', 'BsmtFinType1']
# garage

# again, it's better to deal with all garage features in the mean time

garage_feats_num = ['GarageYrBlt', 'GarageCars', 'GarageArea']

garage_feats_obj = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for x in garage_feats_num:

    print(x, df.loc[df[x].isnull(), :].shape[0])
for x in garage_feats_obj:

    print(x, df.loc[df[x].isnull(), :].shape[0])
df.loc[df.GarageCars.isnull(), garage_feats_num+garage_feats_obj]
df.loc[df.GarageCars.isnull(), ['GarageCars', 'GarageArea']] = 0
df.loc[df.GarageType.isnull(), garage_feats_num+garage_feats_obj].head()
# as in the data description, NaN means no garage

df.loc[df.GarageType.isnull(), garage_feats_obj] = 'Without'
df.loc[df.GarageQual.isnull(), garage_feats_num+garage_feats_obj].head()
df.loc[df.GarageCars==1, ['GarageFinish', 'GarageQual', 'GarageCond']].mode()
df.loc[(df.GarageFinish.isnull())&(df.GarageCars==1), 'GarageFinish'] = 'Unf'

df.loc[(df.GarageQual.isnull())&(df.GarageCars==1), 'GarageQual'] = 'TA'

df.loc[(df.GarageCond.isnull())&(df.GarageCars==1), 'GarageCond'] = 'TA'
df.loc[(df.GarageFinish.isnull())&(df.GarageCars==0), 'GarageFinish'] = 'Without'

df.loc[(df.GarageQual.isnull())&(df.GarageCars==0), 'GarageQual'] = 'Without'

df.loc[(df.GarageCond.isnull())&(df.GarageCars==0), 'GarageCond'] = 'Without'
df.loc[df.GarageYrBlt.isnull(), 'GarageYrBlt'] = 0
garage_feats = garage_feats_num + garage_feats_obj

for x in garage_feats:

    print(x, df.loc[df[x].isnull(), :].shape[0])
NA_obj_btw = NA_obj_btw + garage_feats_obj
# let's check the outcome of the missing values processing

for x in NA_num:

    print(x, df.loc[df[x].isnull(), :].shape[0])
# OK, not bad, let's move on to object features

# let's have a look at what have been processed "BTW"

NA_obj_btw
NA_obj_new = [x for x in NA_obj if x not in NA_obj_btw]

print(NA_obj_new)
for x in NA_obj_new:

    print(x, df.loc[df[x].isnull(), :].shape[0])
# MSZoning

print(df.MSZoning.mode())

df.loc[df.MSZoning.isnull(), 'MSZoning'] = 'RL'
# Alley

# NaN means no alley

df.loc[df.Alley.isnull(), 'Alley'] = 'Without'
# Utilities

print(df.Utilities.mode())

df.loc[df.Utilities.isnull(), 'Utilities'] = 'AllPub'
# 'Exterior1st' and 'Exterior2nd'

df.loc[df.Exterior1st.isnull(), ['Exterior1st', 'Exterior2nd']]
print(df[['Exterior1st', 'Exterior2nd']].mode())
df.loc[df.Exterior1st.isnull(), ['Exterior1st', 'Exterior2nd']] = 'VinylSd'
# Electrical

df.Electrical.mode()

df.loc[df.Electrical.isnull(), 'Electrical'] = 'SBrkr'
# KitchenQual

df.KitchenQual.mode()

df.loc[df.KitchenQual.isnull(), 'KitchenQual'] = 'TA'
# Functional

df.Functional.mode()

df.loc[df.Functional.isnull(), 'Functional'] = 'Typ'
# FireplaceQu

# NaN means no fireplace

df.loc[df.FireplaceQu.isnull(), 'FireplaceQu'] = 'Without'
# PoolQC

# NaN means no pool

df.loc[df.PoolQC.isnull(), 'PoolQC'] = 'Without'
# Fence

# NaN means no fence

df.loc[df.Fence.isnull(), 'Fence'] = 'Without'
# MiscFeature

# NaN means no MiscFeature

df.loc[df.MiscFeature.isnull(), 'MiscFeature'] = 'Without'
# SaleType

df.SaleType.mode()

df.loc[df.SaleType.isnull(), 'SaleType'] = 'WD'
for x in NA_obj_new:

    print(x, df.loc[df[x].isnull(), :].shape[0])
# let's check if there is still NAs

na_counts = df.isnull().sum(axis=0)

print(na_counts.loc[na_counts > 0])
df.drop(['BsmtUnfinishRatio'], inplace=True, axis=1)
print('Before', len(object_feats))
print(object_feats)
ordinal_words = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
ordinal_feats = []

for x in object_feats:

    flag = 0

    for i in ordinal_words:

        if df.loc[df[x].str.contains(i, case=True), x].shape[0] > 0:

            flag = flag + 1

    if flag >= 2:

        ordinal_feats.append(x)
print(ordinal_feats)
for x in ordinal_feats:

    print(x, df[x].unique(), len(df[x].unique()))
ordinal_dict = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'Without':0}
ordinal_num_feats = []

for x in ordinal_feats:

    df[x+'_num'] = df[x].map(ordinal_dict)

    ordinal_num_feats.append(x+'_num')
df.drop(ordinal_feats, axis=1, inplace=True)
ordinal_obj_feats = ['PavedDrive', 'Functional', 'CentralAir', 'Fence', 'Utilities']



df['Functional_num'] = df.Functional.replace({'Typ': 6,

                                            'Min1': 5,

                                            'Min2': 5,

                                            'Mod': 4,

                                            'Maj1': 3,

                                            'Maj2': 3,

                                            'Sev': 2,

                                            'Sal': 1})

df['PavedDrive_num'] = df.PavedDrive.replace({'Y': 3, 'P': 2, 'N': 1})

df['CentralAir_num'] = df.CentralAir.replace({'Y': 1, 'N': 0})

df['Fence_num'] = df.CentralAir.replace({'GdPrv': 2, 'GdWo': 2, 'MnPrv': 1, 'MnWw': 1, 'NoFence': 0})

df['Utilities_num'] = df.Utilities.replace({'AllPub': 1, 'NoSewr': 0, 'NoSeWa': 0, 'ELO': 0})
toDrop_feats = []

toDrop_feats = toDrop_feats + ordinal_obj_feats
df.shape
toDrop_feats = []
# some maybe newer

df['newer_dwelling'] = df.MSSubClass.replace({20: 1, 30: 0, 40: 0, 45: 0, 50: 0, 60: 1, 70: 0, 75: 0,\

                                              80: 0, 85: 0, 90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})



# but we still keep the original feature and transform it into categorical

map_MSS = {x: 'Subclass_'+str(x) for x in df.MSSubClass.unique()}

df['MSSubClass'] = df.MSSubClass.replace(map_MSS)
# it maybe helpful to transform some quality style features into binary

quality_feats = ['OverallQual', 'OverallCond', 'ExterQual_num', 'ExterCond_num', 'BsmtCond_num',\

                 'GarageQual_num', 'GarageCond_num', 'KitchenQual_num']

toDrop_feats = toDrop_feats + quality_feats



all_data = df

overall_poor_qu = all_data.OverallQual.copy()

overall_poor_qu = 5 - overall_poor_qu

overall_poor_qu[overall_poor_qu<0] = 0

overall_poor_qu.name = 'overall_poor_qu'



overall_good_qu = all_data.OverallQual.copy()

overall_good_qu = overall_good_qu - 5

overall_good_qu[overall_good_qu<0] = 0

overall_good_qu.name = 'overall_good_qu'



overall_poor_cond = all_data.OverallCond.copy()

overall_poor_cond = 5 - overall_poor_cond

overall_poor_cond[overall_poor_cond<0] = 0

overall_poor_cond.name = 'overall_poor_cond'



overall_good_cond = all_data.OverallCond.copy()

overall_good_cond = overall_good_cond - 5

overall_good_cond[overall_good_cond<0] = 0

overall_good_cond.name = 'overall_good_cond'



exter_poor_qu = all_data.ExterQual_num.copy()

exter_poor_qu[exter_poor_qu<3] = 1

exter_poor_qu[exter_poor_qu>=3] = 0

exter_poor_qu.name = 'exter_poor_qu'



exter_good_qu = all_data.ExterQual_num.copy()

exter_good_qu[exter_good_qu<=3] = 0

exter_good_qu[exter_good_qu>3] = 1

exter_good_qu.name = 'exter_good_qu'



exter_poor_cond = all_data.ExterCond_num.copy()

exter_poor_cond[exter_poor_cond<3] = 1

exter_poor_cond[exter_poor_cond>=3] = 0

exter_poor_cond.name = 'exter_poor_cond'



exter_good_cond = all_data.ExterCond_num.copy()

exter_good_cond[exter_good_cond<=3] = 0

exter_good_cond[exter_good_cond>3] = 1

exter_good_cond.name = 'exter_good_cond'



bsmt_poor_cond = all_data.BsmtCond_num.copy()

bsmt_poor_cond[bsmt_poor_cond<3] = 1

bsmt_poor_cond[bsmt_poor_cond>=3] = 0

bsmt_poor_cond.name = 'bsmt_poor_cond'



bsmt_good_cond = all_data.BsmtCond_num.copy()

bsmt_good_cond[bsmt_good_cond<=3] = 0

bsmt_good_cond[bsmt_good_cond>3] = 1

bsmt_good_cond.name = 'bsmt_good_cond'



garage_poor_qu = all_data.GarageQual_num.copy()

garage_poor_qu[garage_poor_qu<3] = 1

garage_poor_qu[garage_poor_qu>=3] = 0

garage_poor_qu.name = 'garage_poor_qu'



garage_good_qu = all_data.GarageQual_num.copy()

garage_good_qu[garage_good_qu<=3] = 0

garage_good_qu[garage_good_qu>3] = 1

garage_good_qu.name = 'garage_good_qu'



garage_poor_cond = all_data.GarageCond_num.copy()

garage_poor_cond[garage_poor_cond<3] = 1

garage_poor_cond[garage_poor_cond>=3] = 0

garage_poor_cond.name = 'garage_poor_cond'



garage_good_cond = all_data.GarageCond_num.copy()

garage_good_cond[garage_good_cond<=3] = 0

garage_good_cond[garage_good_cond>3] = 1

garage_good_cond.name = 'garage_good_cond'



kitchen_poor_qu = all_data.KitchenQual_num.copy()

kitchen_poor_qu[kitchen_poor_qu<3] = 1

kitchen_poor_qu[kitchen_poor_qu>=3] = 0

kitchen_poor_qu.name = 'kitchen_poor_qu'



kitchen_good_qu = all_data.KitchenQual_num.copy()

kitchen_good_qu[kitchen_good_qu<=3] = 0

kitchen_good_qu[kitchen_good_qu>3] = 1

kitchen_good_qu.name = 'kitchen_good_qu'



df_qual = pd.concat((overall_poor_qu, overall_good_qu, overall_poor_cond, overall_good_cond, exter_poor_qu,

                     exter_good_qu, exter_poor_cond, exter_good_cond, bsmt_poor_cond, bsmt_good_cond, garage_poor_qu,

                     garage_good_qu, garage_poor_cond, garage_good_cond, kitchen_poor_qu, kitchen_good_qu), axis=1)

df = pd.concat((df, df_qual), axis=1)
# for some categorical features, certain levels may imply better quality

toDrop_feats = toDrop_feats + ['MasVnrType', 'SaleCondition', 'Neighborhood']



map_Mas = {'BrkCmn': 1, 'BrkFace': 1, 'CBlock': 1, 'Stone': 1, 'None': 0}

MasVnrType_Any = all_data.MasVnrType.replace(map_Mas)

MasVnrType_Any.name = 'MasVnrType_Any'



map_Sale = {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0}

SaleCondition_PriceDown = all_data.SaleCondition.replace(map_Sale)

SaleCondition_PriceDown.name = 'SaleCondition_PriceDown'



neigh_good_feats = ['NridgHt', 'Crawfor', 'StoneBr', 'Somerst', 'NoRidge']

df['Neighborhood_good'] = 0

df.loc[df.Neighborhood.isin(neigh_good_feats), 'Neighborhood_good'] = 1



df = pd.concat((df, MasVnrType_Any, SaleCondition_PriceDown), axis=1)
# Monthes with the lagest number of deals may be significant

df['season'] = df.MoSold.replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})
# Numer month is not significant, it maybe helpful to transform them into object feature

map_Mo = {1: 'Yan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Avg', 9: 'Sep', 10: 'Oct', \

 11: 'Nov', 12: 'Dec'}

df = df.replace({'MoSold': map_Mo})
# some features about years

# sold at the same year as bulit

df['SoldImmediate'] = 0

df.loc[(df.YrSold == df.YearBuilt), 'SoldImmediate'] = 1



# reconstructed since first built

df['Recon'] = 0

df.loc[(df.YearBuilt < df.YearRemodAdd), 'Recon'] = 1



# reconstructed after sold

df['ReconAfterSold'] = 0

df.loc[(df.YrSold < df.YearRemodAdd), 'ReconAfterSold'] = 1



# reconstructed the same year as sold

df['ReconEqualSold'] = 0

df.loc[(df.YrSold == df.YearRemodAdd), 'ReconEqualSold'] = 1
# Years are too much, it maybe helpful to devide them into groups, and delete the original ones

year_map = pd.concat(pd.Series('YearGroup' + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))

df.YearBuilt = df.YearBuilt.map(year_map)

df.YearRemodAdd = df.YearRemodAdd.map(year_map)

df.GarageYrBlt = df.GarageYrBlt.map(year_map)

df.loc[df.GarageYrBlt==0, 'GarageYrBlt'] = 'NoGarage'
print(len(toDrop_feats), df.shape)
t = ['PavedDrive', 'Functional', 'CentralAir', 'Fence', 'OverallQual', 'OverallCond', 'ExterQual_num', 'ExterCond_num', 'BsmtCond_num', 'GarageQual_num', 'GarageCond_num', 'KitchenQual_num', 'MasVnrType', 'SaleCondition', 'Neighborhood']
[x for x in t if x not in toDrop_feats]
temp = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'CentralAir', 'Electrical', 'X1stFlrSF', 'X2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'X3SsnPorch', 'ScreenPorch', 'PoolArea', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition', 'ExterQual_num', 'ExterCond_num', 'BsmtQual_num', 'BsmtCond_num', 'HeatingQC_num', 'KitchenQual_num', 'FireplaceQu_num', 'GarageQual_num', 'GarageCond_num', 'PoolQC_num', 'Functional_num', 'PavedDrive_num', 'CentralAir_num', 'Fence_num', 'Utilities_num', 'newer_dwelling', 'overall_poor_qu', 'overall_good_qu', 'overall_poor_cond', 'overall_good_cond', 'exter_poor_qu', 'exter_good_qu', 'exter_poor_cond', 'exter_good_cond', 'bsmt_poor_cond', 'bsmt_good_cond', 'garage_poor_qu', 'garage_good_qu', 'garage_poor_cond', 'garage_good_cond', 'kitchen_poor_qu', 'kitchen_good_qu', 'MasVnrType_Any', 'SaleCondition_PriceDown', 'Neighborhood_good', 'season', 'SoldImmediate', 'Recon', 'ReconAfterSold', 'ReconEqualSold']
from sklearn.svm import SVC

svm = SVC(C=100)

# price categories

pc = pd.Series(np.zeros(train.shape[0]))

pc[:] = 'pc1'

pc[train.SalePrice >= 150000] = 'pc2'

pc[train.SalePrice >= 220000] = 'pc3'

columns_for_pc = ['Exterior1st', 'Exterior2nd', 'RoofMatl', 'Condition1', 'Condition2', 'BldgType']

X_t = pd.get_dummies(train.loc[:, columns_for_pc])

svm.fit(X_t, pc)

pc_pred = svm.predict(X_t)



price_category = pd.DataFrame(np.zeros((df.shape[0],1)), columns=['pc'], index=df.index)

X_t = pd.get_dummies(df.loc[:, columns_for_pc])

pc_pred = svm.predict(X_t)

price_category[pc_pred=='pc2'] = 1

price_category[pc_pred=='pc3'] = 2



toDrop_feats = toDrop_feats + columns_for_pc

df = pd.concat((df, price_category), axis=1)
numeric_feats = df.dtypes[df.dtypes != "object"].index

t = df[numeric_feats].quantile(.95)

use_max_scater = t[t == 0].index

use_95_scater = t[t != 0].index

df[use_max_scater] = df[use_max_scater]/df[use_max_scater].max()

df[use_95_scater] = df[use_95_scater]/df[use_95_scater].quantile(.95)
df_new = df.copy()

from scipy.stats import skew

numeric_feats = list(df_new.dtypes[df_new.dtypes != "object"].index)

feat_skewness = df_new[numeric_feats].apply(lambda x: skew(x.dropna()))

skewed_feats = list(feat_skewness[feat_skewness > 0.75].index)

df_new[skewed_feats] = np.log1p(df_new[skewed_feats])



df = df_new.copy()
# print(len(toDrop_feats), toDrop_feats)

# df.drop(toDrop_feats, inplace=True, axis=1)
df_new = pd.get_dummies(df)

print(df.shape, df_new.shape)
from itertools import product, chain



def poly(X):

    areas = ['LotArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'BsmtUnfSF']

    t = chain(df_qual.axes[1].get_values(), 

              ['OverallQual_num', 'OverallCond_num', 'ExterQual_num', 'ExterCond_num', 'BsmtCond_num', \

               'GarageQual_num', 'GarageCond_num', 'KitchenQual_num', 'HeatingQC_num', \

               'MasVnrType_Any', 'SaleCondition_PriceDown', 'Recon',

               'ReconAfterSold', 'SoldImmediate'])

    for a, t in product(areas, t):

        x = X.loc[:, [a, t]].prod(1)

        x.name = a + '_' + t

        yield x



XP = pd.concat(poly(df_new), axis=1)

df_new = pd.concat((df_new, XP), axis=1)



df_new.shape
df_final = df_new.copy()



X_train = df_final[:train.shape[0]]

print(X_train.shape)

y_train = np.log1p(train.SalePrice)

print(y_train.shape)

X_test = df_final[test.shape[0]+1:]

print(X_test.shape)
import warnings

warnings.filterwarnings('ignore')



from sklearn.cross_validation import cross_val_score

from sklearn.metrics import make_scorer, mean_squared_error



def rmse_cv(model, X_train, y_train):

    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=10))

    return(rmse)
from sklearn.linear_model import Lasso

#alphas = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.000075, 0.0001, 0.00025, 0.0005]

alphas = [0.00001, 0.00005, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002]

cv_lasso = [rmse_cv(Lasso(alpha = alpha, max_iter=100), X_train, y_train).mean() for alpha in alphas];

result = pd.Series(cv_lasso, index = alphas)

result.plot()

result.min()
lasso_test = Lasso(alpha=0.00005, max_iter=1000).fit(X_train, y_train)

y_test = lasso_test.predict(X_train)
dfy = pd.DataFrame({"id": train.Id, "y_real": y_train, "y_test": y_test, "residual": y_test - y_train},\

                   columns=['id', 'y_real', 'y_test', 'residual'])



outliers_id = np.array([524, 1299])

outliers_id = outliers_id - 1 # id starts with 1, index starts with 0
ax = dfy.plot(x='y_real', y='y_test', kind='scatter')

ax.scatter(dfy.iloc[outliers_id, 1], dfy.iloc[outliers_id, 2], c='r')
ax = dfy.plot(x='y_real', y='residual', kind='scatter')

ax.scatter(dfy.iloc[outliers_id, 1], dfy.iloc[outliers_id, 3], c='r')
# this come from iterational model improvment. I was trying to understand why the model gives to the two points much better price

x_plot = X_train.loc[X_train['SaleCondition_Partial']==1, 'GrLivArea']

y_plot = y_train[X_train['SaleCondition_Partial']==1]

ax = plt.scatter(x_plot, y_plot)
outliers_id = np.array([524, 1299])



outliers_id = outliers_id - 1 # id starts with 1, index starts with 0

X_train = X_train.drop(outliers_id, axis=0)

y_train = y_train.drop(outliers_id, axis=0)



from sklearn.linear_model import Lasso

#alphas = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.000075, 0.0001, 0.00025, 0.0005]

alphas = [0.00001, 0.00005, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002]

cv_lasso = [rmse_cv(Lasso(alpha = alpha, max_iter=100), X_train, y_train).mean() for alpha in alphas];

result2 = pd.Series(cv_lasso, index = alphas)

df_result = pd.concat((result, result2), axis=1)

df_result.columns= ['original', 'without outliers']

df_result.plot()

# result.min()
# this come from iterational model improvment. I was trying to understand why the model gives to the two points much better price

x_plot = X_train.loc[X_train['SaleCondition_Partial']==1, 'GrLivArea']

y_plot = y_train[X_train['SaleCondition_Partial']==1]

plt.scatter(x_plot, y_plot)
lasso = Lasso(alpha=0.00075, max_iter=1000).fit(X_train, y_train)

y_pred = lasso.predict(X_test)

y_pred.shape
solution = pd.DataFrame({"id": test.Id, "SalePrice": np.expm1(y_pred)}, columns=['id', 'SalePrice'])

solution.to_csv("lasso_sol.csv", index = False)