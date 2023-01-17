# standard
import pandas as pd
import numpy as np

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
# ---------- DF IMPORT -------------
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
combine = [df_train, df_test]
df_train.name = "Train"
df_test.name = "Test"
pd.set_option("display.max_columns", None)
#Function to check how a categorical variable segments the target
def segm_target(var):
    count = df_train[[var, "SalePrice"]].groupby([var], as_index=True).count()
    count.columns = ['Count']
    mean = df_train[[var, "SalePrice"]].groupby([var], as_index=True).mean()
    mean.columns = ['Mean']
    ma = df_train[[var, "SalePrice"]].groupby([var], as_index=True).max()
    ma.columns = ['Max']
    mi = df_train[[var, "SalePrice"]].groupby([var], as_index=True).min()
    mi.columns = ['Min']
    median = df_train[[var, "SalePrice"]].groupby([var], as_index=True).median()
    median.columns = ['Median']
    std = df_train[[var, "SalePrice"]].groupby([var], as_index=True).std()
    std.columns = ['Std']
    df = pd.concat([count, mean, ma, mi, median, std], axis=1)
    fig, ax = plt.subplots(1,2, figsize=(12, 5))
    sns.boxplot(var,"SalePrice",data=df_train, ax=ax[0])
    sns.boxplot(var,"LogPrice",data=df_train, ax=ax[1])
    fig.show()
    return df

#Function to check correlation with target of a number of numerical features
def corr_target(*arg):
    print(df_train[[f for f in arg]].corr())
    num = len(arg) - 1
    rows = int(num/2) + (num % 2 > 0)
    arg = list(arg)
    target = arg[-1]
    del arg[-1]
    y = df_train[target]
    fig, ax = plt.subplots(rows, 2, figsize=(12, 5 * (rows)))
    i = 0
    j = 0
    for feat in arg:
        x = df_train[feat]
        if (rows > 1):
            sns.regplot(x=x, y=y, ax=ax[i][j])
            j = (j+1)%2
            i = i + 1 - j
        else:
            sns.regplot(x=x, y=y, ax=ax[i])
            i = i+1
    fig.show()
# Cell containing all the pre-processing performed in the previous notebook by just following the documentation
# (and some common sense)
for df in combine:
    #LotFrontage
    df.loc[df.LotFrontage.isnull(), 'LotFrontage'] = 0
    #Alley
    df.loc[df.Alley.isnull(), 'Alley'] = "NoAlley"
    #MSSubClass
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    #MissingBasement
    fil = ((df.BsmtQual.isnull()) & (df.BsmtCond.isnull()) & (df.BsmtExposure.isnull()) &
          (df.BsmtFinType1.isnull()) & (df.BsmtFinType2.isnull()))
    fil1 = ((df.BsmtQual.notnull()) | (df.BsmtCond.notnull()) | (df.BsmtExposure.notnull()) |
          (df.BsmtFinType1.notnull()) | (df.BsmtFinType2.notnull()))
    df.loc[fil1, 'MisBsm'] = 0
    df.loc[fil, 'MisBsm'] = 1
    #BsmtQual
    df.loc[fil, 'BsmtQual'] = "NoBsmt" #missing basement
    #BsmtCond
    df.loc[fil, 'BsmtCond'] = "NoBsmt" #missing basement
    #BsmtExposure
    df.loc[fil, 'BsmtExposure'] = "NoBsmt" #missing basement
    #BsmtFinType1
    df.loc[fil, 'BsmtFinType1'] = "NoBsmt" #missing basement
    #BsmtFinType2
    df.loc[fil, 'BsmtFinType2'] = "NoBsmt" #missing basement
    #FireplaceQu
    df.loc[(df.Fireplaces == 0) & (df.FireplaceQu.isnull()), 'FireplaceQu'] = "NoFire" #missing
    #MisGarage
    fil = ((df.GarageYrBlt.isnull()) & (df.GarageType.isnull()) & (df.GarageFinish.isnull()) &
          (df.GarageQual.isnull()) & (df.GarageCond.isnull()))
    fil1 = ((df.GarageYrBlt.notnull()) | (df.GarageType.notnull()) | (df.GarageFinish.notnull()) |
          (df.GarageQual.notnull()) | (df.GarageCond.notnull()))
    df.loc[fil1, 'MisGarage'] = 0
    df.loc[fil, 'MisGarage'] = 1
    #GarageYrBlt
    df.loc[df.GarageYrBlt > 2200, 'GarageYrBlt'] = 2007 #correct mistake
    df.loc[fil, 'GarageYrBlt'] = 0
    #GarageType
    df.loc[fil, 'GarageType'] = "NoGrg" #missing garage
    #GarageFinish
    df.loc[fil, 'GarageFinish'] = "NoGrg" #missing
    #GarageQual
    df.loc[fil, 'GarageQual'] = "NoGrg" #missing
    #GarageCond
    df.loc[fil, 'GarageCond'] = "NoGrg" #missing
    #Fence
    df.loc[df.Fence.isnull(), 'Fence'] = "NoFence" #missing fence
# We know already we will need to deal with homoscedasticity (written without checking the real spelling)
df_train["LogPrice"] = np.log1p(df_train["SalePrice"])
df_train.LogPrice.hist(bins = 100)
corr_target('LotFrontage', 'LotArea', 'LogPrice')
# I am curious to see if the sum of the two is better (using the sqrt of the Area because it makes more sense)
df_train['LotFrontTot'] = df_train['LotFrontage'] + np.sqrt(df_train['LotArea'])

print(df_train[['LotFrontTot', 'LotFrontage','LotArea']].corr())
print("_"*40)
corr_target('LotFrontTot', 'LogPrice')
var = "MSZoning"
segm_target(var)
# Trying a grouping to handle sparse classes
df_train.loc[(df_train.MSZoning == 'RH') | (df_train.MSZoning == 'RM'), 'MSZoningGroup'] = 'ResMedHig'
df_train.loc[(df_train.MSZoning == 'FV'), 'MSZoningGroup'] = 'Vil'
df_train.loc[(df_train.MSZoning == 'RL')| (df_train.MSZoning == 'C (all)'), 'MSZoningGroup'] = 'ResLowCom'
var = 'MSZoningGroup'
segm_target(var)
var = "Street"
segm_target(var)
var = "Alley"
segm_target(var)
df_train.loc[(df_train.Alley == 'Grvl') | (df_train.Alley == 'Pave'), 'AlleyGroup'] = 'Alley'
df_train.loc[df_train.Alley == 'NoAlley', 'AlleyGroup'] = 'NoAlley'

var = 'AlleyGroup'
segm_target(var)
var = "LotShape"
segm_target(var)
irr = ['IR1', 'IR2', 'IR3']
df_train.loc[(df_train.LotShape.isin(irr)), 'LotShapeGroup'] = 'Irreg'
df_train.loc[df_train.LotShape == 'Reg', 'LotShapeGroup'] = 'Reg'

var = 'LotShapeGroup'
segm_target(var)
var = "LandContour"
segm_target(var)
irr = ['Bnk', 'Low', 'HLS']
df_train.loc[(df_train.LandContour.isin(irr)), 'LandContourGroup'] = 'NotLvl'
df_train.loc[df_train.LandContour == 'Lvl', 'LandContourGroup'] = 'Lvl'

var = 'LandContourGroup'
segm_target(var)
var = "LotConfig"
segm_target(var)
df_train.loc[(df_train.LotConfig == 'FR2') | (df_train.LotConfig == 'FR3'), 'LotConfigGroup'] = 'FR'
df_train.loc[df_train.LotConfig == 'Corner', 'LotConfigGroup'] = 'Corner'
df_train.loc[df_train.LotConfig == 'CulDSac', 'LotConfigGroup'] = 'CulDSac'
df_train.loc[df_train.LotConfig == 'Inside', 'LotConfigGroup'] = 'Inside'

var = 'LotConfigGroup'
segm_target(var)
var = "LandSlope"
segm_target(var)
df_train.loc[(df_train.LandSlope == 'Mod') | (df_train.LandSlope == 'Sev'), 'LandSlopeGroup'] = 'NonGlt'
df_train.loc[df_train.LandSlope == 'Gtl', 'LandSlopeGroup'] = 'Gtl'

var = 'LandSlopeGroup'
segm_target(var)
var = "Neighborhood"
segm_target(var)
var = "Condition1"
segm_target(var)
ArtFee = ['Artery', 'Feedr']
stat = ['RRAe', 'RRAn', 'RRNe', 'RRNn']
pos = ['PosA', 'PosN']
df_train.loc[(df_train.Condition1.isin(ArtFee)), 'Condition1Group'] = 'ArtFee'
df_train.loc[(df_train.Condition1.isin(stat)), 'Condition1Group'] = 'Station'
df_train.loc[(df_train.Condition1.isin(pos)), 'Condition1Group'] = 'Station'
df_train.loc[df_train.Condition1 == 'Norm', 'Condition1Group'] = 'Norm'

var = 'Condition1Group'
segm_target(var)
var = "Condition2"
segm_target(var)
pd.crosstab(df_train['LotShapeGroup'], df_train['LotConfigGroup'])
pd.crosstab(df_train['Condition1Group'], df_train['LotShapeGroup'])
g = sns.FacetGrid(df_train, col='LotConfigGroup', hue='LotShapeGroup')
g.map(plt.hist, 'LogPrice', alpha= 0.3, bins=20)
g.add_legend()
g = sns.FacetGrid(df_train, col='Condition1Group', hue='LotShapeGroup', size = 5)
g.map(plt.hist, 'LogPrice', alpha= 0.3, bins=20)
g.add_legend()
corr_target('OverallQual', 'OverallCond','SalePrice')
corr_target('OverallQual', 'OverallCond','LogPrice')
corr_target('YearBuilt', 'YearRemodAdd', 'SalePrice')
x = df_train['YearBuilt']
x1 = df_train['YearRemodAdd']
y = df_train['LogPrice']

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0], x_estimator = np.mean)
sns.regplot(x=x1, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()
# let's take the most recent year between build and remod
df_train['YearMostRec'] = df_train[['YearBuilt', 'YearRemodAdd']].apply(np.max, axis=1)

x = df_train['YearMostRec']
y = df_train['LogPrice']

print(df_train[['YearMostRec', 'LogPrice']].corr())
fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()
var = "MSSubClass"
segm_target(var)
var = "BldgType"
segm_target(var)
df_train.loc[(df_train.BldgType == '2fmCon') | (df_train.BldgType == 'Duplex'), 'BldTypeGroup'] = '2FamDup'
df_train.loc[(df_train.BldgType == 'Twnhs') | (df_train.BldgType == 'TwnhsE'), 'BldTypeGroup'] = 'Twnhs+E'
df_train.loc[(df_train.BldgType == '1Fam'), 'BldTypeGroup'] = '1Fam'
var = 'BldTypeGroup'
segm_target(var)
var = "HouseStyle"
segm_target(var)
onepl = ['1.5Fin', '1.5Unf']
twopl = ['2.5Fin', '2.5Unf', '2Story']
spl = ['SFoyer', 'SLvl']
df_train.loc[df_train.HouseStyle.isin(onepl), 'HStyleGroup'] = '1.5'
df_train.loc[df_train.HouseStyle.isin(twopl), 'HStyleGroup'] = '2plus'
df_train.loc[df_train.HouseStyle.isin(spl), 'HStyleGroup'] = 'Split'
df_train.loc[df_train.HouseStyle == '1Story', 'HStyleGroup'] = "1Story"
var = 'HStyleGroup'
segm_target(var)
pd.crosstab(df_train["HStyleGroup"],df_train['OverallQual'])
g = sns.FacetGrid(df_train, hue="HStyleGroup", size = 5)
g.map(plt.hist, 'OverallQual', alpha= 0.5, bins=10)
g.add_legend()
pd.crosstab(df_train["BldTypeGroup"],df_train['OverallQual'])
g = sns.FacetGrid(df_train, hue="BldTypeGroup", size = 5)
g.map(plt.hist, 'OverallQual', alpha= 0.5, bins=10)
g.add_legend()
g = sns.FacetGrid(df_train, col="HStyleGroup", hue="BldTypeGroup")
g.map(plt.hist, 'OverallQual', alpha= 0.3, bins=10)
g.add_legend()
g = sns.FacetGrid(df_train, hue="BldTypeGroup", size= 7)
g.map(plt.hist, 'YearBuilt', alpha= 0.3, bins=50)
g.add_legend()
g = sns.FacetGrid(df_train, hue="BldTypeGroup", size = 7)
g.map(plt.scatter, 'YearBuilt', 'LogPrice', edgecolor="w")
g.add_legend()
g = sns.FacetGrid(df_train, hue="HStyleGroup", size=7)
g.map(plt.hist, 'YearBuilt', alpha= 0.3, bins=50)
g.add_legend()
x = df_train['MasVnrArea']
y = df_train['SalePrice']

sns.regplot(x=x, y=y)
x = df_train['MasVnrArea']
y = df_train['LogPrice']

sns.regplot(x=x, y=y)
var = "Foundation"
segm_target(var)
fancy = ['BrkTil', 'Stone', 'Wood']
cement = ['PConc', 'Slab']
df_train.loc[df_train.Foundation.isin(fancy), 'FoundGroup'] = 'Fancy'
df_train.loc[df_train.Foundation.isin(cement), 'FoundGroup'] = 'Cement'
df_train.loc[df_train.Foundation == 'CBlock', 'FoundGroup'] = 'Cider'

var = "FoundGroup"
segm_target(var)
var = "RoofStyle"
segm_target(var)
nogable = ['Flat', 'Gambrel', 'Hip', 'Mansard', 'Shed']
df_train.loc[df_train.RoofStyle.isin(nogable), 'RoofStyleGroup'] = 'NoGable'
df_train.loc[df_train.RoofStyle == 'Gable', 'RoofStyleGroup'] = 'Gable'

var = "RoofStyleGroup"
segm_target(var)
var = "RoofMatl"
segm_target(var)
var = "Exterior1st"
segm_target(var)
other = ['Stucco', 'ImStucc', 'CemntBd', 'AsbShng', 'AsphShn', 'CBlock', 'Stone']
wood = ['Wd Sdng', 'WdShing', 'Plywood']

df_train.loc[df_train.Exterior1st.isin(other), 'Ext1Group'] = 'Other'
df_train.loc[df_train.Exterior1st.isin(wood), 'Ext1Group'] = 'Wood'
df_train.loc[df_train.Exterior1st == 'MetalSd', 'Ext1Group'] = 'MetalSd'
df_train.loc[df_train.Exterior1st == 'HdBoard', 'Ext1Group'] = 'HdBoard'

var = "Ext1Group"
segm_target(var)
var = "Exterior2nd"
segm_target(var)
other = ['Stucco', 'ImStucc', 'BrkFace', 'Brk Cmn', 'CmentBd', 'AsbShng', 
        'AsphShn', 'CBlock', 'Stone', 'Other']
wood = ['Wd Sdng', 'Wd Shng']

df_train.loc[df_train.Exterior2nd.isin(other), 'Ext2Group'] = 'Other'
df_train.loc[df_train.Exterior2nd.isin(wood), 'Ext2Group'] = 'Wood'
df_train.loc[df_train.Exterior2nd == 'MetalSd', 'Ext2Group'] = 'MetalSd'
df_train.loc[df_train.Exterior2nd == 'HdBoard', 'Ext2Group'] = 'HdBoard'

var = "Ext2Group"
segm_target(var)
var = "MasVnrType"
segm_target(var)
df_train.loc[df_train.MasVnrType == 'None', 'MasTypeGroup'] = 'None'
df_train.loc[df_train.MasVnrType == 'Stone', 'MasTypeGroup'] = 'Stone'
df_train.loc[(df_train.MasVnrType == 'BrkCmn') | (df_train.MasVnrType == 'BrkFace'), 'MasTypeGroup'] = 'Bricks'

var = "MasTypeGroup"
segm_target(var)
var = "ExterQual"
segm_target(var)
df_train.loc[df_train.ExterQual == 'Fa', 'ExtQuGroup'] = 1
df_train.loc[df_train.ExterQual == 'TA', 'ExtQuGroup'] = 1
df_train.loc[df_train.ExterQual == 'Gd', 'ExtQuGroup'] = 2
df_train.loc[df_train.ExterQual == 'Ex', 'ExtQuGroup'] = 3

x = df_train['ExtQuGroup']
y = df_train['LogPrice']
print(df_train[['ExtQuGroup','SalePrice' ]].corr())

sns.regplot(x = x, y = y, x_estimator = np.mean)
var = "ExterCond"
segm_target(var)
df_train.loc[df_train.ExterCond == 'Po', 'ExtCoGroup'] = 1
df_train.loc[df_train.ExterCond == 'Fa', 'ExtCoGroup'] = 1
df_train.loc[df_train.ExterCond == 'TA', 'ExtCoGroup'] = 1
df_train.loc[df_train.ExterCond == 'Gd', 'ExtCoGroup'] = 2
df_train.loc[df_train.ExterCond == 'Ex', 'ExtCoGroup'] = 2

x = df_train['ExtCoGroup']
y = df_train['LogPrice']
print(df_train[['ExtCoGroup','SalePrice']].corr())

sns.regplot(x = x, y = y, x_estimator = np.mean)
x = df_train['ExtQuGroup']
y = df_train['MasVnrArea']

print(df_train[['ExtQuGroup', 'MasVnrArea']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)
fig.show()
g = sns.FacetGrid(df_train, hue='MasTypeGroup', size = 5)
g.map(plt.hist, 'MasVnrArea', alpha= 0.5, bins=10)
g.add_legend()
pd.crosstab(df_train['MasTypeGroup'], df_train['ExtQuGroup'])
pd.crosstab(df_train['MasTypeGroup'], df_train['FoundGroup'])
g = sns.FacetGrid(df_train, col='MasTypeGroup', hue='FoundGroup')
g.map(plt.hist, 'LogPrice', alpha= 0.3, bins=50)
g.add_legend()
corr_target('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'LogPrice')
# I am interested in the total of finished SF
df_train['BsmtFinTotSF'] = df_train['BsmtFinSF1'] + df_train['BsmtFinSF2']

print(df_train[['BsmtFinTotSF', 'LogPrice']].corr())

x = df_train['BsmtFinTotSF']
y = df_train['LogPrice']

sns.regplot(x = x, y = y)
# What about the percentage of unfinished SF
df_train['BsmtPercUnf'] = df_train['BsmtUnfSF'] / df_train['TotalBsmtSF']
df_train['BsmtPercUnf'].fillna(1) #37 missing basements are actually complete

print(df_train[['BsmtPercUnf', 'LogPrice']].corr())

x = df_train['BsmtPercUnf']
y = df_train['LogPrice']

sns.regplot(x = x, y = y)
corr_target('BsmtFullBath', 'BsmtHalfBath', 'LogPrice')
x = df_train['BsmtFullBath']
x1 = df_train['BsmtHalfBath']
y = df_train['LogPrice']

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0], x_estimator = np.mean)
sns.regplot(x=x1, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()
df_train['BsmtBath'] = 0
df_train.loc[(df_train['BsmtFullBath'] > 0) | (df_train['BsmtHalfBath'] > 0), 'BsmtBath'] = 1

var = 'BsmtBath'
segm_target(var)
var = "BsmtQual"
segm_target(var)
df_train.loc[df_train.BsmtQual == 'NoBsmt', 'BsmtQuGroup'] = 0
df_train.loc[df_train.BsmtQual == 'Fa', 'BsmtQuGroup'] = 1
df_train.loc[df_train.BsmtQual == 'TA', 'BsmtQuGroup'] = 4
df_train.loc[df_train.BsmtQual == 'Gd', 'BsmtQuGroup'] = 10
df_train.loc[df_train.BsmtQual == 'Ex', 'BsmtQuGroup'] = 21

x = df_train['BsmtQuGroup']
y = df_train['LogPrice']
print(df_train[['BsmtQuGroup','SalePrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)
var = "BsmtCond"
segm_target(var)
var = "BsmtExposure"
segm_target(var)
df_train.loc[df_train.BsmtExposure == 'NoBsmt', 'BsmtExGroup'] = 0
df_train.loc[df_train.BsmtExposure == 'No', 'BsmtExGroup'] = 6
df_train.loc[df_train.BsmtExposure == 'Mn', 'BsmtExGroup'] = 7
df_train.loc[df_train.BsmtExposure == 'Av', 'BsmtExGroup'] = 8
df_train.loc[df_train.BsmtExposure == 'Gd', 'BsmtExGroup'] = 12


x = df_train['BsmtExGroup']
y = df_train['LogPrice']
print(df_train[['BsmtExGroup','SalePrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)
var = "BsmtFinType1"
segm_target(var)
df_train.loc[df_train.BsmtFinType1 == 'NoBsmt', 'BsmtF1Group'] = 0
df_train.loc[df_train.BsmtFinType1 == 'Unf', 'BsmtF1Group'] = 6
df_train.loc[df_train.BsmtFinType1 == 'LwQ', 'BsmtF1Group'] = 4
df_train.loc[df_train.BsmtFinType1 == 'Rec', 'BsmtF1Group'] = 4
df_train.loc[df_train.BsmtFinType1 == 'BLQ', 'BsmtF1Group'] = 4
df_train.loc[df_train.BsmtFinType1 == 'ALQ', 'BsmtF1Group'] = 5
df_train.loc[df_train.BsmtFinType1 == 'GLQ', 'BsmtF1Group'] = 11


x = df_train['BsmtF1Group']
y = df_train['LogPrice']
print(df_train[['BsmtF1Group','SalePrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,8))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)
var = "BsmtFinType2"
segm_target(var)
var = "MisBsm"
segm_target(var)
df_train[['TotalBsmtSF','BsmtFinTotSF', 'BsmtBath', 'BsmtQuGroup']].corr()
g = sns.FacetGrid(df_train, hue='BsmtBath', size = 5)
g.map(plt.hist, 'TotalBsmtSF', alpha= 0.5, bins=50)
g.add_legend()
g = sns.FacetGrid(df_train, hue='BsmtBath', size = 7)
g.map(plt.scatter, 'TotalBsmtSF', 'LogPrice', edgecolor="w")
g.add_legend()
pd.crosstab(df_train['BsmtQuGroup'], df_train['BsmtBath'])
var = "Utilities"
segm_target(var)
var = "Heating"
segm_target(var)
var = "HeatingQC"
segm_target(var)
df_train.loc[df_train.HeatingQC == 'Po', 'HeatQGroup'] = 1
df_train.loc[df_train.HeatingQC == 'Fa', 'HeatQGroup'] = 1
df_train.loc[df_train.HeatingQC == 'TA', 'HeatQGroup'] = 3
df_train.loc[df_train.HeatingQC == 'Gd', 'HeatQGroup'] = 4
df_train.loc[df_train.HeatingQC == 'Ex', 'HeatQGroup'] = 7

x = df_train['HeatQGroup']
y = df_train['LogPrice']

print(df_train[['HeatQGroup', 'LogPrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()
var = "CentralAir"
segm_target(var)
var = "Electrical"
segm_target(var)
df_train.loc[df_train.Electrical == "SBrkr", "ElecGroup"] = "SBrkr"
fuse = ['FuseA', 'FuseF', 'FuseP', 'Mix']
df_train.loc[df_train.Electrical.isin(fuse), "ElecGroup"] = "Fuse"

var = "ElecGroup"
segm_target(var)
g = sns.FacetGrid(df_train, hue='CentralAir', size = 5)
g.map(plt.hist, 'HeatQGroup', alpha= 0.5, bins=10)
g.add_legend()
corr_target('1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'LogPrice')
# checking the sum of 1st and 2nd floor, not always it consistent with GrLivArea, thus I assume it is a 3rd floor
df_train['HasThirdFl'] = 0
df_train.loc[df_train.GrLivArea - df_train['1stFlrSF'] - df_train['2ndFlrSF'] > 0, 'HasThirdFl'] = 1

var = 'HasThirdFl'
segm_target(var)
x = df_train['FullBath']
x1 = df_train['HalfBath']
y = df_train['LogPrice']

print(df_train[['FullBath', 'HalfBath', 'LogPrice']].corr())

fig, ax =plt.subplots(2,2, figsize=(12,10))
sns.regplot(x=x, y=y, ax=ax[0][0])
sns.regplot(x=x, y=y, ax=ax[0][1], x_estimator = np.mean)
sns.regplot(x=x1, y=y, ax=ax[1][0])
sns.regplot(x=x1, y=y, ax=ax[1][1], x_estimator = np.mean)

fig.show()
# I bet the total number of bathrooms is more important than the segmentation per type
df_train['TotBath'] = df_train.FullBath + df_train.HalfBath

x = df_train['TotBath']
y = df_train['LogPrice']

print(df_train[['FullBath', 'HalfBath', 'TotBath', 'LogPrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)
# No house is actually without a bathroom
df_train[df_train.TotBath == 0]['BsmtBath']
x = df_train['BedroomAbvGr']
x1 = df_train['KitchenAbvGr']
x2 = df_train['TotRmsAbvGrd']
y = df_train['LogPrice']

print(df_train[['BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'LogPrice']].corr())

fig, ax =plt.subplots(3,2, figsize=(12,15))
sns.regplot(x=x, y=y, ax=ax[0][0])
sns.regplot(x=x, y=y, ax=ax[0][1], x_estimator = np.mean)
sns.regplot(x=x1, y=y, ax=ax[1][0])
sns.regplot(x=x1, y=y, ax=ax[1][1], x_estimator = np.mean)
sns.regplot(x=x2, y=y, ax=ax[2][0])
sns.regplot(x=x2, y=y, ax=ax[2][1], x_estimator = np.mean)

fig.show()
df_train['TotRooms+Bath'] = df_train.TotRmsAbvGrd + df_train.TotBath

x = df_train['TotRooms+Bath']
y = df_train['LogPrice']

print(df_train[['TotRooms+Bath', 'LogPrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)
var = "KitchenQual"
segm_target(var)
df_train.loc[df_train.KitchenQual == 'Fa', 'KitchQuGroup'] = 1
df_train.loc[df_train.KitchenQual == 'TA', 'KitchQuGroup'] = 4
df_train.loc[df_train.KitchenQual == 'Gd', 'KitchQuGroup'] = 10
df_train.loc[df_train.KitchenQual == 'Ex', 'KitchQuGroup'] = 21

x = df_train['KitchQuGroup']
y = df_train['LogPrice']

print(df_train[['KitchQuGroup','SalePrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)
var = "Functional"
segm_target(var)
x = df_train['TotRmsAbvGrd']
y = df_train['GrLivArea']

print(df_train[['TotRmsAbvGrd', 'GrLivArea']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)
fig.show()
g = sns.FacetGrid(df_train, hue='TotRmsAbvGrd', size = 8)
g.map(plt.scatter, 'GrLivArea', 'LogPrice', edgecolor="w")
g.add_legend()
x = df_train['TotBath']
y = df_train['GrLivArea']

print(df_train[['TotBath', 'GrLivArea']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)
fig.show()
g = sns.FacetGrid(df_train, hue='TotBath', size = 8)
g.map(plt.scatter, 'GrLivArea', 'LogPrice', edgecolor="w")
g.add_legend()
x = df_train['KitchQuGroup']
y = df_train['GrLivArea']

print(df_train[['KitchQuGroup', 'GrLivArea']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)
fig.show()
g = sns.FacetGrid(df_train, hue='KitchQuGroup', size = 8)
g.map(plt.scatter, 'GrLivArea', 'LogPrice', edgecolor="w")
g.add_legend()
corr_target('GarageCars', 'GarageArea', 'SalePrice')
x = df_train['GarageArea']
y = df_train['LogPrice']

sns.regplot(x=x, y=y)
x = df_train[df_train.GarageYrBlt > 1000]['GarageYrBlt'] #filter to avoid the one I filled in with neg values
y = df_train[df_train.GarageYrBlt > 1000]['LogPrice']

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()
x = df_train['Fireplaces']
y = df_train['LogPrice']

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()
df_train.loc[df_train.Fireplaces > 0, 'FireGroup'] = 1
df_train.loc[df_train.Fireplaces == 0, 'FireGroup'] = 0

var = "FireGroup"
segm_target(var)
var = "FireplaceQu"
segm_target(var)
df_train.loc[df_train.FireplaceQu == 'NoFire', 'FrpQuGroup'] = 0
df_train.loc[df_train.FireplaceQu == 'Po', 'FrpQuGroup'] = 0
df_train.loc[df_train.FireplaceQu == 'Fa', 'FrpQuGroup'] = 2
df_train.loc[df_train.FireplaceQu == 'TA', 'FrpQuGroup'] = 3
df_train.loc[df_train.FireplaceQu == 'Gd', 'FrpQuGroup'] = 4
df_train.loc[df_train.FireplaceQu == 'Ex', 'FrpQuGroup'] = 8

x = df_train['FrpQuGroup']
y = df_train['LogPrice']

print(df_train[['FrpQuGroup', 'LogPrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()
var = "GarageType"
segm_target(var)
incl = ['Attchd', 'Basment', 'BuiltIn']
escl = ['2Types', 'CarPort', 'Detchd']
df_train.loc[df_train.GarageType.isin(incl), 'GrgTypeGroup'] = 'Connected'
df_train.loc[df_train.GarageType.isin(escl), 'GrgTypeGroup'] = 'NonConnected'
df_train.loc[df_train.GarageType == 'NoGrg', 'GrgTypeGroup'] = 'NoGrg'

var = 'GrgTypeGroup'
segm_target(var)
var = "GarageFinish"
segm_target(var)
var = "GarageQual"
segm_target(var)
var = "GarageCond"
segm_target(var)
var = "PavedDrive"
segm_target(var)
df_train.loc[df_train.PavedDrive == 'N', 'PvdGroup'] = 0
df_train.loc[df_train.PavedDrive == 'P', 'PvdGroup'] = 1
df_train.loc[df_train.PavedDrive == 'Y', 'PvdGroup'] = 1

print(df_train[['PvdGroup', 'LogPrice']].corr())

var = "PvdGroup"
segm_target(var)
var = "MisGarage"
segm_target(var)
x = df_train['GarageCars']
y = df_train['GarageArea']

print(df_train[['GarageCars', 'GarageArea']].corr())

sns.regplot(x=x, y=y)
g = sns.FacetGrid(df_train, hue="GarageCars", size = 8)
g.map(plt.scatter, "GarageArea", "LogPrice", edgecolor="w")
g.add_legend()
fig, ax = plt.subplots(1,2, figsize=(12, 5))

sns.boxplot('GarageType',"GarageArea",data=df_train, ax=ax[0])
sns.boxplot('GrgTypeGroup',"GarageArea",data=df_train, ax=ax[1])

fig.show()
pd.crosstab(df_train["GrgTypeGroup"],df_train['GarageFinish'])
g = sns.FacetGrid(df_train, hue="GarageFinish", size=7)
g.map(plt.hist, 'GarageArea', alpha= 0.3, bins=50)
g.add_legend()
g = sns.FacetGrid(df_train, hue="GarageFinish", size = 8)
g.map(plt.scatter, "GarageArea", "LogPrice", edgecolor="w")
g.add_legend()
pd.crosstab(df_train["PvdGroup"],df_train['GrgTypeGroup'])
pd.crosstab(df_train["PvdGroup"],df_train['GarageFinish'])
corr_target('WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'LogPrice')
df_train['TotPorch'] = (df_train['WoodDeckSF'] + df_train['OpenPorchSF'] + df_train['EnclosedPorch'] + 
                       df_train['3SsnPorch'] + df_train['ScreenPorch'])

print(df_train[['TotPorch', 'LogPrice']].corr())

x = df_train['TotPorch']
y = df_train['LogPrice']

sns.regplot(x=x, y=y)
fil = ((df_train['TotPorch'] != df_train['WoodDeckSF']) &
      (df_train['TotPorch'] != df_train['OpenPorchSF']) &
      (df_train['TotPorch'] != df_train['EnclosedPorch']) &
      (df_train['TotPorch'] != df_train['3SsnPorch']) &
      (df_train['TotPorch'] != df_train['ScreenPorch']))

df_train[fil][['TotPorch', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']].count()
var = "Fence"
segm_target(var)
df_train.loc[df_train.Fence == 'NoFence', 'FenceGroup'] = 'NoFence'
df_train.loc[(df_train.Fence == 'MnPrv') | (df_train.Fence == 'GdPrv'), 'FenceGroup'] = 'Prv'
df_train.loc[(df_train.Fence == 'MnWw') | (df_train.Fence == 'GdWo'), 'FenceGroup'] = 'Wo'

var = "FenceGroup"
segm_target(var)
var = "PoolQC"
segm_target(var)
g = sns.FacetGrid(df_train, hue="FenceGroup", size = 5)
g.map(plt.scatter, "TotPorch", "LogPrice", edgecolor="w")
g.add_legend()
x = df_train['YrSold']
y = df_train['LogPrice']

print(df_train[['YrSold', 'LogPrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()
x = df_train['MoSold']
y = df_train['LogPrice']

print(df_train[['YrSold', 'MoSold', 'LogPrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()
# Checking if there is some seasonality in the air
df_train[['YrSold', 'MoSold', 'LogPrice']].groupby(['YrSold', 'MoSold']).mean().plot(figsize=(12,5))
df_train[['YrSold', 'MoSold', 'LogPrice']].groupby(['YrSold', 'MoSold']).median().plot(figsize=(12,5))
df_train['OldwhenSold'] = df_train.YrSold - df_train.YearBuilt

x = df_train['OldwhenSold']
y = df_train['LogPrice']

print(df_train[['OldwhenSold', 'LogPrice']].corr())

fig, ax =plt.subplots(1,2, figsize=(12,5))
sns.regplot(x=x, y=y, ax=ax[0])
sns.regplot(x=x, y=y, ax=ax[1], x_estimator = np.mean)

fig.show()
x = df_train['MiscVal']
y = df_train['LogPrice']

sns.regplot(x=x, y=y)
var = "MiscFeature"
segm_target(var)
var = "SaleType"
segm_target(var)
other = ['Oth', 'New']
cont = ['ConLw', 'ConLi', 'ConLD', 'Con', 'CWD', 'COD']
df_train.loc[df_train.SaleType.isin(other), 'SaleTyGroup'] = 'Other'
df_train.loc[df_train.SaleType.isin(cont), 'SaleTyGroup'] = 'Contract'
df_train.loc[df_train.SaleType == 'WD', 'SaleTyGroup'] = 'WD'

var = "SaleTyGroup"
segm_target(var)
var = "SaleCondition"
segm_target(var)
feat = ['MSZoningGroup', 'AlleyGroup', 'LotShapeGroup', 'Condition1Group',
       'YearBuilt', 'BldTypeGroup', 'HStyleGroup', 'OverallQual', 
       'MasVnrArea', 'MasTypeGroup', 'FoundGroup', 'ExtQuGroup', 
       'TotalBsmtSF',' BsmtUnfSF', 'BsmtBath', 'BsmtQuGroup', 'MisBsm',
       'HeatQGroup', 'CentralAir', 'ElecGroup',
       'GrLivArea', 'KitchQuGroup',
       'GarageArea', 'FireGroup', 'FrpQuGroup', 'GrgTypeGroup', 'GarageFinish', 'PvdGroup', 'MisGarage',
       'TotPorch', 'FenceGroup', 'SaleTyGroup']