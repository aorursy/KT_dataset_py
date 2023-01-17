# standard
import pandas as pd
import numpy as np

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
# ---------- DF IMPORT -------------
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
combine = [df_train, df_test]
df_train.name = 'Train'
df_test.name = 'Test'
df_train.sample(10)
df_train.columns
for df in combine:
    if df.name == 'Train':
        mis_train = []
        cols = df.columns
        for col in cols:
            mis = df[col].isnull().sum()
            if mis > 0:
                print("{}: {} missing, {}%".format(col, mis, round(mis/df.shape[0] * 100, 3)))
                mis_train.append(col)
        print("_"*40)
        print("_"*40)
    if df.name == 'Test':
        mis_test = []
        cols = df.columns
        for col in cols:
            mis = df[col].isnull().sum()
            if mis > 0:
                print("{}: {} missing, {}%".format(col, mis, round(mis/df.shape[0] * 100, 3)))
                mis_test.append(col)

print("\n")
print(mis_train)
print("_"*40)
print(mis_test)
#helper functions to avoid repeating myself too much

def traintest_hist(feat, nbins):
    fig, axes = plt.subplots(1, 2)

    df_train[feat].hist(bins = nbins, ax=axes[0])
    df_test[feat].hist(bins = nbins, ax=axes[1])

    print("{}: {} missing, {}%".format('Train', df_train[feat].isnull().sum(), 
                                   round(df_train[feat].isnull().sum()/df_train.shape[0] * 100, 3)))
    print("{}: {} missing, {}%".format('Test', df_test[feat].isnull().sum(), 
                                   round(df_test[feat].isnull().sum()/df_test.shape[0] * 100, 3)))
    
def categ_dist_mis(cats):
    for cat in cats:
        print(cat)
        print("_"*40)
        print("In train: ")
        print(df_train[cat].value_counts(dropna = False))
        print('_'*40)
        print("In test: ")
        print(df_test[cat].value_counts(dropna = False))
        print("_"*40)
        print("_"*40)
        
def checkclean(feats):
    for feat in feats:
        print(feat)
        for df in combine:
            print("In {}".format(df.name))
            print(df[feat].value_counts(dropna = False))
            print("_"*40)
    print("_"*40)
traintest_hist('LotFrontage', 50)
traintest_hist('LotArea', 50)
cats = ['MSZoning', 'Street', 'Alley', 'LotShape', 
       'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2']

categ_dist_mis(cats)
for df in combine:
    df.loc[(df.MSZoning == 'FV') | (df.MSZoning == 'RH') | (df.MSZoning == 'C (all)'), 'MSZoning'] = 3 #other
    df.loc[df.MSZoning == 'RL', 'MSZoning'] = 1
    df.loc[df.MSZoning == 'RM', 'MSZoning'] = 2
    #Condition1
    df.loc[(df.Condition1 != 'Norm'), 'Condition1'] = 0 # NotNormal
    df.loc[df.Condition1 == 'Norm', 'Condition1'] = 1
    #LotFrontage
    df.loc[df.LotFrontage.isnull(), 'LotFrontage'] = 0
    #Alley
    df.loc[df.Alley.notnull(), 'Alley'] = 1 # Alley
    df.loc[df.Alley.isnull(), 'Alley'] = 0 #NotAlley
    #LotShape
    df.loc[(df.LotShape != 'Reg'), 'LotShape'] = 0 # NotRegular
    df.loc[df.LotShape == 'Reg', 'LotShape'] = 1
    #LotConfig
    df.loc[df.LotConfig == 'Inside', 'LotConfig'] = 1
    df.loc[df.LotConfig == 'Corner', 'LotConfig'] = 2
    df.loc[df.LotConfig == 'CulDSac', 'LotConfig'] = 3
    df.loc[(df.LotConfig == 'FR2') | (df.LotConfig == 'FR3'), 'LotConfig'] = 4
    #LandSlope
    df.loc[(df.LandSlope != 'Gtl'), 'LandSlope'] = 0 # NotGentle
    df.loc[df.LandSlope == 'Gtl', 'LandSlope'] = 1
    #LandContour
    df.loc[(df.LandContour != 'Lvl'), 'LandContour'] = 0 # NotLeveled
    df.loc[df.LandContour == 'Lvl', 'LandContour'] = 1      
    
print("LotFrontage")    
print("In train: ")
print(df_train.LotFrontage.isnull().sum())
print('_'*40)
print("In test: ")
print(df_test.LotFrontage.isnull().sum())
print('_'*40)
print('_'*40)

feats = ["MSZoning", "Condition1", "Alley", "LotShape", "LotShape", "LotConfig", "LandSlope", "LandContour"]
    
checkclean(feats)
traintest_hist('OverallQual', 10)
traintest_hist('OverallCond', 10)
traintest_hist('YearBuilt', 20)
traintest_hist('YearRemodAdd', 20)
cats = ['MSSubClass', 'BldgType', 'HouseStyle']

categ_dist_mis(cats)
for df in combine:
    df.loc[df.BldgType == '1Fam', 'BldgType'] = 1
    df.loc[(df.BldgType == '2fmCon') | (df.BldgType == 'Duplex'), 'BldgType'] = 2
    df.loc[df.BldgType== 'CulDSac', 'BldgType'] = 3
    df.loc[(df.BldgType == 'TwnhsE') | (df.BldgType == 'Twnhs'), 'BldgType'] = 4
    #HouseStyle
    df.loc[df.HouseStyle == '1Story', 'HouseStyle'] = 1
    df.loc[(df.HouseStyle == 'SFoyer') | (df.HouseStyle == 'SLvl'), 'HouseStyle'] = 2
    sto = ['1.5Fin', '1.5Unf', '2.5Fin', '2.5Unf', '2Story']
    df.loc[df.HouseStyle.isin(sto), 'HouseStyle'] = 3
    
feats = ["BldgType", "HouseStyle"]
    
checkclean(feats)
traintest_hist('MasVnrArea', 20)
cats = ['Foundation','RoofStyle', 'RoofMatl', 'Exterior1st', 
        'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond']

categ_dist_mis(cats)
for df in combine:
    #MasVnrArea
    df.loc[(df.MasVnrArea == 0), 'MasVnr'] = 0 
    df.loc[df.MasVnrArea > 0, 'MasVnr'] = 1
    #MasVnrType
    df.loc[(df.MasVnrType == 'None'), 'MasVnrType'] = 0 
    df.loc[(df.MasVnrType == 'BrkFace') |(df.MasVnrType == 'BrkCmn') , 'MasVnrType'] = 1 #bricks
    df.loc[(df.MasVnrType == 'Stone'), 'MasVnrType'] = 2
    #ExterQual
    df.loc[(df.ExterQual == 'TA') |(df.ExterQual == 'Fa') , 'ExterQual'] = 0 #Typical or Fair
    df.loc[(df.ExterQual == 'Gd') |(df.ExterQual == 'Ex') , 'ExterQual'] = 1 #Good or Excellent
    #ExterCond
    df.loc[(df.ExterCond == 'TA') |(df.ExterCond == 'Fa') | 
           (df.ExterCond == 'Po'), 'ExterCond'] = 0 #Typical or Fair or Poor
    df.loc[(df.ExterCond == 'Gd') |(df.ExterCond == 'Ex') , 'ExterCond'] = 1 #Good or Excellent
    ###
    feat = 'Exterior1st'
    ##
    df.loc[(df[feat] == 'VinylSd'), feat] = 0 
    df.loc[(df[feat] == 'Stucco') |(df[feat] == 'ImStucc') , feat] = 1 #Stucco or similar
    df.loc[(df[feat] == 'Wd Sdng') |(df[feat] == 'WdShing') | (df[feat] == 'Plywood'), feat] = 2 #woods
    df.loc[(df[feat] == 'MetalSd'), feat] = 3
    df.loc[(df[feat] == 'HdBoard'), feat] = 4
    df.loc[(df[feat] == 'BrkFace') |(df[feat] == 'BrkComm') , feat] = 5 #Bricks
    df.loc[(df[feat] == 'CemntBd') |(df[feat] == 'AsbShng') | 
           (df[feat] == 'AsphShn') | (df[feat] == 'CBlock') | (df[feat] == 'Stone'), feat] = 1 #Cement and similar
    # avoided sparse class
    feat = 'Exterior2nd'
    ##
    df.loc[(df[feat] == 'VinylSd'), feat] = 0 
    df.loc[(df[feat] == 'Stucco') |(df[feat] == 'ImStucc') , feat] = 1 #Stucco or similar (actually just Other)
    df.loc[(df[feat] == 'Wd Sdng') |(df[feat] == 'Wd Shng') | (df[feat] == 'Plywood'), feat] = 2 #woods
    df.loc[(df[feat] == 'MetalSd'), feat] = 3
    df.loc[(df[feat] == 'HdBoard'), feat] = 4
    df.loc[(df[feat] == 'BrkFace') |(df[feat] == 'Brk Cmn') , feat] = 1 #Bricks (actually just Other)
    df.loc[(df[feat] == 'CmentBd') |(df[feat] == 'AsbShng') | 
           (df[feat] == 'AsphShn') | (df[feat] == 'CBlock') | 
           (df[feat] == 'Stone') |(df[feat] == 'Other'), feat] = 1 #Cement and similar (actually just Other)
    # avoided sparse class
    
feats = ['MasVnr', 'MasVnrType', 'ExterQual', 'ExterCond', 'Exterior1st', 'Exterior2nd']

checkclean(feats)
traintest_hist('BsmtFinSF1', 30)
traintest_hist('BsmtFinSF2', 30)
traintest_hist('BsmtUnfSF', 30)
traintest_hist('TotalBsmtSF', 30)
traintest_hist('BsmtFullBath', 5)
traintest_hist('BsmtHalfBath', 5)
cats = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

categ_dist_mis(cats)
for df in combine:
    df['BsmFinSFTot'] = df['BsmtFinSF1'] + df['BsmtFinSF2']
    #BsmtFullBath and BsmtHalfBath
    df['BsmBath'] = df['BsmtFullBath'] + df['BsmtHalfBath']
    df.loc[(df.BsmBath == 0), 'BsmBath'] = 0 #noBath
    df.loc[(df.BsmBath > 0), 'BsmBath'] = 1
    #MissingBasement
    fil = ((df.BsmtQual.isnull()) & (df.BsmtCond.isnull()) & (df.BsmtExposure.isnull()) &
          (df.BsmtFinType1.isnull()) & (df.BsmtFinType2.isnull()))
    df['MisBsm'] = 0
    df.loc[fil, 'MisBsm'] = 1
    #BsmtQual
    df.loc[fil, 'BsmtQual'] = -99 #missing basement
    df.loc[(df.BsmtQual == 'Fa') | (df.BsmtQual == 'TA'), 'BsmtQual'] = 1 #fair or typical
    df.loc[(df.BsmtQual == 'Gd'), 'BsmtQual'] = 2
    df.loc[(df.BsmtQual == 'Ex'), 'BsmtQual'] = 3
    #BsmtCond
    df.loc[fil, 'BsmtCond'] = -99 #missing basement
    df.loc[(df.BsmtCond == 'Fa') | (df.BsmtCond == 'TA') |
           (df.BsmtCond == 'Po'), 'BsmtCond'] = 1 #fair or typical or poor
    df.loc[(df.BsmtCond == 'Gd'), 'BsmtCond'] = 2
    df.loc[(df.BsmtCond == 'Ex'), 'BsmtCond'] = 3
    #BsmtExposure
    df.loc[fil, 'BsmtExposure'] = -99 #missing basement
    df.loc[df.BsmtExposure == 'Gd', 'BsmtExposure'] = 1
    df.loc[df.BsmtExposure == 'Av', 'BsmtExposure'] = 2
    df.loc[df.BsmtExposure == 'Mn', 'BsmtExposure'] = 3
    df.loc[df.BsmtExposure == 'No', 'BsmtExposure'] = 4
    #BsmtFinType1
    df.loc[fil, 'BsmtFinType1'] = -99 #missing basement
    df.loc[(df.BsmtFinType1 == 'Unf') , 'BsmtFinType1'] = 1
    df.loc[(df.BsmtFinType1 == 'GLQ') , 'BsmtFinType1'] = 2
    df.loc[(df.BsmtFinType1 == 'ALQ') , 'BsmtFinType1'] = 3
    df.loc[(df.BsmtFinType1 == 'BLQ') , 'BsmtFinType1'] = 4
    df.loc[(df.BsmtFinType1 == 'Rec') , 'BsmtFinType1'] = 5
    df.loc[(df.BsmtFinType1 == 'LwQ') , 'BsmtFinType1'] = 6
    #BsmtFinType2
    df.loc[fil, 'BsmtFinType2'] = -99 #missing basement
    df.loc[(df.BsmtFinType2 != 'Unf') & (df.BsmtFinType2 != -99), 'BsmtFinType2'] = 0 #notUnf
    df.loc[(df.BsmtFinType2 == 'Unf') , 'BsmtFinType2'] = 1
    
feats = ['BsmBath', 'MisBsm', 'BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

checkclean(feats)

traintest_hist('BsmFinSFTot', 30)
# Checking consistency with numerical variable
df_train[df_train.BsmFinSFTot== 0]['MisBsm'].value_counts(dropna=False)
# Checking consistency between numerical variables
fil = (df_train.BsmFinSFTot + df_train.BsmtUnfSF != df_train.TotalBsmtSF)
df[fil].shape
cats = ['Utilities', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical']

categ_dist_mis(cats)
for df in combine:
    df.loc[df.HeatingQC == 'Ex', 'HeatingQC'] = 1
    df.loc[df.HeatingQC == 'Gd', 'HeatingQC'] = 2
    df.loc[df.HeatingQC == 'TA', 'HeatingQC'] = 3
    df.loc[(df.HeatingQC == 'Fa') | (df.HeatingQC == 'Po'), 'HeatingQC'] = 4 #Fair or poor
    #CentralAir
    df.loc[df.CentralAir == 'N', 'CentralAir'] = 0
    df.loc[df.CentralAir == 'Y', 'CentralAir'] = 1
    #Electrical
    fil = ((df.Electrical == 'FuseA') | (df.Electrical == 'FuseF') |
          (df.Electrical == 'FuseP') | (df.Electrical == 'Mix'))
    df.loc[fil, 'Electrical'] = 0
    df.loc[df.Electrical == 'SBrkr', 'Electrical'] = 1
    
feats = ['HeatingQC', 'CentralAir', 'Electrical']

checkclean(feats)
traintest_hist('1stFlrSF', 30)
traintest_hist('2ndFlrSF', 30)
traintest_hist('LowQualFinSF', 30)
traintest_hist('GrLivArea', 30)
traintest_hist('FullBath', 10)
traintest_hist('HalfBath', 10)
traintest_hist('BedroomAbvGr', 10)
traintest_hist('KitchenAbvGr', 10)
traintest_hist('TotRmsAbvGrd', 20)
cats = ['KitchenQual', 'Functional']

categ_dist_mis(cats)
for df in combine:
    df['2ndFlr'] = 0 #no2floor
    df.loc[df['2ndFlrSF'] > 0, '2ndFlr'] = 1 #with2floor
    #Baths
    df['BathsTot'] = df.FullBath + df.HalfBath 
    #KitchenQual
    df.loc[df.KitchenQual == 'Ex', 'KitchenQual'] = 1
    df.loc[df.KitchenQual == 'Gd', 'KitchenQual'] = 2
    df.loc[(df.KitchenQual == 'TA') | (df.KitchenQual == 'Fa'), 'KitchenQual'] = 3
    #Functional
    fun = ['Min2', 'Min1', 'Mod', 'Maj1', 'Sev', 'Maj2']
    fil = df.Functional.isin(fun)
    df.loc[fil, 'Functional'] = 0 #nontypical
    df.loc[df.Functional == 'Typ', 'Functional'] = 1
    
feats = ['2ndFlr', 'BathsTot', 'KitchenQual',  'Functional']

checkclean(feats)
# Checking consistency between numerical variables
fil = (df_train['1stFlrSF'] + df_train['2ndFlrSF'] != df_train.GrLivArea)
df[fil].shape
# Checking consistency between categorical variables
fil = (df_train['BedroomAbvGr'] + df_train['KitchenAbvGr'] != df_train.TotRmsAbvGrd)
df[fil].shape
traintest_hist('Fireplaces', 10)
traintest_hist('GarageYrBlt', 30)
traintest_hist('GarageCars', 10)
traintest_hist('GarageArea', 30)
cats = ['FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']

categ_dist_mis(cats)
for df in combine:
    df.loc[df['Fireplaces'] > 0, 'Fireplaces'] = 1 #Fireplaces
    df.loc[df['Fireplaces'] == 0, 'Fireplaces'] = 0
    #FireplaceQu
    df.loc[(df.Fireplaces == 0) & (df.FireplaceQu.isnull()), 'FireplaceQu'] = -99 #missing
    df.loc[(df.FireplaceQu == 'Ex') | (df.FireplaceQu== 'Gd'), 'FireplaceQu'] = 1
    df.loc[(df.FireplaceQu == 'TA'), 'FireplaceQu'] = 2
    df.loc[(df.FireplaceQu == 'Fa') | (df.FireplaceQu== 'Po'), 'FireplaceQu'] = 3
    #MisGarage
    fil = ((df.GarageYrBlt.isnull()) & (df.GarageType.isnull()) & (df.GarageFinish.isnull()) &
          (df.GarageQual.isnull()) & (df.GarageCond.isnull()))
    df['MisGarage'] = 0
    df.loc[fil, 'MisGarage'] = 1
    #GarageYrBlt
    df.loc[df.GarageYrBlt > 2200, 'GarageYrBlt'] = 2007 #correct mistake
    df.loc[fil, 'GarageYrBlt'] = -99
    #GarageType
    gty = ['Attchd', 'BuiltIn', 'Basment', '2Types']
    fil1 = df.GarageType.isin(gty)
    fil2 = ((df.GarageType == 'Detchd') | (df.GarageType == 'CarPort'))
    df.loc[fil, 'GarageType'] = -99 #missing garage
    df.loc[fil1, 'GarageType'] = 0 #attached and similar
    df.loc[fil2, 'GarageType'] = 1
    #GarageFinish
    df.loc[fil, 'GarageFinish'] = -99 #missing
    df.loc[df.GarageFinish == 'Unf', 'GarageFinish'] = 1
    df.loc[df.GarageFinish == 'RFn', 'GarageFinish'] = 2
    df.loc[df.GarageFinish == 'Fin', 'GarageFinish'] = 3
    #PavedDrive
    df.loc[df.PavedDrive == 'Y', 'PavedDrive'] = 1
    df.loc[df.PavedDrive == 'P', 'PavedDrive'] = 1
    df.loc[df.PavedDrive == 'N', 'PavedDrive'] = 0
    
    
feats = ['Fireplaces', 'FireplaceQu', 'MisGarage', 'GarageType', 'GarageFinish', 'PavedDrive']

checkclean(feats)

traintest_hist('GarageYrBlt', 30)
traintest_hist('WoodDeckSF', 30)
traintest_hist('OpenPorchSF', 30)
traintest_hist('EnclosedPorch', 30)
traintest_hist('3SsnPorch', 30)
traintest_hist('ScreenPorch', 30)
traintest_hist('PoolArea', 30)
cats = ['PoolQC', 'Fence']

categ_dist_mis(cats)
for df in combine:
    df['Porch'] = df.ScreenPorch + df['3SsnPorch'] + df.EnclosedPorch + df.OpenPorchSF + df.WoodDeckSF
    #Fence
    df.loc[df.Fence.isnull(), 'Fence'] = 0 #missing fence
    df.loc[(df.Fence == 'MnPrv') | (df.Fence == 'GdPrv'), 'Fence'] = 1 #privacy
    df.loc[(df.Fence == 'GdWo') | (df.Fence == 'MnWw'), 'Fence'] = 2 #woods
    
feats = ['Fence']

checkclean(feats) 

traintest_hist('Porch', 30)
traintest_hist('MiscVal', 30)
traintest_hist('YrSold', 10)
traintest_hist('MoSold', 12)
cats = ['MiscFeature', 'SaleType', 'SaleCondition']

categ_dist_mis(cats)
for df in combine:
    war = ['WD', 'CWD', 'VWD']
    con = ['Con', 'ConLw', 'ConLI', 'ConLD', 'COD']
    oth = ['New', 'Oth']
    df.loc[df.SaleType.isin(war), 'SaleType'] = 1
    df.loc[df.SaleType.isin(con), 'SaleType'] = 2
    df.loc[df.SaleType.isin(oth), 'SaleType'] = 3
    #SaleCondition
    nnor = ['Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']
    df.loc[df.SaleCondition.isin(nnor), 'SaleCondition'] = 0 #notNormal
    df.loc[df.SaleCondition == 'Normal', 'SaleCondition'] = 1
    
feats = ['SaleType', 'SaleCondition']

checkclean(feats)
irrs = ['Street', 'Condition2', 'Utilities', 'RoofMatl', 'RoofStyle', 'Heating',
      'LowQualFinSF', 'GarageQual', 'GarageCond', 'PoolArea', 'PoolQC', 'MiscVal', 'MiscFeature', 
       #the next features were just replaced
       'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2','BsmtFullBath','BsmtHalfBath', 'FullBath', 'HalfBath',
        'ScreenPorch', 'EnclosedPorch','3SsnPorch','OpenPorchSF', 'WoodDeckSF']

print('Before: {} features in train and {} in test'.format(len(df_train.columns), len(df_test.columns)))

for df in combine:
    for irr in irrs:
        del df[irr]
        
print('After: {} features in train and {} in test'.format(len(df_train.columns), len(df_test.columns)))
for df in combine:
    if df.name == 'Train':
        mis_train = []
        cols = df.columns
        for col in cols:
            mis = df[col].isnull().sum()
            if mis > 0:
                print("{}: {} missing, {}%".format(col, mis, round(mis/df.shape[0] * 100, 3)))
                mis_train.append(col)
        print("_"*40)
        print("_"*40)
    if df.name == 'Test':
        mis_test = []
        cols = df.columns
        for col in cols:
            mis = df[col].isnull().sum()
            if mis > 0:
                print("{}: {} missing, {}%".format(col, mis, round(mis/df.shape[0] * 100, 3)))
                mis_test.append(col)

print("\n")
print(mis_train)
print("_"*40)
print(mis_test)
def find_segment(df, feat): #returns the columns for which the missing entries have only 1 possible value
    mis = df[feat].isnull().sum()
    cols = df.columns
    seg = []
    for col in cols:
        vc = df[df[feat].isnull()][col].value_counts(dropna=False).iloc[0]
        if (vc == mis):
            seg.append(col)
    return seg

def find_mode(df, feat): #returns the mode to fill in the missing feat
    md = df[df[feat].isnull()][find_segment(df, feat)].dropna(axis = 1).mode()
    md = pd.merge(df, md, how='inner')[feat].mode().iloc[0]
    return md

def find_median(df, feat): #returns the median to fill in the missing feat
    md = df[df[feat].isnull()][find_segment(df, feat)].dropna(axis = 1).mode()
    md = pd.merge(df, md, how='inner')[feat].median()
    return md

def similar_mode(df, col, feats): #returns the mode in a segment made by similarity in feats
    sm = df[df[col].isnull()][feats]
    md = pd.merge(df, sm, how='inner')[col].mode().iloc[0]
    return md
    
def similar_median(df, col, feats): #returns the median in a segment made by similarity in feats
    sm = df[df[col].isnull()][feats]
    md = pd.merge(df, sm, how='inner')[col].median()
    return md
#Train
#MasVnrType
md = find_mode(df_train, 'MasVnrType')
print("MasVnrType {}".format(md))
df_train[['MasVnrType']] = df_train[['MasVnrType']].fillna(md)
#MasVnrArea
md = find_mode(df_train, 'MasVnr')
print("MasVnr {}".format(md))
df_train[['MasVnr']] = df_train[['MasVnr']].fillna(md)
#BsmtExposure
simi = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2']
md = similar_mode(df_train, 'BsmtExposure', simi)
print("BsmtExposure {}".format(md))
df_train[['BsmtExposure']] = df_train[['BsmtExposure']].fillna(md)
#Electrical
simi = ['HeatingQC', 'CentralAir']
md = similar_mode(df_train, 'Electrical', simi)
print("Electrical {}".format(md))
df_train[['Electrical']] = df_train[['Electrical']].fillna(md)

cols = df_train.columns
print("Start printing the missing values...")
for col in cols:
    mis = df_train[col].isnull().sum()
    if mis > 0:
        print("{}: {} missing, {}%".format(col, mis, round(mis/df_train.shape[0] * 100, 3)))
print("...done printing the missing values")
#Test
#MSZoning
md = find_mode(df_test, 'MSZoning')
print("MSZoning {}".format(md))
df_test[['MSZoning']] = df_test[['MSZoning']].fillna(md)
#Exterior1st
simi = ['ExterQual', 'ExterCond']
md = similar_mode(df_test, 'Exterior1st', simi)
print("Exterior1st {}".format(md))
df_test[['Exterior1st']] = df_test[['Exterior1st']].fillna(md)
#Exterior2nd
simi = ['ExterQual', 'ExterCond']
md = similar_mode(df_test, 'Exterior2nd', simi)
print("Exterior2nd {}".format(md))
df_test[['Exterior2nd']] = df_test[['Exterior2nd']].fillna(md)
#MasVnrType
md = find_mode(df_test, 'MasVnrType')
print("MasVnrType {}".format(md))
df_test[['MasVnrType']] = df_test[['MasVnrType']].fillna(md)
#MasVnrArea
md = find_mode(df_test, 'MasVnr')
print("MasVnr {}".format(md))
df_test[['MasVnr']] = df_test[['MasVnr']].fillna(md)
#BsmtQual
simi = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
md = similar_mode(df_test, 'BsmtQual', simi)
print("BsmtQual {}".format(md))
df_test[['BsmtQual']] = df_test[['BsmtQual']].fillna(md)
#BsmtCond
simi = ['BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
md = similar_mode(df_test, 'BsmtCond', simi)
print("BsmtCond {}".format(md))
df_test[['BsmtCond']] = df_test[['BsmtCond']].fillna(md)
#BsmtExposure
simi = ['BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2']
md = similar_mode(df_test, 'BsmtExposure', simi)
print("BsmtExposure {}".format(md))
df_test[['BsmtExposure']] = df_test[['BsmtExposure']].fillna(md)
#BsmtUnfSF and TotalBsmtSF
df_test[['BsmtUnfSF', 'TotalBsmtSF']] = df_test[['BsmtUnfSF', 'TotalBsmtSF']].fillna(-99)
#KitchenQual
md = df_test.KitchenQual.mode().iloc[0]
print("KitchenQual {}".format(md))
df_test[['KitchenQual']] = df_test[['KitchenQual']].fillna(md)
#Functional
df_test[['Functional']] = df_test[['Functional']].fillna(1) #manually inserting the mode
#GarageYrBlt
simi = ['YearBuilt']
md = similar_median(df_test, 'GarageYrBlt', simi)
print("GarageYrBlt {}".format(md))
df_test[['GarageYrBlt']] = df_test[['GarageYrBlt']].fillna(md)
#GarageFinish
simi = ['YearBuilt', 'GarageYrBlt']
md = similar_median(df_test, 'GarageFinish', simi)
print("GarageFinish {}".format(md))
df_test[['GarageFinish']] = df_test[['GarageFinish']].fillna(md)
#GarageCars
simi = ['GarageType', 'MisGarage']
md = similar_mode(df_test, 'GarageCars', simi)
print("GarageCars {}".format(md))
df_test[['GarageCars']] = df_test[['GarageCars']].fillna(md)
#GarageArea
simi = ['GarageType', 'MisGarage', 'GarageCars']
md = similar_median(df_test, 'GarageArea', simi)
print("GarageArea {}".format(md))
df_test[['GarageArea']] = df_test[['GarageArea']].fillna(md)
#SaleType
simi = ['SaleCondition']
md = similar_mode(df_test, 'SaleType', simi)
print("SaleType {}".format(md))
df_test[['SaleType']] = df_test[['SaleType']].fillna(md)
#BsmFinSFTot and BsmBath
df_test[['BsmFinSFTot', 'BsmBath']] = df_test[['BsmFinSFTot', 'BsmBath']].fillna(-99)

cols = df_test.columns
print("Start printing the missing values...")
for col in cols:
    mis = df_test[col].isnull().sum()
    if mis > 0:
        print("{}: {} missing, {}%".format(col, mis, round(mis/df_test.shape[0] * 100, 3)))
print("...done printing the missing values")
#fixing the datatype
strings = ['MSSubClass']
ints = ['Alley', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope',
       'Condition1', 'BldgType', 'HouseStyle', 'Exterior1st', 'Exterior2nd',
       'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2',
       'Electrical', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'FireplaceQu',
       'GarageType', 'GarageFinish', 'PavedDrive', 'Fence', 'SaleType', 'SaleCondition']

for df in combine:
    df[strings] = df[strings].astype(str)
    df[ints] = df[ints].astype(int)
df_train.to_csv("trainclean.csv", index=False)
df_test.to_csv("testclean.csv", index=False)
