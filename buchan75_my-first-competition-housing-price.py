# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.tail()
df_train.info()
df_train.shape
## Before starting the EDA process to understand what target values really is
#histogram and normal probability plot
import scipy.stats as stats
from scipy.stats import norm
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
df_train.describe()
# Delete columns that have more than 600 null or nan values and non-fixed variables (time depended values)

df_train_reduced = df_train.drop(['Id','Alley','Utilities','FireplaceQu','PoolQC', 'Fence','MiscFeature','YearBuilt','YearRemodAdd','GarageYrBlt','MoSold','YrSold'], axis=1)
df_train_reduced.head()
# Delete columns that have more than 600 null or nan values

df_train_reduced2 = df_train_reduced.dropna(axis=0, how='any') #axis = 0 means index
df_train_reduced2.isnull().sum()
df_train_reduced2.info()
df_train_reduced2.columns
#scatter plot grlivarea/saleprice

colum = [ 'MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','MasVnrArea','BsmtFinSF1',
         'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
         'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces',
         'GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']

for var in  colum:
    data = pd.concat([df_train_reduced2['SalePrice'], df_train_reduced2[var]], axis=1)
    sns.lmplot(var,'SalePrice', data)
    plt.title(var)
plt.show()
# Try to convert categorised values to numerical variable

# Define function for converting categorised into numeric
def convert_catego_numeric(x, list_x):
    if x in list_x:
        x = list_x.index(x)
        return  int(x  + 1)  
    
list_MSZoning = ['C','FV','I','RH','RL','RP','RM']
list_LotShape = ['Reg','IR1','IR2','IR3']
list_LandContour = ['Lvl','Bnk','HLS','Low']
list_LotConfig = ['Inside','Corner','CulDSac','FR2','FR3']
list_LandSlope = ['Gtl','Mod','Sev']
list_Neighborhood = ['Blmngtn','Blueste','BrDale','BrkSide','ClearCr','CollgCr','Crawfor','Edwards','Gilbert','IDOTRR','MeadowV','Mitchel','Names','NoRidge','NPkVill','NridgHt','NWAmes','OldTown','SWISU','Sawyer','SawyerW','Somerst','StoneBr','Timber','Veenker']
list_Condition1 = ['Artery','Feedr','Norm','RRNn','RRAn','PosN','PosA','RRNe','RRAe']
list_Condition2 = ['Artery','Feedr','Norm','RRNn','PosN','PosA','RRNe','RRAe']
list_BldgType = ['1Fam','2FmCon','Duplx','TwnhsE','TwnhsI']
list_HouseStyle =['1Story','1.5Fin','1.5Unf','2Story','2.5Fin','2.5Unf','SFoyer','SLvl']
list_RoofStyle = ['Flat','Gable','Gambrel','Hip','Mansard','Shed']
list_RoofMatl = ['ClyTile','CompShg','Membran','Metal','Roll','Tar&Grv','WdShake','WdShngl']
list_Exterior1st = ['AsbShng','AsphShn','BrkComm','BrkFace','CBlock','CemntBd','HdBoard','ImStucc','MetalSd','Other','Plywood','PreCast','Stone','Stucco','VinylSd','Wd Sdng','WdShing']
list_Exterior2nd = ['AsbShng','AsphShn','BrkComm','BrkFace','CBlock','CemntBd','HdBoard','ImStucc','MetalSd','Other','Plywood','PreCast','Stone','Stucco','VinylSd','Wd Sdng','WdShing']
list_MasVnrType =['BrkCmn','BrkFace','CBlock','None','Stone']
list_ExterQual = ['Ex','Gd','TA','Fa','Po']
list_ExterCond = ['Ex','Gd','TA','Fa','Po']
list_Foundation = ['BrkTil','CBlock','PConc','Slab','Stone','Wood']
list_BsmtQual = ['Ex','Gd','TA','Fa','Po','NA']
list_BsmtCond =['Ex','Gd','TA','Fa','Po','NA']
list_BsmtExposure =['Gd','Av','Mn','No','NA']
list_BsmtFinType1 =['GLQ','ALQ','BLQ','Rec','LwQ','Unf','NA']
list_BsmtFinType2 = ['GLQ','ALQ','BLQ','Rec','LwQ','Unf','NA']
list_Heating = ['Floor','GasA','GasW','Grav','OthW','Wall']
list_HeatingQC =['Ex','Gd','TA','Fa','Po']
list_CentralAir =['N','Y']
list_Electrical = ['SBrkr','FuseA','FuseF','FuseP','Mix']
list_KitchenQual = ['Ex','Gd','TA','Fa','Po']
list_Functional = ['Typ','Min1','Min2','Mod','Maj1','Maj2','Sev','Sal']
list_GarageType = ['2Types','Attchd','Basment','BuiltIn','CarPort','Detchd','NA']
list_GarageFinish = ['Fin','RFn','Unf','NA']
list_GarageQual = ['Ex','Gd','TA','Fa','Po','NA']
list_GarageCond = ['Ex','Gd','TA','Fa','Po','NA']
list_PavedDrive = ['Y','P','N']
list_SaleType = ['WD','CWD','VWD','New','COD','Con','ConLw','ConLI','ConLD','Oth']
list_SaleCondition = ['Normal','Abnorml','AdjLand','Alloca','Family','Partial']

df_train_reduced2['MSZoning'] = df_train_reduced2['MSZoning'].apply(lambda x: convert_catego_numeric(x, list_MSZoning))
df_train_reduced2['LotShape'] = df_train_reduced2['LotShape'].apply(lambda x: convert_catego_numeric(x, list_LotShape))
df_train_reduced2['LandContour'] = df_train_reduced2['LandContour'].apply(lambda x: convert_catego_numeric(x, list_LandContour))
df_train_reduced2['LotConfig'] = df_train_reduced2['LotConfig'].apply(lambda x: convert_catego_numeric(x, list_LotConfig))
df_train_reduced2['LandSlope'] = df_train_reduced2['LandSlope'].apply(lambda x: convert_catego_numeric(x, list_LandSlope))
df_train_reduced2['Neighborhood'] = df_train_reduced2['Neighborhood'].apply(lambda x: convert_catego_numeric(x, list_Neighborhood))
df_train_reduced2['Condition1'] = df_train_reduced2['Condition1'].apply(lambda x: convert_catego_numeric(x, list_Condition1))
df_train_reduced2['Condition2'] = df_train_reduced2['Condition2'].apply(lambda x: convert_catego_numeric(x, list_Condition2))
df_train_reduced2['BldgType'] = df_train_reduced2['BldgType'].apply(lambda x: convert_catego_numeric(x, list_BldgType))
df_train_reduced2['HouseStyle'] = df_train_reduced2['HouseStyle'].apply(lambda x: convert_catego_numeric(x, list_HouseStyle))
df_train_reduced2['RoofStyle'] = df_train_reduced2['RoofStyle'].apply(lambda x: convert_catego_numeric(x, list_RoofStyle))
df_train_reduced2['RoofMatl'] = df_train_reduced2['RoofMatl'].apply(lambda x: convert_catego_numeric(x, list_RoofMatl))
df_train_reduced2['Exterior1st'] = df_train_reduced2['Exterior1st'].apply(lambda x: convert_catego_numeric(x, list_Exterior1st))
df_train_reduced2['Exterior2nd'] = df_train_reduced2['Exterior2nd'].apply(lambda x: convert_catego_numeric(x, list_Exterior2nd))
df_train_reduced2['MasVnrType'] = df_train_reduced2['MasVnrType'].apply(lambda x: convert_catego_numeric(x, list_MasVnrType))
df_train_reduced2['ExterQual'] = df_train_reduced2['ExterQual'].apply(lambda x: convert_catego_numeric(x, list_ExterQual))
df_train_reduced2['ExterCond'] = df_train_reduced2['ExterCond'].apply(lambda x: convert_catego_numeric(x, list_ExterCond))
df_train_reduced2['Foundation'] = df_train_reduced2['Foundation'].apply(lambda x: convert_catego_numeric(x, list_Foundation))
df_train_reduced2['BsmtQual'] = df_train_reduced2['BsmtQual'].apply(lambda x: convert_catego_numeric(x, list_BsmtQual))
df_train_reduced2['BsmtCond'] = df_train_reduced2['BsmtCond'].apply(lambda x: convert_catego_numeric(x, list_BsmtCond))
df_train_reduced2['BsmtExposure'] = df_train_reduced2['BsmtExposure'].apply(lambda x: convert_catego_numeric(x, list_BsmtExposure))
df_train_reduced2['BsmtFinType1'] = df_train_reduced2['BsmtFinType1'].apply(lambda x: convert_catego_numeric(x, list_BsmtFinType1))
df_train_reduced2['BsmtFinType2'] = df_train_reduced2['BsmtFinType2'].apply(lambda x: convert_catego_numeric(x, list_BsmtFinType2))
df_train_reduced2['Heating'] = df_train_reduced2['Heating'].apply(lambda x: convert_catego_numeric(x, list_Heating))
df_train_reduced2['HeatingQC'] = df_train_reduced2['HeatingQC'].apply(lambda x: convert_catego_numeric(x, list_HeatingQC))
df_train_reduced2['CentralAir'] = df_train_reduced2['CentralAir'].apply(lambda x: convert_catego_numeric(x, list_CentralAir))
df_train_reduced2['Electrical'] = df_train_reduced2['Electrical'].apply(lambda x: convert_catego_numeric(x, list_Electrical))
df_train_reduced2['KitchenQual'] = df_train_reduced2['KitchenQual'].apply(lambda x: convert_catego_numeric(x, list_KitchenQual))
df_train_reduced2['Functional'] = df_train_reduced2['Functional'].apply(lambda x: convert_catego_numeric(x, list_Functional))
df_train_reduced2['GarageType'] = df_train_reduced2['GarageType'].apply(lambda x: convert_catego_numeric(x, list_GarageType))
df_train_reduced2['GarageFinish'] = df_train_reduced2['GarageFinish'].apply(lambda x: convert_catego_numeric(x, list_GarageFinish))
df_train_reduced2['GarageQual'] = df_train_reduced2['GarageQual'].apply(lambda x: convert_catego_numeric(x, list_GarageQual))
df_train_reduced2['GarageCond'] = df_train_reduced2['GarageCond'].apply(lambda x: convert_catego_numeric(x, list_GarageCond))
df_train_reduced2['PavedDrive'] = df_train_reduced2['PavedDrive'].apply(lambda x: convert_catego_numeric(x, list_PavedDrive))
df_train_reduced2['SaleType'] = df_train_reduced2['SaleType'].apply(lambda x: convert_catego_numeric(x, list_SaleType))
df_train_reduced2['SaleCondition'] = df_train_reduced2['SaleCondition'].apply(lambda x: convert_catego_numeric(x, list_SaleCondition))

#scatter plot grlivarea/saleprice

colum2 = ['MSZoning','LotShape','LandContour','LotConfig','LandSlope','Neighborhood','Condition1',
         'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
          'ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
          'Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish',
           'GarageQual','GarageCond','PavedDrive','SaleType','SaleCondition'] 
for var in  colum2:
    data = pd.concat([df_train_reduced2['SalePrice'], df_train_reduced2[var]], axis=1)
    sns.lmplot(var,'SalePrice', data)
    plt.title(var)
plt.show()
df_train_reduced2.describe()
correlations_df = df_train_reduced2.corr() #Compute correlation for each variable

# Plot bar chart (correation against price)
correlations_df['SalePrice'].plot(kind='bar', figsize=(16,8), fontsize=14) 
#Plot correlation between each variables against saleprice
abs(correlations_df['SalePrice']).sort_values(axis=0,ascending=False)
from sklearn.model_selection import train_test_split
## Case 1 OverallQual
X1 = df_train_reduced2[['OverallQual']] # data frame
y1 = df_train_reduced2[['SalePrice']] # data frame
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3)
print(X1_train.shape, y1_train.shape)
print(X1_test.shape, y1_test.shape)
# Create model for case 1
from sklearn.linear_model import LinearRegression
slr1 = LinearRegression()
model1 = slr1.fit(X1_train, y1_train)


# Result for case 1
predictions1  =  model1.predict(X1_test)
score1        =  model1.score(X1_test, y1_test)

print("score: ", score1)
#print(predictions)
# Cross varidation for case 1

from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics

# Perform 5-fold cross validation
scores = cross_val_score(slr1, X1, y1, cv=5)
print("Cross-validated scores:", scores)

# Make cross validated predictions on the test sets
predictions1 = cross_val_predict(slr1, X1, y1, cv=5)
plt.scatter(y1, predictions1)

# manually calcualte the r2
r2 = metrics.r2_score(y1, predictions1)
print("Cross-Predicted R2:", r2)
