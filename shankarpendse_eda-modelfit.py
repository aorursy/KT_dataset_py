import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.preprocessing import normalize

from sklearn.ensemble import GradientBoostingRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test_df = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train_df.shape
test_df.shape
train_df.info()
train_df.head()
train_df.describe(include = 'all')
train_df.count(0) / train_df.shape[0] * 100
train_df.columns
cat_cols = list(train_df.select_dtypes(include='object'))

num_cols = list(train_df.select_dtypes(exclude='object'))
cat_cols
num_cols
for col in cat_cols:

    print(col+" : ", train_df[col].unique(), len(train_df[col].unique()))
corr_matrix = train_df.corr()

plt.subplots(figsize = (12,10))

sns.heatmap(corr_matrix, vmax=1, square = True)
train_df.corr()
# Let's derive the age property at the time of sale, YearSold-YearBuilt and drop these two attributes



train_df['age'] = train_df['YrSold'] - train_df['YearBuilt']

train_df.drop(['YrSold','YearBuilt'], axis = 1, inplace = True)
test_df['age'] = test_df['YrSold'] - test_df['YearBuilt']

test_df.drop(['YrSold', 'YearBuilt'], axis = 1, inplace = True )
# Let's drop the least correlated attributes with saleprice



train_df.drop(['MSSubClass','OverallCond','3SsnPorch','PoolArea','MiscVal','MoSold'], axis = 1, inplace = True)
test_df.drop(['MSSubClass','OverallCond','3SsnPorch','PoolArea','MiscVal','MoSold'], axis = 1, inplace = True)
pd.set_option('display.max_columns',85)

pd.set_option('display.max_rows',85)
train_df.corr()
# Let's drop the least correlated attributes with saleprice

train_df.drop(['BsmtFinSF2','LowQualFinSF','BsmtHalfBath'],axis = 1, inplace = True)
test_df.drop(['BsmtFinSF2','LowQualFinSF','BsmtHalfBath'],axis = 1, inplace = True)
train_df.corr()
corr_matrix = train_df.corr()

plt.subplots(figsize = (12,10))

sns.heatmap(corr_matrix, vmax=1, square = True)
# Let's work on removing multicollinearity

# we can see that TotalBsmtSF and 1stFlrSF are highly correlated,lets retain TotalBsmtSF

# wecan see that 2ndFlrSF and GrLivArea are highly correlated, lets retaiain GrLivArea

# TotRmsAbvGrd is also highly correlated with GrLivArea, lets remove TotRmsAbvGrd

# GarageYrBlt, GarageCars and GarageArea show high collienarity, lets retain only GarageCars



train_df.drop(['1stFlrSF','2ndFlrSF','TotRmsAbvGrd','GarageYrBlt','GarageArea'], axis = 1, inplace = True)



test_df.drop(['1stFlrSF','2ndFlrSF','TotRmsAbvGrd','GarageYrBlt','GarageArea'], axis = 1, inplace = True)
corr_matrix = train_df.corr()

plt.subplots(figsize = (12,10))

sns.heatmap(corr_matrix, vmax=1, square = True)
train_df.shape
test_df.shape
cat_cols = list(train_df.select_dtypes(include='object'))

num_cols = list(train_df.select_dtypes(exclude='object'))
cat_cols
num_cols
for col in cat_cols:

    print(col+" : ", train_df[col].unique(), len(train_df[col].unique()))
# Street: Type of road access to property either gravel or paved, most will choose paved way and can give more weight to this type. Lets replace the values with 1 and 0



train_df['Street'] = train_df['Street'].replace({'Grvl':0, 'Pave': 1})
test_df['Street'] = test_df['Street'].replace({'Grvl':0, 'Pave': 1})
# Alley: Type of alley access to property. This also follows the same logic as street, lets encode accordingly, but lets just put in

# 0 for no access and 1 for Grvl and Paved



train_df['Alley'] = train_df['Alley'].replace( {np.nan:0,'Grvl':1, 'Pave':1 })
test_df['Alley'] = test_df['Alley'].replace( {np.nan:0,'Grvl':1, 'Pave':1 })
train_df['Alley'].unique()
test_df['Alley'].unique()
# LotShape: General shape of the property. Let's categorize this into two types, regular(Reg) and irregular(Irr) then use one-hot encoding



train_df['LotShape'] = train_df['LotShape'].replace( {'Reg':'Reg', 'IR1':'Irr', 'IR2':'Irr', 'IR3':'Irr'} )

train_df = pd.get_dummies(data = train_df, columns = ['LotShape'])
test_df['LotShape'] = test_df['LotShape'].replace( {'Reg':'Reg', 'IR1':'Irr', 'IR2':'Irr', 'IR3':'Irr'} )

test_df = pd.get_dummies(data = test_df, columns = ['LotShape'])
# LandContour: Flatness of the property, it takes 4 different values, but let's change the name to IsLevel and represent with 1 if level and 0 if not



train_df = train_df.rename(columns={'LandContour':'IsLevel'})

train_df['IsLevel'] = train_df['IsLevel'].replace( {'Lvl':1, 'Bnk':0, 'Low':0, 'HLS':0} )
test_df = test_df.rename(columns={'LandContour':'IsLevel'})

test_df['IsLevel'] = test_df['IsLevel'].replace( {'Lvl':1, 'Bnk':0, 'Low':0, 'HLS':0} )
# Utilities: It has only 2 unique values, AllPub & NoSeWa. Let's rename the column to AllUtilities and represent with 1 for all and 0 for partial



train_df = train_df.rename(columns={'Utilities': 'AllUtilities'})

train_df['AllUtilities'] = train_df['AllUtilities'].replace( {'AllPub':1, 'NoSeWa':0} )
test_df = test_df.rename(columns={'Utilities': 'AllUtilities'})

test_df['AllUtilities'] = test_df['AllUtilities'].replace( {'AllPub':1, 'NoSeWa':0} )
# LotConfig: It can take 5 Unique values and it will affect the property price, let's use get_Dummies to represent in one-hot encoding format

train_df = pd.get_dummies(data = train_df, columns= ['LotConfig'])
test_df = pd.get_dummies(data = test_df, columns= ['LotConfig'])
# LandSlope and LandContour looks to be similar, let's drop Landslope from the attributes list



train_df.drop('LandSlope',axis = 1, inplace = True)
test_df.drop('LandSlope',axis = 1, inplace = True)
#Neighborhood : Let's use one hot encoding for this attribute 

train_df = pd.get_dummies(data = train_df, columns= ['Neighborhood'])
test_df = pd.get_dummies(data = test_df, columns= ['Neighborhood'])
# Proximity to various conditions depends on Neighbourhood, so let's drop the columns Condition1 & Condition2



train_df.drop( ['Condition1','Condition2'] ,axis = 1, inplace = True)
test_df.drop( ['Condition1','Condition2'] ,axis = 1, inplace = True)
# BldgType : Let's use Get_dummies

train_df = pd.get_dummies(data = train_df, columns=['BldgType','HouseStyle','RoofStyle','RoofMatl'])
test_df = pd.get_dummies(data = test_df, columns=['BldgType','HouseStyle','RoofStyle','RoofMatl'])
# Exterior1st and Exterior2nd are type of exterior covering on the house and exterior quality "ExterQual" depends on the type of material used

# Let's drop Exterior1st and Exterior2nd and apply pd.get_dummies on ExterQual



train_df.drop( ['Exterior1st','Exterior2nd'] ,axis = 1, inplace = True)

train_df = pd.get_dummies(data = train_df, columns=['ExterQual'])
test_df.drop( ['Exterior1st','Exterior2nd'] ,axis = 1, inplace = True)

test_df = pd.get_dummies(data = test_df, columns=['ExterQual'])
# MasVnrType, ExterCond, Foundation, BsmtQual, BsmtCond, BsmtExposure. Let's handle the null values accordingly and use pd.get_dummies



# Let's treat missing values for MasVnrType as None (Assuming missing values means there is no vnr used) 

train_df['MasVnrType'] = train_df['MasVnrType'].replace( {np.nan:'None'} )

train_df = pd.get_dummies(data = train_df, columns=['MasVnrType'])



# ExterCond has no missing values, lets use get_dummies

train_df = pd.get_dummies(data = train_df, columns=['ExterCond'])



# Foundation has no missing values, lets use get_dummies

train_df = pd.get_dummies(data = train_df, columns=['Foundation'])



# Let's treat missing values for BsmtQual as None (No basement)

train_df['BsmtQual'] = train_df['BsmtQual'].replace( {np.nan:'None'} )

train_df = pd.get_dummies(data = train_df, columns=['BsmtQual'])



# Let's treat missing values for BsmtCond as None (No basement)

train_df['BsmtCond'] = train_df['BsmtCond'].replace( {np.nan:'None'} )

train_df = pd.get_dummies(data = train_df, columns=['BsmtCond'])



# Let's treat missing values for BsmtExposure as None (No basement) in line with the representation for other related attribute

train_df['BsmtExposure'] = train_df['BsmtExposure'].replace( {np.nan:'None'} )

train_df = pd.get_dummies(data = train_df, columns=['BsmtExposure'])



# Let's drop BsmtFinType1 and BsmtFinType2 as these seems to be related to BsmtQual

train_df.drop( ['BsmtFinType1','BsmtFinType2'] ,axis = 1, inplace = True)

# MasVnrType, ExterCond, Foundation, BsmtQual, BsmtCond, BsmtExposure. Let's handle the null values accordingly and use pd.get_dummies



# Let's treat missing values for MasVnrType as None (Assuming missing values means there is no vnr used) 

test_df['MasVnrType'] = test_df['MasVnrType'].replace( {np.nan:'None'} )

test_df = pd.get_dummies(data = test_df, columns=['MasVnrType'])



# ExterCond has no missing values, lets use get_dummies

test_df = pd.get_dummies(data = test_df, columns=['ExterCond'])



# Foundation has no missing values, lets use get_dummies

test_df = pd.get_dummies(data = test_df, columns=['Foundation'])



# Let's treat missing values for BsmtQual as None (No basement)

test_df['BsmtQual'] = test_df['BsmtQual'].replace( {np.nan:'None'} )

test_df = pd.get_dummies(data = test_df, columns=['BsmtQual'])



# Let's treat missing values for BsmtCond as None (No basement)

test_df['BsmtCond'] = test_df['BsmtCond'].replace( {np.nan:'None'} )

test_df = pd.get_dummies(data = test_df, columns=['BsmtCond'])



# Let's treat missing values for BsmtExposure as None (No basement) in line with the representation for other related attribute

test_df['BsmtExposure'] = test_df['BsmtExposure'].replace( {np.nan:'None'} )

test_df = pd.get_dummies(data = test_df, columns=['BsmtExposure'])



# Let's drop BsmtFinType1 and BsmtFinType2 as these seems to be related to BsmtQual

test_df.drop( ['BsmtFinType1','BsmtFinType2'] ,axis = 1, inplace = True)

# Heating, HeatingQC CentralAir None of these have missing values and CentralAir takes only 2 values. Lets use pd.get_dummies for Heating and HeatingQC and binarize CentralAir

train_df = pd.get_dummies(data = train_df, columns=['Heating'])

train_df = pd.get_dummies(data = train_df, columns=['HeatingQC'])

train_df['CentralAir'] = train_df['CentralAir'].replace( {'Y':1, 'N':0} )
test_df = pd.get_dummies(data = test_df, columns=['Heating'])

test_df = pd.get_dummies(data = test_df, columns=['HeatingQC'])

test_df['CentralAir'] = test_df['CentralAir'].replace( {'Y':1, 'N':0} )
train_df.shape
test_df.shape
# Let's check null values for Electrical attribute:

train_df['Electrical'].isnull().sum()
test_df['Electrical'].isnull().sum()
# Let's fill the null value with mode value of this attribute

train_df['Electrical'].fillna(train_df['Electrical'].mode()[0],inplace = True)
train_df['Electrical'].isnull().sum()
# Let's use pd.get_dummies for Electrical

train_df = pd.get_dummies(data = train_df, columns=['Electrical'])
test_df = pd.get_dummies(data = test_df, columns=['Electrical'])
# KitchenQual: Let's drop this as we have overall quality of the house



train_df.drop(['KitchenQual'], axis = 1, inplace = True)
test_df.drop(['KitchenQual'], axis = 1, inplace = True)
train_df.shape
test_df.shape
# Functional some what similar to overallquality, let's drop it



train_df.drop(['Functional'], axis = 1, inplace = True)
test_df.drop(['Functional'], axis = 1, inplace = True)
# FireplaceQu it has null values, lets check how many are there and lets fill them with None (NO Fireplaces)



train_df['FireplaceQu'].isnull().sum()
test_df['FireplaceQu'].isnull().sum()
train_df['FireplaceQu'].fillna('None',inplace = True)
test_df['FireplaceQu'].fillna('None',inplace = True)
train_df['FireplaceQu'].isnull().sum()
test_df['FireplaceQu'].isnull().sum()
train_df = pd.get_dummies(data = train_df, columns=['FireplaceQu'])
test_df = pd.get_dummies(data = test_df, columns=['FireplaceQu'])
# Let's drop GarageType, GarageFinish as we have attributes GarageQual and GarageCond

train_df.drop(['GarageType','GarageFinish'], axis = 1, inplace = True)
test_df.drop(['GarageType','GarageFinish'], axis = 1, inplace = True)
# Let's check the count of null values for GarageQual and GarageCond

print(train_df['GarageQual'].isnull().sum())

print(train_df['GarageCond'].isnull().sum())
print(test_df['GarageQual'].isnull().sum())

print(test_df['GarageCond'].isnull().sum())
print(train_df['GarageQual'].value_counts())

print(train_df['GarageCond'].value_counts())
train_df[['GarageQual','GarageCond', 'SalePrice']].groupby(['GarageQual','GarageCond'])['SalePrice'].mean()
train_df['SalePrice'].where( (train_df['GarageCond'].isnull()) & (train_df['GarageQual'].isnull()) ).mean()
train_df['GarageQual'].fillna('Po',inplace = True)

train_df['GarageCond'].fillna('Po', inplace = True)
test_df['GarageQual'].fillna('Po',inplace = True)

test_df['GarageCond'].fillna('Po', inplace = True)
print(train_df['GarageQual'].isnull().sum())

print(train_df['GarageCond'].isnull().sum())
print(test_df['GarageQual'].isnull().sum())

print(test_df['GarageCond'].isnull().sum())
# Now that we have handled Null values for GarageCond and GarageQual attributes, lets use get_dummies

train_df = pd.get_dummies(data = train_df, columns=['GarageQual','GarageCond'])
test_df = pd.get_dummies(data = test_df, columns=['GarageQual','GarageCond'])
train_df.shape
test_df.shape
# PavedDrive is having 3 unique values ('Y' 'N' 'P'), lets change it to represent either paved or not paved with 1 and 0 respectively

train_df['PavedDrive'] = train_df['PavedDrive'].replace( {'Y':1,'N':0,'P':0} )
test_df['PavedDrive'] = test_df['PavedDrive'].replace( {'Y':1,'N':0,'P':0} )
# Since we have dropped PoolArea, let's drop PoolQC as well



train_df.drop('PoolQC', axis = 1, inplace = True)



# Fence: Represents fence quality (nan 'MnPrv' 'GdWo' 'GdPrv' 'MnWw') which is also a measure of Privacy, lets modify it to represent Complete Privacy (CP), Partial Privacy(PP) and No Privacy(NP)

train_df['Fence'] = train_df['Fence'].replace( {'GdPrv':'CP', 'MnPrv':'PP', 'GdWo':'PP', 'MnWw':'NP', np.nan:'NP'} )

train_df = pd.get_dummies(data = train_df, columns=['Fence'])



# We have dropped MiscValue, so let's drop the attribute MiscFeature

train_df.drop('MiscFeature', axis = 1, inplace = True)

test_df.drop('PoolQC', axis = 1, inplace = True)



# Fence: Represents fence quality (nan 'MnPrv' 'GdWo' 'GdPrv' 'MnWw') which is also a measure of Privacy, lets modify it to represent Complete Privacy (CP), Partial Privacy(PP) and No Privacy(NP)

test_df['Fence'] = test_df['Fence'].replace( {'GdPrv':'CP', 'MnPrv':'PP', 'GdWo':'PP', 'MnWw':'NP', np.nan:'NP'} )

test_df = pd.get_dummies(data = test_df, columns=['Fence'])



# We have dropped MiscValue, so let's drop the attribute MiscFeature

test_df.drop('MiscFeature', axis = 1, inplace = True)
# let's do some checks on SaleType and SaleCondition



train_df.boxplot(column = 'SalePrice', by = 'SaleType', rot=30)
train_df.boxplot(column = 'SalePrice', by = 'SaleCondition', rot=30)
# We can not clearly makeout the pattern based on SaleCondition and SaleType except for Normal and warranty deed for which we can see a maximum sales

# Let's drop these attributes for now:



train_df.drop('SaleCondition',axis = 1, inplace = True)

train_df.drop('SaleType',axis = 1, inplace = True)

test_df.drop('SaleCondition',axis = 1, inplace = True)

test_df.drop('SaleType',axis = 1, inplace = True)
train_df.shape
test_df.shape
# Now we are left with only one categorical column MSZoning, let's work on that



train_df['MSZoning'].value_counts()
plt.bar(train_df['MSZoning'],train_df['SalePrice'])

plt.show()
# As we can see, the SalePrice varies as per the Zone classification, we must include this in our model design

train_df = pd.get_dummies(data = train_df, columns=['MSZoning'])
test_df = pd.get_dummies(data = test_df, columns=['MSZoning'])
train_df.shape
test_df.shape
train_df.isnull().sum()
test_df.isnull().sum()
# We have missing values for LotFrontage, let's see how we can fill those.

# Let's assume, LotFrontage will be dependent on LotArea,Alley,LotShape_Irr,LotShape_Reg, IsLevel, LotConfig_Corner, LotConfig_CulDSac, LotConfig_FR2, LotConfig_FR3, LotConfig_Inside

# let's Check the correlation among them



train_df[['LotArea','Alley','LotShape_Irr','LotShape_Reg','IsLevel','LotConfig_Corner','LotConfig_CulDSac','LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside','LotFrontage']].corr()
train_df['LotFrontage'].describe()
train_df['LotFrontage'].hist()
train_df['LotFrontage'].fillna(train_df['LotFrontage'].mean(), inplace = True)
test_df['LotFrontage'].fillna(test_df['LotFrontage'].mean(), inplace = True)
pd.set_option('display.max_rows',170)
train_df.isnull().sum()
# There are 8 Null values for MasVnrArea, lets check the distribution



train_df['MasVnrArea'].describe()
plt.scatter(train_df['MasVnrArea'],train_df['SalePrice'])
train_df[train_df.columns[1:]].corr()['SalePrice'][:]
sns.scatterplot(x = 'OverallQual', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'age', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'MasVnrArea', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'BsmtFinSF1', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'TotalBsmtSF', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'GrLivArea', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'FullBath', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'GarageCars', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'LotFrontage', y = 'SalePrice', data=train_df)
sns.scatterplot(x = 'LotArea', y = 'SalePrice', data=train_df)
train_df.drop( train_df[ (train_df['OverallQual'] > 9) & (train_df['SalePrice'] < 200000) ].index, inplace = True)
train_df.drop( train_df[ (train_df['age'] > 100) & (train_df['SalePrice'] > 300000) ].index, inplace = True)
train_df.drop( train_df[ (train_df['MasVnrArea'] > 1200) & (train_df['SalePrice'] < 700000) ].index, inplace = True)
train_df.drop( train_df[ (train_df['BsmtFinSF1'] > 2000) & (train_df['SalePrice'] < 200000) ].index, inplace = True)
train_df.drop( train_df[ (train_df['TotalBsmtSF'] > 3000) & (train_df['SalePrice'] < 300000) ].index, inplace = True)
train_df.drop( train_df[ (train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000) ].index, inplace = True)
train_df.drop( train_df[ (train_df['FullBath'] < 1) & (train_df['SalePrice'] > 300000) ].index, inplace = True)
train_df.drop( train_df[ (train_df['LotFrontage'] > 200) & (train_df['SalePrice'] < 300000) ].index, inplace = True)
train_df.shape
train_df.isnull().sum()
train_df['MasVnrArea'].fillna(train_df['MasVnrArea'].mean(), inplace = True)
test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mean(), inplace = True)
train_df.head()
train_df_p = pd.DataFrame(normalize(train_df.drop(['Id','SalePrice'], axis = 1)), columns = train_df.drop(['Id','SalePrice'],axis = 1).columns)
test_df.head()
test_df.isnull().sum()
test_df['AllUtilities'].fillna(test_df['AllUtilities'].mean(), inplace = True)
test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean(), inplace = True)
test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean(), inplace = True)
test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean(), inplace = True)
test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mean(), inplace = True)
test_df['GarageCars'].fillna(test_df['GarageCars'].mean(), inplace = True)
test_df_p = pd.DataFrame(normalize(test_df.drop('Id', axis = 1)), columns = test_df.drop('Id',axis = 1).columns)
train_df_p.head()
test_df_p.head()
train_df_p.reset_index(inplace = True)
train_df.reset_index(inplace = True)
train_df_p['SalePrice'] = train_df['SalePrice']
train_df_p.shape
train_df_p.head()
train_df_p.drop('index', axis = 1, inplace = True)
train_df_p.shape
train_df_p.head()
test_df_p.head()
# Let's get the difference in columns between train_df_p and test_df_p 

train_df_p.columns.difference(test_df_p.columns)   
# We will have to train our model keeping attributes from Test set as well in mind. Let's drop the above attributes from train_df_p

train_df_p.drop(['Electrical_Mix', 'GarageQual_Ex', 'Heating_Floor', 'Heating_OthW','HouseStyle_2.5Fin', 'RoofMatl_ClyTile', 'RoofMatl_Membran','RoofMatl_Metal', 'RoofMatl_Roll'], axis = 1, inplace = True)
# Let's again check the attributes from both sets (train and test)

print(train_df_p.shape)

print(test_df_p.shape)
# Let's get the difference in columns between train_df_p and test_df_p 

train_df_p.columns.difference(test_df_p.columns)
train_df_p.head()
test_df_p.head()
sns.distplot(train_df_p['SalePrice']).set_title("Distribution of SalePrice")
# probability plot

fig = plt.figure()

res = stats.probplot(train_df_p['SalePrice'], plot=plt)
#Using the log1p function applies log(1+x) to all elements of the column

train_df_p["SalePrice"] = np.log1p(train_df_p["SalePrice"])
#Check the new distribution after log transformation 

sns.distplot(train_df_p['SalePrice'] )
# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train_df_p['SalePrice'])

print( '\n mean = {:.2f} and std dev = {:.2f}\n'.format(mu, sigma))
fig = plt.figure()

res = stats.probplot(train_df_p['SalePrice'], plot=plt)

plt.show()
train_df_p.isnull().sum()
X_train, X_val, y_train, y_val = train_test_split(train_df_p.drop('SalePrice', axis = 1), train_df_p['SalePrice'], test_size = 0.33, shuffle = True, random_state = 42)
X_train.shape
X_val.shape
y_train.shape
y_val.shape
LRM = LinearRegression()
LRM.fit(X_train,y_train)
LRM.score(X_val,y_val)
predictions = LRM.predict(X_val)
print(mean_squared_error(y_val,predictions))
RFM = RandomForestRegressor(max_depth = 10, n_estimators = 1500, random_state = 42)
RFM.fit(X_train,y_train)
RFM.score(X_val,y_val)
predictions = RFM.predict(X_val)
print(mean_squared_error(y_val,predictions))
GBR = GradientBoostingRegressor(max_depth = 3, n_estimators = 1500, verbose = 1, random_state = 42)
GBR.fit(X_train,y_train)
GBR.score(X_val,y_val)
predictions_GBR = GBR.predict(X_val)
print(mean_squared_error(y_val,predictions_GBR))
# Now Let's get predictions on test_df_p, before to that, let's check the shape of test_df and test_df_p 

# because we need Id from test_df tobe appended to predictions set



print(test_df.shape)

print(test_df_p.shape)
predictions_test = GBR.predict(test_df_p)
test_df.head()
len(predictions_test)
predictions_test.max()
predictions_test = np.expm1(predictions_test)
predictions_test.min()
predictions_test.max()
submission = test_df[['Id']].copy()
submission['SalePrice'] = predictions_test
submission.head()
submission.to_csv('submission.csv', index=False, header=True)