# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')
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
#To Display all the columns in the Dataset w/o trimming.

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
#importing both datasets to python notebook

#Loading the Train Dataset:
house_pred = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
#Loading the Test Dataset:
housing_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
#Sample records of the Test Dataset:
house_pred.head()
#Sample records of the Test Dataset:
housing_test.head()
#Rows & Columns info:
house_pred.shape
#Columnwise info:
house_pred.info()
#Numerical features describing:
house_pred.describe()
#Describing Categorical features:
house_pred.describe(include='object')
#Checking the Null values in dataframe rows:
house_pred.isnull().sum(axis =1).any()
#Checking the Null percentages for all the features:
missing_percentages = round(100*(house_pred.isnull().sum()/len(house_pred.index)), 2)
#Plotting Missing values in each column:
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,10))
sns.heatmap(house_pred.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Printing Missing Percentages
pd.options.display.max_rows = None
print(missing_percentages.sort_values(ascending = False))
#Drop the columns having more than 30% missing values.
house_pred = house_pred.drop(house_pred.loc[:,list(missing_percentages> 30)].columns,1)
#Missing Percentages after dropping huge missing valued columns:
mis_perc2 = round(100*(house_pred.isnull().sum()/len(house_pred.index)),2)
mis_perc2.sort_values(ascending = False)
#List of Categorical fields:
house_pred_cat = house_pred.select_dtypes('object')
cat_cols = list(house_pred_cat.columns)
print(type(cat_cols))
cat_cols
#Plotting Missing Categorical fields only:
from matplotlib.gridspec import GridSpec

fig=plt.figure(figsize=(20,20))

gs=GridSpec(4,3) # 4 rows, 3 columns
ax1=fig.add_subplot(gs[0,0]) # First row, first column
sns.countplot(house_pred['GarageFinish'].replace(np.nan, 'No Garage')) 
ax2=fig.add_subplot(gs[0,1]) # First row, second column
sns.countplot(house_pred['GarageType'].replace(np.nan, 'No Garage')) 
ax3=fig.add_subplot(gs[0,2]) # First row, thrid column
sns.countplot(house_pred['GarageCond'].replace(np.nan, 'No Garage')) 
ax4=fig.add_subplot(gs[1,0]) # second row, first column
sns.countplot(house_pred['GarageQual'].replace(np.nan, 'No Garage')) 
ax6=fig.add_subplot(gs[1,2]) 
sns.countplot(house_pred['BsmtExposure'].replace(np.nan, 'No Garage')) 
ax7=fig.add_subplot(gs[2,0]) # Third row, first column
sns.countplot(house_pred['BsmtFinType2'].replace(np.nan, 'No Basement')) 
ax8=fig.add_subplot(gs[2,1])
sns.countplot(house_pred['BsmtFinType1'].replace(np.nan, 'No Basement')) 
ax9 = fig.add_subplot(gs[2,2]) 
sns.countplot(house_pred['BsmtCond'].replace(np.nan, 'No Basement')) 
ax10=fig.add_subplot(gs[3,0]) # 4th row
sns.countplot(house_pred['BsmtQual'].replace(np.nan, 'No Basement')) 
ax11=fig.add_subplot(gs[3,1]) 
sns.countplot(house_pred['MasVnrType'].replace(np.nan, 'None'))
ax11=fig.add_subplot(gs[3,2]) 
sns.countplot(house_pred['Electrical'].replace(np.nan, 'None')) 
plt.tight_layout()
#cols=['PoolQC', 'MiscFeature','Alley','Fence','FireplaceQu','LotFrontage','GarageCond','GarageType','GarageYrBlt','GarageFinish','GarageQual','BsmtExposure','BsmtFinType2','BsmtFinType1','BsmtCond','BsmtQual','MasVnrArea','MasVnrType','Electrical']

house_pred['GarageFinish'].fillna('No Garage', inplace = True)

house_pred['GarageType'].fillna('No Garage',inplace = True)

house_pred['GarageCond'].fillna('No Garage',inplace = True)

house_pred['GarageQual'].fillna('No Garage',inplace = True)

house_pred['BsmtExposure'].fillna('No Basement',inplace = True)

house_pred['BsmtFinType2'].fillna('No Basement',inplace = True)

house_pred['BsmtFinType1'].fillna('No Basement',inplace = True)

house_pred['BsmtCond'].fillna('No Basement',inplace = True)

house_pred['BsmtQual'].fillna('No Basement',inplace = True)

house_pred['MasVnrType'].fillna('None',inplace = True)


#Shape after some columns drop:
house_pred.shape
#Missing Percentages after dropping huge missing valued columns:
mis_perc3 = round(100*(house_pred.isnull().sum()/len(house_pred.index)),2)
mis_perc3.sort_values(ascending = False)
#Dropped only the row which have missing value
house_pred.dropna(subset=['Electrical'],inplace=True)
#Plotting 'LotFrontage' Distrbution:
plt.grid(color='red', linestyle='-', linewidth=0.5)
sns.distplot(house_pred['LotFrontage'].dropna(),kde=False,color='green',bins = 75)
#Boxplot to find out the range of LotFrontage:
sns.set_style('whitegrid')
plt.figure(figsize = (6,4))
sns.boxplot(y='LotFrontage',data=house_pred,palette='winter')
#Replacing missing with 70 feet.
house_pred['LotFrontage'].fillna(70,inplace = True)
#Boxplot for GarageYrBlt
sns.set_style('whitegrid')
plt.figure(figsize = (6,6))
sns.boxplot(x='GarageType',y='GarageYrBlt',data=house_pred,palette='winter')
#Replacing GarageYrBlt with 0 
house_pred['GarageYrBlt'].fillna(0,inplace = True)
#Boxplot for MasVnrArea:
sns.set_style('whitegrid')
plt.figure(figsize = (6,6))
sns.boxplot(x='MasVnrType',y='MasVnrArea',data=house_pred,palette='winter')
#
plt.figure(figsize=(10,5));
plt.xlabel('xlabel', fontsize=16);
plt.rc('xtick', labelsize=14); 
plt.rc('ytick', labelsize=14); 
sns.distplot(house_pred['MasVnrArea'].dropna());
#Replacing GarageYrBlt with 0 
house_pred.drop('MasVnrArea', axis=1, inplace=True)
#house_pred['MasVnrArea'].fillna(200,inplace = True)
house_pred.shape
#To visualize the null values in columns, if any!
plt.figure(figsize=(20,10))
sns.heatmap(house_pred.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Now drop the 'Id' column since it's unnecessary for the prediction process.
house_pred.drop("Id", axis = 1, inplace = True)
#Group the columns based on dtypes:
grouped_hcols = house_pred.columns.to_series().groupby(house_pred.dtypes).groups
grouped_hcols
#Count of Numerical fields:
house_pred_int = house_pred.select_dtypes(include = ['int64','float64'])
len(house_pred_int.columns)
#Count of Categorical fields:
house_pred_cat = house_pred.select_dtypes('object')
len(house_pred_cat.columns)
#plotting first few set of the Numerical columns:
sns.set()
cols = ['MSSubClass','LotArea','OverallQual', 'OverallCond','LotFrontage', 'GarageYrBlt',
        'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']
sns.pairplot(house_pred_int[cols], size = 2.5)
plt.show();
#Second set of Numerical features:
sns.set()
cols = [ 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
        'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr']
sns.pairplot(house_pred[cols], size = 2.5)
plt.show();
#Third set of Numerical features:
sns.set()
cols = [ 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
        'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch']
sns.pairplot(house_pred[cols], size = 2.5)
plt.show();
#Rest of the Numerical features
sns.set()
cols = [ '3SsnPorch',
        'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold', 'MoSold','SalePrice']
sns.pairplot(house_pred[cols], size = 2.5)
plt.show();
#Few set of Categorical features which need to be imputed:
from matplotlib.gridspec import GridSpec

fig=plt.figure(figsize=(20,20))

gs=GridSpec(5,3) # 4 rows, 3 columns
ax1=fig.add_subplot(gs[0,0]) # First row, first column
sns.countplot(house_pred['MSSubClass']) 
ax2=fig.add_subplot(gs[0,1]) # First row, second column
sns.countplot(house_pred['OverallQual']) 
ax3=fig.add_subplot(gs[0,2]) # First row, thrid column
sns.countplot(house_pred['OverallCond']) 
ax4=fig.add_subplot(gs[1,0])  # second row, first column
sns.countplot(house_pred['ExterQual']) 
ax6=fig.add_subplot(gs[1,2]) 
sns.countplot(house_pred['ExterCond']) 

ax7=fig.add_subplot(gs[2,0]) # Third row, first column
sns.countplot(house_pred['BsmtQual']) 
ax8=fig.add_subplot(gs[2,1])
sns.countplot(house_pred['BsmtExposure']) 
ax9 = fig.add_subplot(gs[2,2]) 
sns.countplot(house_pred['BsmtCond']) 

ax10=fig.add_subplot(gs[3,0]) # 4th roww, first column
sns.countplot(house_pred['BsmtQual']) 
ax11=fig.add_subplot(gs[3,1]) 
sns.countplot(house_pred['Heating'])
ax12=fig.add_subplot(gs[3,2]) 
sns.countplot(house_pred['KitchenQual']) 

ax13=fig.add_subplot(gs[4,0]) #5th roww, first column
sns.countplot(house_pred['GarageQual']) 
ax14=fig.add_subplot(gs[4,2]) 
sns.countplot(house_pred['GarageCond']) 

plt.tight_layout()
#Trying to reduce No.of sub categories from 10 to 3 (for below features):
house_pred['OverallQual'].replace({1:1,2:1,3:1,4:1,5:2,6:3,7:3,8:3,9:3,10:3 }, inplace = True)
house_pred['OverallCond'].replace({1:1,2:1,3:1,4:1,5:2,6:3,7:3,8:3,9:3,10:3 }, inplace = True)
sns.countplot(house_pred['OverallQual']) 
sns.countplot(house_pred['OverallCond'])
# From above Pairplot's we can depict these respective columns from each Graph need to converted into object type:
cols_conv = ['MSSubClass','OverallQual', 'OverallCond','BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr','KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces']
house_pred[cols_conv] = house_pred[cols_conv].astype('object')
# Now drop the 'Id' column since it's unnecessary for the prediction process.
house_pred.drop("MoSold", axis = 1, inplace = True)
#Seperation of Features:
categorical_features = house_pred.select_dtypes(include = ["object"]).columns
numerical_features = house_pred.select_dtypes(exclude = ["object"]).columns
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
house_pred_num = house_pred[numerical_features]
house_pred_cat = house_pred[categorical_features]
#List of categorical_features:
categorical_features
#List of numerical_features:
numerical_features
#After dropping total no.of features till now:
len(house_pred.columns)
#Checking the Target feature data type:
house_pred['SalePrice'].dtype
from scipy import stats
from scipy.stats import norm, skew #for some statistics
#To see the skewness on Numeric features:
skewness = house_pred_num.apply(lambda x: skew(x))
skewness.sort_values(ascending=False)
#Filtering features based on Skewness above 0.8
skewness = skewness[abs(skewness)>0.8]
skewness.index
#Total features having skewness:
len(skewness.index)
#Checking categorical features count:
house_pred_cat.shape
type(house_pred.nunique())
#Unique sub categories in all categorical columns:
house_pred.nunique().sort_values(ascending=False)
#Countplots for 2-4 subcategory features:
sns.set(style="darkgrid")
fig, axs = plt.subplots(6,4, figsize = (30,18))
plt4 = sns.countplot(house_pred['GarageFinish'], ax = axs[0,0])
plt5 = sns.countplot(house_pred['GarageType'], ax = axs[0,1])
plt26 = sns.countplot(house_pred['Fireplaces'], ax = axs[0,2])
plt25 = sns.countplot(house_pred['KitchenQual'], ax = axs[0,3])

plt6 = sns.countplot(house_pred['GarageCond'], ax = axs[1,0])
plt7 = sns.countplot(house_pred['GarageQual'], ax = axs[1,1])
plt8 = sns.countplot(house_pred['BsmtExposure'], ax = axs[1,2])
plt9 = sns.countplot(house_pred['BsmtFinType2'], ax = axs[1,3])


plt11 = sns.countplot(house_pred['BsmtCond'], ax = axs[2,0])
plt12 = sns.countplot(house_pred['BsmtQual'], ax = axs[2,1])
plt13 = sns.countplot(house_pred['MasVnrType'], ax = axs[2,2])
plt14 = sns.countplot(house_pred['Street'], ax = axs[2,3])

plt16 = sns.countplot(house_pred['LandContour'], ax = axs[3,0])
plt17 = sns.countplot(house_pred['Utilities'], ax = axs[3,1])
plt18 = sns.countplot(house_pred['LandSlope'], ax = axs[3,2])
plt19 = sns.countplot(house_pred['CentralAir'], ax = axs[3,2])


plt21 = sns.countplot(house_pred['FullBath'], ax = axs[4,0])
plt22 = sns.countplot(house_pred['BsmtHalfBath'], ax = axs[4,1])
plt23 = sns.countplot(house_pred['HalfBath'], ax = axs[4,2])
plt24= sns.countplot(house_pred['KitchenAbvGr'], ax = axs[4,3])



plt10 = sns.countplot(house_pred['BsmtFinType1'], ax = axs[5,0])
plt15 = sns.countplot(house_pred['LotShape'], ax = axs[5,1])
plt20 = sns.countplot(house_pred['BsmtFullBath'], ax = axs[5,2])

plt.tight_layout()
# Value counts for the 2-4 subcategory features:
print(house_pred['Street'].value_counts())
print(house_pred['LotShape'].value_counts())
print(house_pred['LandContour'].value_counts())
print(house_pred['Utilities'].value_counts())
print(house_pred['LandSlope'].value_counts())
print(house_pred['MasVnrType'].value_counts())
print(house_pred['ExterQual'].value_counts())
print(house_pred['BsmtQual'].value_counts())
print(house_pred['BsmtCond'].value_counts())
print(house_pred['BsmtExposure'].value_counts())
print(house_pred['CentralAir'].value_counts())
print(house_pred['BsmtFullBath'].value_counts())
print(house_pred['BsmtHalfBath'].value_counts())
print(house_pred['FullBath'].value_counts())
print(house_pred['HalfBath'].value_counts())
print(house_pred['KitchenAbvGr'].value_counts())
print(house_pred['KitchenQual'].value_counts())
print(house_pred['Fireplaces'].value_counts())
print(house_pred['GarageFinish'].value_counts())
print(house_pred['PavedDrive'].value_counts())


house_pred.columns
house_pred.head
house_pred.shape
#Creating new features
house_pred['HouseAge'] =  house_pred['YrSold'] - house_pred['YearBuilt']
house_pred['HasGarage'] = house_pred['GarageArea'].apply(lambda x:1 if x > 0 else 0)
house_pred['TotalBath']= house_pred['FullBath'] + (0.5 * house_pred['HalfBath'] )+ house_pred['BsmtFullBath'] + (0.5 * house_pred['FullBath'])
house_pred['TotalSqrFt'] =  house_pred['1stFlrSF'] + house_pred['2ndFlrSF'] + house_pred['GarageArea'] + house_pred['WoodDeckSF'] + house_pred['OpenPorchSF']
house_pred['Has2ndFloor'] = house_pred['2ndFlrSF'].apply(lambda x:1 if x > 0 else 0)
house_pred['TotalBath']
from datetime import date
# Converting year to number of years:
current_year = date.today().year
house_pred['YearsSinceHouseSold'] = current_year - house_pred['YrSold']
house_pred['YearsSinceHouseBuilt']=current_year-house_pred['YearBuilt']
house_pred['YearsSinceLastRemodeled']= current_year-house_pred['YearRemodAdd']
house_pred['YearsSinceGarageYrBlt']=current_year-house_pred['GarageYrBlt']

#After converting from actual year to age for 'YrSold':
house_pred['YearsSinceHouseSold']
house_pred['Has2ndFloor'].dtype
house_pred['TotalBath'].dtype
house_pred['HouseAge'].dtype
house_pred['HasGarage'].dtype
house_pred['TotalSqrFt'].dtype
house_pred['TotalBath'] = house_pred['TotalBath'].astype('float')
house_pred['TotalBath'].dtype
house_pred.shape
# Dropping columns that are highly skewed:
house_pred.drop(['Street','LandContour','Utilities','LandSlope','BsmtCond','CentralAir','BsmtHalfBath','KitchenAbvGr','PavedDrive'],inplace=True,axis=1)
house_pred.shape
house_pred.drop(['YrSold','YearBuilt','YearRemodAdd','GarageYrBlt'],inplace=True,axis=1)
house_pred.shape
#correlation matrix
corr_all = house_pred.corr()
f, ax = plt.subplots(figsize=(20, 19))
sns.heatmap(corr_all, vmax=.8, annot=True,cmap="YlGnBu");
#Most Correlation with SalePrice
corr = house_pred.corr()
print(corr['SalePrice'].sort_values(ascending=False))
#Describing House SalesPrice:
house_pred['SalePrice'].describe()
# Lets visualize Top 10 Most correlated features with SalesPrice
k = 10 #number of variables for heatmap
cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(house_pred[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values,cmap="YlGnBu")
plt.show()
#Top 10 correlated features with SalePrice
top_corr = pd.DataFrame(cols)
top_corr.columns = ['Top Correlatied Features for SalePrice']
top_corr
#scatterplot for these top correlated features:
sns.set()
cols = ['SalePrice','TotalSqrFt','GrLivArea', 'GarageCars', 'TotalBsmtSF', '1stFlrSF','BsmtFinSF1','LotFrontage']
sns.pairplot(house_pred[cols], size = 2.5)
plt.show();
#Plotting  'TotalSqrFt' vs SalePrice:
fig, ax = plt.subplots()
ax.scatter(x = house_pred['TotalSqrFt'], y = house_pred['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotalSqrFt', fontsize=13)
plt.show()
#Treating outliers for GrLivArea
house_pred = house_pred.drop(house_pred[(house_pred['TotalSqrFt']>6000) & (house_pred['SalePrice']<300000)].index).reset_index(drop=True)

#Plotting Jointplot for '1stFlrSF' vs SalePrice:
sns.jointplot(x=house_pred['TotalSqrFt'], y = house_pred['SalePrice'],kind='reg')
fig, ax = plt.subplots()
ax.scatter(x = house_pred['GrLivArea'], y = house_pred['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
#Treating outliers for GrLivArea
house_pred = house_pred.drop(house_pred[(house_pred['GrLivArea']>4000) & (house_pred['SalePrice']<300000)].index).reset_index(drop=True)

#Only Plotting (No ) by applying log transformation on 'GrLivArea' w.r.to SalePrice
sns.jointplot(x=np.log(house_pred['GrLivArea']), y = house_pred['SalePrice'],kind='reg')
# Checking there is no log transformation on 'GrLivArea'.
house_pred['GrLivArea']
#GarageCars vs SalePrice 
#Plotting How HousePrice can be increased with Quality:
f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=house_pred['GarageCars'], y = house_pred['SalePrice'])
#Treating of Outliers on observations: 
# More than 4-Cars & SalePrice less than 3 lakhs:

house_pred = house_pred.drop(house_pred[(house_pred['GarageCars'] > 3) & (house_pred['SalePrice']<300000)].index).reset_index(drop=True)
#Plotting GarageCars: 
f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=house_pred['GarageCars'], y = house_pred['SalePrice'])
#GarageArea vs SalePrice 
fig, ax = plt.subplots()
ax.scatter(x = house_pred['GarageArea'], y = house_pred['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GarageArea', fontsize=13)
plt.show()
#Treating outliers
house_pred = house_pred.drop(house_pred[(house_pred['GarageArea']>1000) & (house_pred['SalePrice']<300000)].index).reset_index(drop=True)

#Jointplot for GarageArea & SalePrice:
sns.jointplot(x=house_pred['GarageArea'], y = house_pred['SalePrice'],kind='reg')
#Plotting Jointplot for'TotalBsmtSF' vs SalePrice:
sns.jointplot(x=house_pred['TotalBsmtSF'], y = house_pred['SalePrice'],kind='reg')
#Plotting Jointplot for '1stFlrSF' vs SalePrice:
sns.jointplot(x=house_pred['1stFlrSF'], y = house_pred['SalePrice'],kind='reg')
#Plotting Jointplot for 'TotalBsmtSF' vs SalePrice:
fig, ax = plt.subplots()
ax.scatter(x = house_pred['TotalBsmtSF'], y = house_pred['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('TotalBsmtSF', fontsize=13)
plt.show()
#Plotting Jointplot for 'LotFrontage' vs SalePrice:
fig, ax = plt.subplots()
ax.scatter(x = house_pred['LotFrontage'], y = house_pred['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('LotFrontage', fontsize=13)
plt.show()
#Treating outlier for 'LotFrontage':
house_pred = house_pred.drop(house_pred[(house_pred['LotFrontage']>200) & (house_pred['SalePrice']<300000)].index).reset_index(drop=True)
#Plotting Jointplot for 'LotFrontage' & 'SalePrice'
sns.jointplot(x=house_pred['LotFrontage'], y = house_pred['SalePrice'],kind='reg')
#Plotting 'YearsSinceHouseSold' distribution:
sns.distplot( house_pred['YearsSinceHouseSold'],bins = 15)
#Box Plotting for YrSold vs SalePrice:
f, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='YearsSinceHouseSold',y='SalePrice',data=house_pred)
#len(house_pred_num)
categorical_features = house_pred.select_dtypes(include = ["object"]).columns
numerical_features = house_pred.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("SalePrice")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
house_pred_num = house_pred[numerical_features]
house_pred_cat = house_pred[categorical_features]
#SalePrice Normality check:
sns.distplot(house_pred['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(house_pred['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
#Probability Plot check for 'SalePrice'
fig = plt.figure()
res = stats.probplot(house_pred['SalePrice'], plot=plt)
plt.show()
#After applying log Transformation Normality Check 
from scipy.stats import norm
from scipy import stats
#histogram and normal probability plot
house_pred["SalePrice"] = np.log1p(house_pred["SalePrice"])
sns.distplot(house_pred.SalePrice, fit=norm);
fig = plt.figure()
#Probability Plot after applying log transformation:
res = stats.probplot(house_pred.SalePrice, plot=plt)
#Checking any nulls in categorical:
str(house_pred_cat.isnull().values.sum())
#Packages To Import for modeling:
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
#categorical_features for modeling:
categorical_features = house_pred.select_dtypes(include = ["object"]).columns
categorical_features
#numerical_features for modeling:
numerical_features
house_pred.shape
# Creating a dummy variable for some of the categorical variables and dropping the first one.
house_pred_catd = pd.get_dummies(house_pred[['MSSubClass', 'MSZoning', 'LotShape', 'LotConfig', 'Neighborhood',
       'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual',
       'OverallCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
       'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
       'Electrical', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
       'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'GarageType',
       'GarageFinish', 'GarageQual', 'GarageCond', 'SaleType',
       'SaleCondition']], drop_first=True)
house_pred_catd.head()
# house_pred_train is a combination of Dummy categorical features & Numerical features:
house_pred_train = pd.concat([house_pred_catd,house_pred_num],axis=1)
house_pred_train.shape
# house_pred_train features:
print(house_pred_train.columns)
# Target feature:
y = house_pred['SalePrice']
#standardizing data
house_pred_scaled = StandardScaler().fit_transform(house_pred_train);
low_range = house_pred_scaled[house_pred_scaled[:,0].argsort()][:10]
high_range= house_pred_scaled[house_pred_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
# house_pred_train features:
house_pred_train.head()
#split the data to train & test for the model 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(house_pred_train,y,test_size = 0.3,random_state= 0)

#Shape of Train & Test Data:
X_train.shape,X_test.shape,y_train.shape,y_test.shape
from sklearn import metrics
def test(models,data,iterations = 100):
    results = {}
    for i in models:
        r2_train = []
        r2_test = []
        for j in range(iterations):
            X_train,X_test,y_train,y_test = train_test_split(house_pred_train,y,test_size = 0.3,random_state= 0)
            r2_test.append(metrics.r2_score(y_test,models[i].fit(X_train,y_train).predict(X_test)))
            r2_train.append(metrics.r2_score(y_train,models[i].fit(X_train,y_train).predict(X_train)))
        results[i] = [np.mean(r2_train),np.mean(r2_test)]
    return pd.DataFrame(results)
models = { 'Lasso': linear_model.Lasso(),
           'Ridge': linear_model.Ridge(),
}
test(models,house_pred_train)
# list of alphas to tune
params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20.0,30.0, 40.0, 50.0,60.0,80.0,100.0,200.0,500.0,1000.0]}


ridge = Ridge()

# cross validation
folds = 5
#picking the best alpha using GridSearchCV.
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train, y_train) 
#checking the value of optimum number of parameters
print(model_cv.best_params_)
print(model_cv.best_score_)
#cv_results 
alpha = 10.0
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results = cv_results[cv_results['param_alpha']<=alpha]
cv_results.head()
# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='lower right')
plt.show()
#Best Alpha:
alpha
#Ridge Fitting  
ridge = Ridge(alpha=alpha)
#Priting Ridge Regression Coefficients
ridge.fit(X_train, y_train)
ridge.coef_
#checking the value of optimum number of parameters
print(model_cv.best_params_)
print(model_cv.best_score_)
#predicting Ridge R2 scores:
from sklearn.metrics import r2_score
y_train_pred = ridge.predict(X_train)
print('R2 score of Training Data:',r2_score(y_true=y_train,y_pred=y_train_pred) )
y_test_pred = ridge.predict(X_test)
print('R2 score of Testing Data:',r2_score(y_true=y_test,y_pred=y_test_pred) )
#To Print the Ridge Features & Coefficents initially:
model_parameter = list(ridge.coef_)
model_parameter.insert(0,ridge.intercept_)
cols = house_pred_train.columns
cols =  cols.insert(0,'constant')
ridge_coef = pd.DataFrame(list(zip(cols,model_parameter)))
ridge_coef.columns = ['Feature','coef']
ridge_coef.sort_values(by='coef',ascending=False).head(10)
X_train.shape
#No.of variables Picked by Ridge:
coef = pd.Series(ridge.coef_, index = X_train.columns)

print("Ridge picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
#Double the Alpha :
alpha = 2 * (model_cv.best_params_['alpha'])
ridge = Ridge(alpha=alpha)

#Fit Ridge model:
ridge.fit(X_train,y_train)

#predict:
y_train_pred = ridge.predict(X_train)
print('R2 score of Training Data:',r2_score(y_train,y_train_pred) )
y_test_pred = ridge.predict(X_test)
print('R2 score of Testing Data:',r2_score(y_test,y_test_pred) )
#To Print the Ridge Features & Coefficents after double the alpha:
model_parameter = list(ridge.coef_)
model_parameter.insert(0,ridge.intercept_)
cols = house_pred_train.columns
cols = cols.insert(0,'constant')
ridge_coef = pd.DataFrame(list(zip(cols,model_parameter)))
ridge_coef.columns = ['Feature','coef']
ridge_coef.sort_values(by='coef',ascending=False).head(10)
#Lasso Regression Regularization:
lasso = Lasso()

# cross validation
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
#fit the Lasso regularization model
model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()
# plotting mean test and train scoes with alpha 
cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='top right')
plt.show()
#checking the value of optimum number of parameters
print(model_cv.best_params_)
print(model_cv.best_score_)
#Assiging the Best alpha:
alpha = 0.001

#Fitting the Lasso Model:
lasso = Lasso(alpha=alpha)        
lasso.fit(X_train, y_train)
#predicting the r2 score for Train & Test set:
y_train_pred = lasso.predict(X_train)
print('R2 score of Training Data:',r2_score(y_train,y_train_pred) )
y_test_pred = lasso.predict(X_test)
print('R2 score of Testing Data:',r2_score(y_test,y_test_pred) )
#Lasso coefficients:
lasso.coef_
#Printing the Features & coefficents for the lasso model
model_param = list(lasso.coef_)
model_param.insert(0,lasso.intercept_)
cols = house_pred_train.columns
cols = cols.insert(0,'const')
lasso_coef= pd.DataFrame(list(zip(cols,model_param)))
lasso_coef.columns = ['Feature','coef']
lasso_coef.sort_values(by='coef',ascending=False).head(10)
X_train.shape

coef = pd.Series(lasso.coef_, index = X_train.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
#Double the Alpha :
alpha = 2 * 0.001
#Fit the Lasso Model:
lasso1 = Lasso(alpha=alpha)      
lasso1.fit(X_train, y_train)

#predict for double alpha:
y_train_pred = lasso1.predict(X_train)
print('R2 score of Training Data:',r2_score(y_train,y_train_pred) )
y_test_pred = lasso1.predict(X_test)
print('R2 score of Testing Data:',r2_score(y_test,y_test_pred) )
#Lasso coefficients for Double Alpha:
lasso1.coef_
#Lasso Features & coefficients for Double Alpha:
model_param = list(lasso1.coef_)
model_param.insert(0,lasso1.intercept_)
cols = house_pred_train.columns
cols = cols.insert(0,'const')
lasso1_coef= pd.DataFrame(list(zip(cols,model_param)))
lasso1_coef.columns = ['Feature','coef']
lasso1_coef.sort_values(by='coef',ascending=False).head(10)
#Dropping the top 5 features from the single alpha Lasso is:
var = ['OverallCond_3','Neighborhood_Crawfor','SaleType_New','Functional_Typ','Neighborhood_StoneBr']
X_train.drop(var,axis=1,inplace=True)
X_test.drop(var,axis=1,inplace=True )
lasso = Lasso()

# cross validation
model_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
#fit the lasso model
model_cv.fit(X_train, y_train) 
#fit the model for rest of top5:
alpha = 0.001
lasso2 = Lasso(alpha=alpha)
        
lasso2.fit(X_train, y_train)
#predicting r2 values for both Train & Test for rest of top5:
y_train_pred = lasso2.predict(X_train)
print('R2 score of Training Data:',r2_score(y_train,y_train_pred) )
y_test_pred = lasso2.predict(X_test)
print('R2 score of Testing Data:',r2_score(y_test,y_test_pred) )
#Coefficients for rest of top5:
lasso2.coef_
#Printing the features & coefficients for rest of top5 from single alpha lasso:
model_param = list(lasso2.coef_)
model_param.insert(0,lasso2.intercept_)
cols = X_train.columns
cols = cols.insert(0,'const')
lasso2_coef= pd.DataFrame(list(zip(cols,model_param)))
lasso2_coef.columns = ['Feature','coef']
lasso2_coef.sort_values(by='coef',ascending=False).head(10)