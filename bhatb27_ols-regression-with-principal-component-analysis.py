# Importing all the packages:

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Scaling data

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler



# PCA Model

from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA



# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Data import

dtrain = pd.read_csv('../input/train.csv')

dtest = pd.read_csv('../input/test.csv')
dtrain.info()
dtrain.shape
dtest.shape
dtrain.head()
dtest.head()
dtrain.describe()
dtest['SalePrice'] = 0.0
# Merging both data set for cleaning

dall = dtrain.append(dtest,ignore_index=True)
dall.shape
dall.columns
dall1 = dall.copy()
dall1.loc[dall1['MSZoning'].isnull(),'MSZoning'] = dall1['MSZoning'].mode()[0]
dall1.loc[dall1['LotFrontage'].isnull(),'LotFrontage'] = np.sqrt(dall1.loc[dall1['LotFrontage'].isnull(),'LotArea'])
dall1.drop('Alley',axis=1,inplace=True)
dall1.drop('Utilities',axis=1,inplace=True)
dall1.loc[dall1['Exterior1st'].isnull(),'Exterior1st'] = dall1['Exterior1st'].mode()[0]
dall1.loc[dall1['Exterior2nd'].isnull(),'Exterior2nd'] = dall1['Exterior2nd'].mode()[0]
dall1.loc[dall1['MasVnrType'].isnull(),'MasVnrType'] = dall1['MasVnrType'].mode()[0]
dall1.loc[dall1['MasVnrArea'].isnull(),'MasVnrArea'] = 0.0
dall1.loc[dall1['TotalBsmtSF'].isnull(),'TotalBsmtSF'] = 0.0
dall1.loc[(dall1['BsmtQual'].isnull() & (dall1['TotalBsmtSF']==0)),'BsmtQual'] = 'NA'

dall1.loc[dall1['BsmtQual'].isnull(),'BsmtQual'] = dall1['BsmtQual'].mode()[0]
dall1.loc[(dall1['BsmtCond'].isnull() & (dall1['TotalBsmtSF']==0)),'BsmtCond'] = 'NA'

dall1.loc[dall1['BsmtCond'].isnull(),'BsmtCond'] = dall1['BsmtCond'].mode()[0]
dall1.loc[(dall1['BsmtExposure'].isnull() & (dall1['TotalBsmtSF']==0)),'BsmtExposure'] = 'NA'

dall1.loc[dall1['BsmtExposure'].isnull(),'BsmtExposure'] = dall1['BsmtExposure'].mode()[0]
dall1.loc[(dall1['BsmtFinType1'].isnull() & (dall1['TotalBsmtSF']==0)),'BsmtFinType1'] = 'NA'

dall1.loc[dall1['BsmtFinType1'].isnull(),'BsmtFinType1'] = dall1['BsmtFinType1'].mode()[0]
dall1.loc[(dall1['BsmtFinType2'].isnull() & (dall1['TotalBsmtSF']==0)),'BsmtFinType2'] = 'NA'

dall1.loc[dall1['BsmtFinType2'].isnull(),'BsmtFinType2'] = dall1['BsmtFinType2'].mode()[0]
dall1.loc[dall1['BsmtFinSF1'].isnull(),'BsmtFinSF1'] = 0.0

dall1.loc[dall1['BsmtFinSF2'].isnull(),'BsmtFinSF2'] = 0.0

dall1.loc[dall1['BsmtUnfSF'].isnull(),'BsmtUnfSF'] = 0.0
dall1.loc[dall1['Electrical'].isnull(),'Electrical'] = dall1['Electrical'].mode()[0]
dall1.loc[dall1['BsmtFullBath'].isnull(),'BsmtFullBath'] = 0.0
dall1.loc[dall1['BsmtHalfBath'].isnull(),'BsmtHalfBath'] = 0.0
dall1.loc[dall1['KitchenQual'].isnull(),'KitchenQual'] = dall1['KitchenQual'].mode()[0]
dall1.loc[dall1['Functional'].isnull(),'Functional'] = dall1['Functional'].mode()[0]
dall1.loc[dall1['FireplaceQu'].isnull(),'FireplaceQu'] = 'NA'
dall1.loc[dall1['GarageArea'].isnull(),'GarageArea'] = 0.0
dall1.loc[(dall1['GarageType'].isnull() & (dall1['GarageArea']==0)),'GarageType'] = 'NA'

dall1.loc[dall1['GarageType'].isnull(),'GarageType'] = dall1['GarageType'].mode()[0]
dall1.loc[(dall1['GarageFinish'].isnull() & (dall1['GarageArea']==0)),'GarageFinish'] = 'NA'

dall1.loc[dall1['GarageFinish'].isnull(),'GarageFinish'] = dall1['GarageFinish'].mode()[0]
dall1.loc[(dall1['GarageQual'].isnull() & (dall1['GarageArea']==0)),'GarageQual'] = 'NA'

dall1.loc[dall1['GarageQual'].isnull(),'GarageQual'] = dall1['GarageQual'].mode()[0]
dall1.loc[(dall1['GarageCond'].isnull() & (dall1['GarageArea']==0)),'GarageCond'] = 'NA'

dall1.loc[dall1['GarageCond'].isnull(),'GarageCond'] = dall1['GarageCond'].mode()[0]
dall1.loc[dall1['GarageCars'].isnull(),'GarageCars'] = 0.0
dall1.drop('PoolQC',axis=1,inplace=True)
dall1.loc[dall1['Fence'].isnull(),'Fence'] = 'NA'
dall1.loc[dall1['MiscFeature'].isnull(),'MiscFeature'] = 'NA'
dall1.loc[dall1['SaleType'].isnull(),'SaleType'] = dall1['SaleType'].mode()[0]
dall1.loc[dall1['GarageYrBlt']>2010,'GarageYrBlt'] = 2007
dall1.loc[dall1['GarageYrBlt'].isnull(),'GarageYrBlt'] = 2020
blanks = list(dall1.columns[dall1.isna().sum()>0])

dall1[blanks].isnull().sum()*100/len(dall1.index)
dall2 = dall1.copy()
dall2.columns
col_num = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea']
plt.figure(figsize=[20,15])

i=1

for x in col_num:

    plt.subplot(3,3,i)

    plt.boxplot(dall2[x])

    i=i+1

plt.show()
# For LotFrontage

dall2.loc[dall2['LotFrontage'] > dall2['LotFrontage'].quantile(0.75) + (dall2['LotFrontage'].quantile(0.75) - dall2['LotFrontage'].quantile(0.25))*1.5,'LotFrontage'] = dall2['LotFrontage'].quantile(0.75) + (dall2['LotFrontage'].quantile(0.75) - dall2['LotFrontage'].quantile(0.25))*1.5

#dall2.loc[dall2['PC3'] < dall2['PC3'].quantile(0.25) - (dall2['PC3'].quantile(0.75) - dall2['PC3'].quantile(0.25))*1.5,'PC3'] = dall2['PC3'].quantile(0.25) - (dall2['PC3'].quantile(0.75) - dall2['PC3'].quantile(0.25))*1.5
# For LotArea

dall2.loc[dall2['LotArea'] > dall2['LotArea'].quantile(0.75) + (dall2['LotArea'].quantile(0.75) - dall2['LotArea'].quantile(0.25))*1.5,'LotArea'] = dall2['LotArea'].quantile(0.75) + (dall2['LotArea'].quantile(0.75) - dall2['LotArea'].quantile(0.25))*1.5

dall2.loc[dall2['LotArea'] < dall2['LotArea'].quantile(0.25) - (dall2['LotArea'].quantile(0.75) - dall2['LotArea'].quantile(0.25))*1.5,'LotArea'] = dall2['LotArea'].quantile(0.25) - (dall2['LotArea'].quantile(0.75) - dall2['LotArea'].quantile(0.25))*1.5
# For MasVnrArea

dall2.loc[dall2['MasVnrArea'] > dall2['MasVnrArea'].quantile(0.75) + (dall2['MasVnrArea'].quantile(0.75) - dall2['MasVnrArea'].quantile(0.25))*1.5,'MasVnrArea'] = dall2['MasVnrArea'].quantile(0.75) + (dall2['MasVnrArea'].quantile(0.75) - dall2['MasVnrArea'].quantile(0.25))*1.5

dall2.loc[dall2['MasVnrArea'] < dall2['MasVnrArea'].quantile(0.25) - (dall2['MasVnrArea'].quantile(0.75) - dall2['MasVnrArea'].quantile(0.25))*1.5,'MasVnrArea'] = dall2['MasVnrArea'].quantile(0.25) - (dall2['MasVnrArea'].quantile(0.75) - dall2['MasVnrArea'].quantile(0.25))*1.5
# For BsmtFinSF1

dall2.loc[dall2['BsmtFinSF1'] > dall2['BsmtFinSF1'].quantile(0.75) + (dall2['BsmtFinSF1'].quantile(0.75) - dall2['BsmtFinSF1'].quantile(0.25))*1.5,'BsmtFinSF1'] = dall2['BsmtFinSF1'].quantile(0.75) + (dall2['BsmtFinSF1'].quantile(0.75) - dall2['BsmtFinSF1'].quantile(0.25))*1.5

dall2.loc[dall2['BsmtFinSF1'] < dall2['BsmtFinSF1'].quantile(0.25) - (dall2['BsmtFinSF1'].quantile(0.75) - dall2['BsmtFinSF1'].quantile(0.25))*1.5,'BsmtFinSF1'] = dall2['BsmtFinSF1'].quantile(0.25) - (dall2['BsmtFinSF1'].quantile(0.75) - dall2['BsmtFinSF1'].quantile(0.25))*1.5
# For TotalBsmtSF

dall2.loc[dall2['TotalBsmtSF'] > dall2['TotalBsmtSF'].quantile(0.75) + (dall2['TotalBsmtSF'].quantile(0.75) - dall2['TotalBsmtSF'].quantile(0.25))*1.5,'TotalBsmtSF'] = dall2['TotalBsmtSF'].quantile(0.75) + (dall2['TotalBsmtSF'].quantile(0.75) - dall2['TotalBsmtSF'].quantile(0.25))*1.5

dall2.loc[dall2['TotalBsmtSF'] < dall2['TotalBsmtSF'].quantile(0.25) - (dall2['TotalBsmtSF'].quantile(0.75) - dall2['TotalBsmtSF'].quantile(0.25))*1.5,'TotalBsmtSF'] = dall2['TotalBsmtSF'].quantile(0.25) - (dall2['TotalBsmtSF'].quantile(0.75) - dall2['TotalBsmtSF'].quantile(0.25))*1.5
# For 1stFlrSF

dall2.loc[dall2['1stFlrSF'] > dall2['1stFlrSF'].quantile(0.75) + (dall2['1stFlrSF'].quantile(0.75) - dall2['1stFlrSF'].quantile(0.25))*1.5,'1stFlrSF'] = dall2['1stFlrSF'].quantile(0.75) + (dall2['1stFlrSF'].quantile(0.75) - dall2['1stFlrSF'].quantile(0.25))*1.5

dall2.loc[dall2['1stFlrSF'] < dall2['1stFlrSF'].quantile(0.25) - (dall2['1stFlrSF'].quantile(0.75) - dall2['1stFlrSF'].quantile(0.25))*1.5,'1stFlrSF'] = dall2['1stFlrSF'].quantile(0.25) - (dall2['1stFlrSF'].quantile(0.75) - dall2['1stFlrSF'].quantile(0.25))*1.5
# For 2ndFlrSF

dall2.loc[dall2['2ndFlrSF'] > dall2['2ndFlrSF'].quantile(0.75) + (dall2['2ndFlrSF'].quantile(0.75) - dall2['2ndFlrSF'].quantile(0.25))*1.5,'2ndFlrSF'] = dall2['2ndFlrSF'].quantile(0.75) + (dall2['2ndFlrSF'].quantile(0.75) - dall2['2ndFlrSF'].quantile(0.25))*1.5

dall2.loc[dall2['2ndFlrSF'] < dall2['2ndFlrSF'].quantile(0.25) - (dall2['2ndFlrSF'].quantile(0.75) - dall2['2ndFlrSF'].quantile(0.25))*1.5,'2ndFlrSF'] = dall2['2ndFlrSF'].quantile(0.25) - (dall2['2ndFlrSF'].quantile(0.75) - dall2['2ndFlrSF'].quantile(0.25))*1.5
# For GrLivArea

dall2.loc[dall2['GrLivArea'] > dall2['GrLivArea'].quantile(0.75) + (dall2['GrLivArea'].quantile(0.75) - dall2['GrLivArea'].quantile(0.25))*1.5,'GrLivArea'] = dall2['GrLivArea'].quantile(0.75) + (dall2['GrLivArea'].quantile(0.75) - dall2['GrLivArea'].quantile(0.25))*1.5

dall2.loc[dall2['GrLivArea'] < dall2['GrLivArea'].quantile(0.25) - (dall2['GrLivArea'].quantile(0.75) - dall2['GrLivArea'].quantile(0.25))*1.5,'GrLivArea'] = dall2['GrLivArea'].quantile(0.25) - (dall2['GrLivArea'].quantile(0.75) - dall2['GrLivArea'].quantile(0.25))*1.5
col_num = ['PoolArea','ScreenPorch','GarageArea','OpenPorchSF','WoodDeckSF','EnclosedPorch','3SsnPorch','BsmtFullBath','BsmtHalfBath']



plt.figure(figsize=[20,15])

i=1

for x in col_num:

    plt.subplot(3,3,i)

    plt.boxplot(dall2[x])

    i=i+1

plt.show()
# For GarageArea

dall2.loc[dall2['GarageArea'] > dall2['GarageArea'].quantile(0.75) + (dall2['GarageArea'].quantile(0.75) - dall2['GarageArea'].quantile(0.25))*1.5,'GarageArea'] = dall2['GarageArea'].quantile(0.75) + (dall2['GarageArea'].quantile(0.75) - dall2['GarageArea'].quantile(0.25))*1.5

dall2.loc[dall2['GarageArea'] < dall2['GarageArea'].quantile(0.25) - (dall2['GarageArea'].quantile(0.75) - dall2['GarageArea'].quantile(0.25))*1.5,'GarageArea'] = dall2['GarageArea'].quantile(0.25) - (dall2['GarageArea'].quantile(0.75) - dall2['GarageArea'].quantile(0.25))*1.5
# For OpenPorchSF

dall2.loc[dall2['OpenPorchSF'] > dall2['OpenPorchSF'].quantile(0.75) + (dall2['OpenPorchSF'].quantile(0.75) - dall2['OpenPorchSF'].quantile(0.25))*1.5,'OpenPorchSF'] = dall2['OpenPorchSF'].quantile(0.75) + (dall2['OpenPorchSF'].quantile(0.75) - dall2['OpenPorchSF'].quantile(0.25))*1.5

dall2.loc[dall2['OpenPorchSF'] < dall2['OpenPorchSF'].quantile(0.25) - (dall2['OpenPorchSF'].quantile(0.75) - dall2['OpenPorchSF'].quantile(0.25))*1.5,'OpenPorchSF'] = dall2['OpenPorchSF'].quantile(0.25) - (dall2['OpenPorchSF'].quantile(0.75) - dall2['OpenPorchSF'].quantile(0.25))*1.5
# For WoodDeckSF

dall2.loc[dall2['WoodDeckSF'] > dall2['WoodDeckSF'].quantile(0.75) + (dall2['WoodDeckSF'].quantile(0.75) - dall2['WoodDeckSF'].quantile(0.25))*1.5,'WoodDeckSF'] = dall2['WoodDeckSF'].quantile(0.75) + (dall2['WoodDeckSF'].quantile(0.75) - dall2['WoodDeckSF'].quantile(0.25))*1.5

dall2.loc[dall2['WoodDeckSF'] < dall2['WoodDeckSF'].quantile(0.25) - (dall2['WoodDeckSF'].quantile(0.75) - dall2['WoodDeckSF'].quantile(0.25))*1.5,'WoodDeckSF'] = dall2['WoodDeckSF'].quantile(0.25) - (dall2['WoodDeckSF'].quantile(0.75) - dall2['WoodDeckSF'].quantile(0.25))*1.5
# For BsmtFullBath

dall2.loc[dall2['BsmtFullBath'] > dall2['BsmtFullBath'].quantile(0.75) + (dall2['BsmtFullBath'].quantile(0.75) - dall2['BsmtFullBath'].quantile(0.25))*1.5,'BsmtFullBath'] = dall2['BsmtFullBath'].quantile(0.75) + (dall2['BsmtFullBath'].quantile(0.75) - dall2['BsmtFullBath'].quantile(0.25))*1.5

dall2.loc[dall2['BsmtFullBath'] < dall2['BsmtFullBath'].quantile(0.25) - (dall2['BsmtFullBath'].quantile(0.75) - dall2['BsmtFullBath'].quantile(0.25))*1.5,'BsmtFullBath'] = dall2['BsmtFullBath'].quantile(0.25) - (dall2['BsmtFullBath'].quantile(0.75) - dall2['BsmtFullBath'].quantile(0.25))*1.5
dall3 = dall2.copy()
dall3.loc[:,dall3.dtypes=='object'].columns
col_list = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig','LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType','HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd','MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating','HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional','FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','PavedDrive', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']



for x in col_list:

    h = pd.get_dummies(dall3[x],prefix=x,drop_first=True)

    dall3 = pd.concat([dall3,h],axis=1)
dall3.shape
# Dropping main object variable

dall3.drop(col_list,axis=1,inplace=True)
dall3.shape
dall3.loc[:,dall3.dtypes=='object'].columns
# Converting Id as Object

dall3['Id'] = dall3['Id'].astype('object')
dall3.info()
dall4 = dall3.copy()
dall4.drop('Id',axis=1,inplace=True)
scaler = StandardScaler()
col_list = list(dall4.columns)

col_list.remove('SalePrice')
dall4[col_list] = scaler.fit_transform(dall4[col_list])
dall5 = dall4[col_list]
# Applying PCA on data

pca = PCA(svd_solver='randomized', random_state=15)

pca.fit(dall5)
#SCREE plot

fig = plt.figure(figsize = (12,8))

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.show()
# Applying PCA for 150 PCs

pca_final = IncrementalPCA(n_components=150)

dall_pca = pca_final.fit_transform(dall5)
# Coorelation between new 150 PCs

plt.figure(figsize = (15,10))

sns.heatmap(np.corrcoef(dall_pca.transpose()),annot = False)
# Creating Column names for PCs

col_list = list()

for x in range(1,151,1):

    col_list.append('PC'+str(x))
dall_pca_new = pd.DataFrame(data=dall_pca, columns = col_list)
dall_pca_new = pd.concat([dall_pca_new,dall3['SalePrice']],axis=1)
dtest1 = dall_pca_new.loc[dall_pca_new['SalePrice']==0]

dtrain1 = dall_pca_new.loc[~(dall_pca_new['SalePrice']==0)]
dtrain1.shape
dtest1.shape
dtrainX = dtrain1.drop('SalePrice',axis=1)

dtrainy = dtrain1['SalePrice']

dtestX = dtest1.drop('SalePrice',axis=1)
lm = LinearRegression()

lm.fit(dtrainX,dtrainy)



rfe = RFE(lm,62)

rfe = rfe.fit(dtrainX,dtrainy)
col_list = dtrainX.columns[rfe.support_]

dtrainX = dtrainX[col_list]
# Adding a constant variable 

import statsmodels.api as sm  

dtrainX = sm.add_constant(dtrainX)

lm1 = sm.OLS(dtrainy,dtrainX).fit()

print(lm1.summary())
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = dtrainX.drop('const',axis=1)

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
#Predicted value for train dataset

y_pred = lm1.predict(dtrainX)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((dtrainy - y_pred), bins = 20)

fig.suptitle('Residual Hist', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 10) 
# Plot the scatter of the error terms/price

fig = plt.figure()

plt.scatter(dtrainy,dtrainy-y_pred)

fig.suptitle('Error Vs Price', fontsize = 20)                  # Plot heading 

plt.ylabel('Errors', fontsize = 10)

plt.xlabel('Price', fontsize = 10) 
# Plot the scatter of the error terms/price

fig = plt.figure()

plt.scatter(dtrainy,y_pred)

fig.suptitle('Correlation b/w actual and predicted price', fontsize = 20)                  # Plot heading 

plt.ylabel('Predicted price', fontsize = 10)

plt.xlabel('Price', fontsize = 10) 
# Selecting same columns as Train

dtestX = dtestX[col_list]
# Adding a constant variable 

import statsmodels.api as sm  

dtestX = sm.add_constant(dtestX)
ypred = lm1.predict(dtestX)
ypred = pd.DataFrame(ypred,columns=['SalePrice'])
ypred = ypred.set_index(dtest['Id'])

ypred.to_csv('Submit.csv')