# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm, skew #for some statistics

data_file_path= '../input/house-prices-dataset/train.csv'
train_melbourne_data=pd.read_csv(data_file_path)

#descriptive statistics summary
train_melbourne_data['SalePrice'].describe()
#histogram
sns.distplot(train_melbourne_data['SalePrice'], fit=norm);
#skewness and kurtosis
print("Skewness: %f" % train_melbourne_data['SalePrice'].skew())
print("Kurtosis: %f" % train_melbourne_data['SalePrice'].kurt())
#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_melbourne_data['SalePrice'], plot=plt)
plt.show()
plt.figure(figsize=(15,8))
sns.boxplot(train_melbourne_data.YearBuilt, train_melbourne_data.SalePrice)
plt.figure(figsize=(12,6))
plt.scatter(x=train_melbourne_data.GrLivArea, y=train_melbourne_data.SalePrice)
plt.xlabel("GrLivArea", fontsize=13)
plt.ylabel("SalePrice", fontsize=13)
plt.ylim(0,800000)
# As is discussed in other kernels, the bottom right two two points with extremely large GrLivArea are likely to be outliers.
#So we delete them.
train_melbourne_data.drop(train_melbourne_data[(train_melbourne_data["GrLivArea"]>4000)
                                               &(train_melbourne_data["SalePrice"]<300000)].index,inplace=True)

train_melbourne_data.drop(['Id'],axis=1, inplace=True)
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train_melbourne_data["SalePrice"] = np.log1p(train_melbourne_data["SalePrice"])
#Check the new distribution 
sns.distplot(train_melbourne_data['SalePrice'] , fit=norm);

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_melbourne_data['SalePrice'], plot=plt)
plt.show()
#### Check NULL values in Train Data
all_data_missing = train_melbourne_data.isnull().sum()
all_data_missing = all_data_missing[all_data_missing>0].sort_values(ascending=False)

all_data_missing = pd.DataFrame({'Missing Numbers' :all_data_missing})
all_data_missing

temp = train_melbourne_data.copy()

all_data_missing = temp.isnull().sum() 
all_data_missing = all_data_missing.drop(all_data_missing[all_data_missing == 0].index).sort_values(ascending=False)
all_data_missing =  all_data_missing / len(temp)*100

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_missing.index, y=all_data_missing)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
# Print all the 'Numerical' columns
print("Numerical Columns -> {}".format(list(train_melbourne_data.select_dtypes(include=[np.number]).columns)))
# Print all the 'Categorical' columns
print("Categorical Columns -> {}".format(list(train_melbourne_data.select_dtypes(exclude=[np.number]).columns)))
# for numerical data
train_num_df = train_melbourne_data.select_dtypes(include=[np.number])

fig, axs = plt.subplots(12,3, figsize=(16, 30), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2, right=0.95)

axs = axs.ravel()

for ind, col in enumerate(train_num_df.columns):
    if col != 'SalePrice':
        sns.regplot(train_num_df[col], train_num_df['SalePrice'], ax = axs[ind])
    
plt.show()
# for Categorical data
pd.set_option('chained',None)

train_cat_df  = train_melbourne_data.select_dtypes(exclude=[np.number])
train_cat_df['SalePrice'] = train_melbourne_data['SalePrice']

fig, axs = plt.subplots(15,3, figsize=(16, 30), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.2, right=0.95)

axs = axs.ravel()

for ind, col in enumerate(train_cat_df.columns):
    if col != 'SalePrice':
        sns.barplot(train_cat_df[col], train_cat_df['SalePrice'], ax = axs[ind])

plt.show()
#### BsmtUnfSF
if (train_melbourne_data['BsmtUnfSF'].isnull().sum() != 0):
    train_melbourne_data['BsmtUnfSF'].fillna(round(train_melbourne_data['BsmtUnfSF'].mean()), inplace = True)
    
##### TotalBsmtSF
if (train_melbourne_data['TotalBsmtSF'].isnull().sum() != 0):
    train_melbourne_data['TotalBsmtSF'].fillna(round(train_melbourne_data['TotalBsmtSF'].mean()), inplace = True)
    
#### BsmtFinSF1
if (train_melbourne_data['BsmtFinSF1'].isnull().sum() != 0):
    train_melbourne_data['BsmtFinSF1'].fillna(round(train_melbourne_data['BsmtFinSF1'].mean()), inplace = True) 
    
#### BsmtFinSF2
if (train_melbourne_data['BsmtFinSF2'].isnull().sum() != 0):
    train_melbourne_data['BsmtFinSF2'].fillna(round(train_melbourne_data['BsmtFinSF2'].mean()), inplace = True)
#### BsmtFullBath
if (train_melbourne_data['BsmtFullBath'].isnull().sum() != 0):
    train_melbourne_data['BsmtFullBath'].fillna(train_melbourne_data['BsmtFullBath'].mode().iloc[0], inplace = True)
    
 #### BsmtHalfBath
if (train_melbourne_data['BsmtHalfBath'].isnull().sum() != 0):
    train_melbourne_data['BsmtHalfBath'].fillna(train_melbourne_data['BsmtHalfBath'].mode().iloc[0], inplace = True)
    
    
#### GarageCars
if (train_melbourne_data['GarageCars'].isnull().sum() != 0):
    train_melbourne_data['GarageCars'].fillna(train_melbourne_data['GarageCars'].mode().iloc[0], inplace = True)
    
#### KitchenQual ......................................
if (train_melbourne_data['KitchenQual'].isnull().sum() != 0):
    train_melbourne_data['KitchenQual'].fillna(train_melbourne_data['KitchenQual'].mode().iloc[0], inplace = True)
    
#### Electrical .........................................
if (train_melbourne_data['Electrical'].isnull().sum() != 0):
    train_melbourne_data['Electrical'].fillna(train_melbourne_data['Electrical'].mode().iloc[0], inplace = True)
    
    
#### SaleType.....................................................
if (train_melbourne_data['SaleType'].isnull().sum() != 0):
    train_melbourne_data['SaleType'].fillna(train_melbourne_data['SaleType'].mode().iloc[0], inplace = True)
    
#### Exterior1st........................................
if (train_melbourne_data['Exterior1st'].isnull().sum() != 0):
    train_melbourne_data['Exterior1st'].fillna(train_melbourne_data['Exterior1st'].mode().iloc[0], inplace = True)
    
#### Exterior2nd....................................................
if (train_melbourne_data['Exterior2nd'].isnull().sum() != 0):
    train_melbourne_data['Exterior2nd'].fillna(train_melbourne_data['Exterior2nd'].mode().iloc[0], inplace = True)
    
#### Functional................................................................
if (train_melbourne_data['Functional'].isnull().sum() != 0):
    train_melbourne_data['Functional'].fillna(train_melbourne_data['Functional'].mode().iloc[0], inplace = True)
    
#### Utilities........................................................
if (train_melbourne_data['Utilities'].isnull().sum() != 0):
    train_melbourne_data['Utilities'].fillna(train_melbourne_data['Utilities'].mode().iloc[0], inplace = True)
    
#### MSZoning.......................................................
if (train_melbourne_data['MSZoning'].isnull().sum() != 0):
    train_melbourne_data['MSZoning'].fillna(train_melbourne_data['MSZoning'].mode().iloc[0], inplace = True)
    
    
#### BsmtCond.......................................................
if (train_melbourne_data['BsmtCond'].isnull().sum() != 0):
    train_melbourne_data['BsmtCond'].fillna(train_melbourne_data['BsmtCond'].mode().iloc[0], inplace = True)
    
#### BsmtExposure............................
if (train_melbourne_data['BsmtExposure'].isnull().sum() != 0):
    train_melbourne_data['BsmtExposure'].fillna(train_melbourne_data['BsmtExposure'].mode().iloc[0], inplace = True)
    
#### BsmtQual........................................
if (train_melbourne_data['BsmtQual'].isnull().sum() != 0):
    train_melbourne_data['BsmtQual'].fillna(train_melbourne_data['BsmtQual'].mode().iloc[0], inplace = True)
    
#### BsmtFinType1...........................
if (train_melbourne_data['BsmtFinType1'].isnull().sum() != 0):
    train_melbourne_data['BsmtFinType1'].fillna(train_melbourne_data['BsmtFinType1'].mode().iloc[0], inplace = True)
    
#### BsmtFinType2..................................................
if (train_melbourne_data['BsmtFinType2'].isnull().sum() != 0):
    train_melbourne_data['BsmtFinType2'].fillna(train_melbourne_data['BsmtFinType2'].mode().iloc[0], inplace = True)
    
#### GarageType.............................................
if (train_melbourne_data['GarageType'].isnull().sum() != 0):
    train_melbourne_data['GarageType'].fillna(train_melbourne_data['GarageType'].mode().iloc[0], inplace = True)
    
#### GarageCond........................................
if (train_melbourne_data['GarageCond'].isnull().sum() != 0):
    train_melbourne_data['GarageCond'].fillna(train_melbourne_data['GarageCond'].mode().iloc[0], inplace = True)
    
#### GarageQual..........................................
if (train_melbourne_data['GarageQual'].isnull().sum() != 0):
    train_melbourne_data['GarageQual'].fillna(train_melbourne_data['GarageQual'].mode().iloc[0], inplace = True)
    
#### GarageFinish....................................
if (train_melbourne_data['GarageFinish'].isnull().sum() != 0):
    train_melbourne_data['GarageFinish'].fillna(train_melbourne_data['GarageFinish'].mode().iloc[0], inplace = True)
#### MasVnrArea
if (train_melbourne_data['MasVnrArea'].isnull().sum() != 0):
    train_melbourne_data['MasVnrArea'].fillna(0, inplace = True)
    
#### MasVnrType............................
if (train_melbourne_data['MasVnrType'].isnull().sum() != 0):
    train_melbourne_data['MasVnrType'].fillna('None', inplace = True)
    
    #### GarageYrBlt
if (train_melbourne_data['GarageYrBlt'].isnull().sum() != 0):
    train_melbourne_data['GarageYrBlt'].fillna(train_melbourne_data['YearBuilt'], inplace = True)
#Categorical

#### FireplaceQu
if (train_melbourne_data['FireplaceQu'].isnull().sum().sum() != 0):
    train_melbourne_data['FireplaceQu'] = train_melbourne_data['FireplaceQu'].fillna('NoFirePlace')
    
#### Fence
if (train_melbourne_data['Fence'].isnull().sum().sum() != 0):
    train_melbourne_data['Fence'] = train_melbourne_data['Fence'].fillna('NoFence')
    
#### Alley
if (train_melbourne_data['Alley'].isnull().sum().sum() != 0):
    train_melbourne_data['Alley'] = train_melbourne_data['Alley'].fillna('NoAlley')
    
#### MiscFeature
if (train_melbourne_data['MiscFeature'].isnull().sum().sum() != 0):
    train_melbourne_data['MiscFeature'] = train_melbourne_data['MiscFeature'].fillna('NoMiscFeature')
    
    
#### PoolQC
if (train_melbourne_data['PoolQC'].isnull().sum().sum() != 0):
    train_melbourne_data['PoolQC'] = train_melbourne_data['PoolQC'].fillna('NoPoolQC')
print("Total Features with NaN in Test After Imputation = " + str(train_melbourne_data.columns[train_melbourne_data.isnull().sum() != 0].size))
# Use a heatmap to see which features have strongest correlation with house price
#correlation matrix

correlation_matrix = train_melbourne_data.corr()
f, ax = plt.subplots(figsize=(18, 20))
sns.heatmap(correlation_matrix, vmax=.8,cbar=True, annot=True,  square=True);
correlation_matrix.sort_values(["SalePrice"], ascending = False, inplace = True)
print(correlation_matrix.SalePrice.head(30))
train_y = train_melbourne_data.SalePrice

train_melbourne_neumeric_data =  train_melbourne_data._get_numeric_data()
print(sorted(train_melbourne_neumeric_data.columns))
display(train_melbourne_neumeric_data.head(5))
# Based on the heatmap and the neumeric values, we choose the following parameters
predictor_parameters = [ 'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF',
                        'YearBuilt', 'YearRemodAdd', 'GarageYrBlt',  'TotRmsAbvGrd', 'Fireplaces',]

#    'YearBuilt',  'YearRemodAdd'
train_X = train_melbourne_data[predictor_parameters]

#imp = Imputer(missing_values='NaN', axis=0)
#train_X = imp.fit_transform(train_X)  

model = RandomForestRegressor()
model.fit(train_X, train_y)
test_data_file_path= '../input/house-prices-dataset/test.csv'
test_melbourne_data=pd.read_csv(test_data_file_path)

# test data in the same way as training data. -- pull same columns.
test_X = test_melbourne_data[predictor_parameters]
test_X = imp.fit_transform(test_X) 

# Use the model to make predictions
predicted_prices = model.predict(test_X)
print(predicted_prices)
my_submission = pd.DataFrame({'Id': test_melbourne_data.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)