#Let us begin with importing with all the necessary library that we'll be using



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #used for data visulization

import seaborn as sns  #used for data visulization



import warnings  #This will ignore all the unnecessary warnings                

warnings.simplefilter("ignore")



import os  #Display the list of files present in the directory.         

print(os.listdir("../input"))



#It is used display the visulization into this notebook

%matplotlib inline 
#Loading Data

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

train.head()
#We can see that our target column (SalePrice) has a mean value of 180921.

#We can also note the minimum and maximum value of that column

train.describe()
#Let us now create a histogram of our Target column.

sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

sns.distplot(train['SalePrice'],bins=30,kde=True)
#Let us have a look at all the other columns in our dataset 

train.info()
#Lets check whether we have any duplicate data

sum(train['Id'].duplicated()),sum(test['Id'].duplicated())
# Drop Id column from train dataset

train.drop(columns=['Id'],inplace=True)
#categorize the data:



num_cols=[var for var in train.columns if train[var].dtypes != 'O']

cat_cols=[var for var in train.columns if train[var].dtypes != 'int64' and train[var].dtypes != 'float64']



print('No of Numerical Columns: ',len(num_cols))

print('No of Categorical Columns: ',len(cat_cols))

print('Total No of Cols: ',len(num_cols+cat_cols))
#Lets create a heatmap to see which all columns has null values

plt.figure(figsize=(30,12))

sns.heatmap(train.isnull(), yticklabels=False, cmap='viridis',cbar='cyan')
#Columns with null values in the Train dataFrame

var_with_na=[var for var in train.columns if train[var].isnull().sum()>=1 ]



for var in var_with_na:

    print(var, np.round(train[var].isnull().mean(),3), '% missing values')
#Columns with null values in the Test dataFrame

var_with_na2=[var for var in test.columns if test[var].isnull().sum()>=1 ]



for var in var_with_na2:

    print(var, np.round(test[var].isnull().mean(),3), '% missing values')
#I have categorized the missing columns based on the datatypes and no of null values.



missing_cols=['MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

           'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond']

num_list=['LotFrontage','MasVnrArea','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2',

           'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']

drop_cols=['PoolQC','Fence','MiscFeature','GarageYrBlt','Alley']
combine=[train,test]



for df in combine:

    # Fill missing values in cetgorical variables with None

    for col in missing_cols:

        df[col]=df[col].fillna('None')

    # Fill missing values in numerical variables with 0

    for col in num_list:

        df[col]=df[col].fillna(0)

    # Drop columns with large number of missing values

    df.drop(columns=drop_cols,inplace=True) 
plt.figure(figsize=(30,12))

sns.heatmap(train.isnull(), yticklabels=False, cmap='viridis',cbar='cyan')  
#Columns with null values in the Train dataFrame

var_with_na=[var for var in train.columns if train[var].isnull().sum()>=1 ]



for var in var_with_na:

    print(var, np.round(train[var].isnull().mean(),3), '% missing values')
#Columns with null values in the Test dataFrame

var_with_na2=[var for var in test.columns if test[var].isnull().sum()>=1 ]



for var in var_with_na2:

    print(var, np.round(test[var].isnull().mean(),3), '% missing values')
train['Electrical'].fillna('None', inplace=True)
# Fill missing values in categorical variables in test dataset

mode_list=['Utilities','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType','MSZoning']

for col in mode_list:

    mode=test[col].mode()

    test[col]=test[col].fillna(mode[0])
train.isnull().sum().max(),test.isnull().sum().max()
plt.figure(figsize=(30,12))

sns.heatmap(train.isnull(), yticklabels=False, cmap='viridis',cbar='cyan') 
plt.figure(figsize=(20,10))

sns.heatmap(train.corr(), vmax=.8, square=True,cbar=True,cmap='RdGy');
# Plot the correlation matrix

imp_list=['SalePrice','OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea' ,'GarageArea',

         'GarageCars','LotArea','PoolArea','Fireplaces','1stFlrSF','FullBath']

corrmat = train[imp_list].corr()

plt.subplots(figsize=(10, 9))

sns.heatmap(corrmat, vmax=.8, square=True,cbar=True, annot=True, fmt='.2f', annot_kws={'size': 10});
#Lets plot a boxplot on Overall Quality Vs The salesprice

plt.figure(figsize=(10,8))

sns.boxplot(x='OverallQual',y='SalePrice',data=train)
#Lets plot a barplot on Built Year Vs The salesprice

plt.figure(figsize=(30,8))

sns.barplot(x=train['YearBuilt'],y=train['SalePrice'],palette='viridis_r')
#Lets plot a scatterplot on Total Basement Area Vs salesprice

plt.figure(figsize=(10,8))

sns.scatterplot(x=train['TotalBsmtSF'],y=train['SalePrice'], alpha=0.8)
index=train[train['TotalBsmtSF']>4000].index

train.drop(index,inplace=True)
#Lets plot a scatterplot on Total Basement Area Vs salesprice

plt.figure(figsize=(10,8))

sns.scatterplot(x=train['TotalBsmtSF'],y=train['SalePrice'], alpha=0.8)
#Lets plot a Jointplot on Carpet Area Vs The salesprice

plt.figure(figsize=(10,8))

sns.scatterplot(x=train['GrLivArea'],y=train['SalePrice'])
# Remove outliers from Above Ground Area

index=train[train['GrLivArea']>4000].index

train.drop(index,inplace=True)
#Lets plot a Jointplot on Carpet Area Vs The salesprice

plt.figure(figsize=(10,8))

sns.scatterplot(x=train['GrLivArea'],y=train['SalePrice'])
#Lets plot a boxplot on Garage Capacity Vs The salesprice

plt.figure(figsize=(10,8))

sns.boxplot(x=train['GarageCars'],y=train['SalePrice'])
# Removing outliers manually (More than 4-cars, less than $300k)

train = train.drop(train[(train['GarageCars']>3) & (train['SalePrice']<300000)].index).reset_index(drop=True)
plt.figure(figsize=(10,8))

sns.boxplot(x=train['GarageCars'],y=train['SalePrice'])
#Lets plot a line plot on Lot Area vs Salesprice

plt.figure(figsize=(10,6))

sns.lineplot(x=train['LotArea'],y=train['SalePrice'],palette='viridis_r')
# Bivariate plot of Quality vs. Area

plt.figure(figsize=(10,10))

plt.subplot(3,1,1)

sns.boxplot(data=train,y='TotalBsmtSF',x='OverallQual');

plt.subplot(3,1,2)

sns.boxplot(data=train,y='GarageArea',x='OverallQual');

plt.subplot(3,1,3)

sns.boxplot(data=train,y='GrLivArea',x='OverallQual');
#Lets plot a scatterplot on Firstfloor sqFt  Vs The salesprice

plt.figure(figsize=(10,8))

sns.jointplot(x=train['1stFlrSF'],y=train['SalePrice'],kind='hex')
# Drop object features

for df in combine:

    df.drop(columns=['MSZoning','SaleCondition','SaleType','PavedDrive','GarageCond','GarageQual','GarageFinish',

                    'GarageType','FireplaceQu','Functional','KitchenQual','Heating','HeatingQC','CentralAir',

                     'Electrical','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure',

                     'BsmtFinType1','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Street',

                     'LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1',

                     'Condition2','BldgType','HouseStyle','BsmtFinType2' ],inplace=True)
# Merge the two datasets

ntrain = train.shape[0]

ntest = test.shape[0]

all_data = pd.concat((train, test))
# Get dummy variables

all_data=pd.get_dummies(all_data)
# Seperate the combined dataset into test and train data

test=all_data[all_data['SalePrice'].isnull()]

train=all_data[all_data['Id'].isnull()]
# Check if the new and old sizes are equal

assert train.shape[0]==ntrain

assert test.shape[0]==ntest
# Drop extra columns

test.drop(columns='SalePrice',inplace=True)

train.drop(columns='Id',inplace=True)

test['Id']=test['Id'].astype(int)
X_train=train.drop(columns='SalePrice')

Y_train=train['SalePrice']

X_test=test.drop(columns='Id')
Y_train.head()
#from sklearn.ensemble import RandomForestClassifier
'''

# Apply Random Forest

random_forest = RandomForestClassifier(n_estimators=1000)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest

'''
from sklearn.linear_model import RidgeCV



ridge_cv = RidgeCV(alphas=(0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10))

ridge_cv.fit(X_train, Y_train)

ridge_cv_preds=ridge_cv.predict(X_test)
# Apply XGBRegressor

import xgboost as xgb



xgb = xgb.XGBRegressor(n_estimators=340, learning_rate=0.08, max_depth=2)

xgb.fit(X_train,Y_train)

Y_pred = xgb.predict(X_test)
predictions = ( ridge_cv_preds + Y_pred )/2
final_df = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": predictions

    })



solution = pd.DataFrame(final_df)

solution.head()
# Save the dataframe to a csv file

final_df.to_csv('submission.csv',index=False)