# Import necesaary packages

import numpy as np 

import pandas as pd 

import seaborn as sb

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.ensemble import RandomForestClassifier

# suppress warnings from final output

import warnings

warnings.simplefilter("ignore")

# List files provided

import os

print(os.listdir("../input"))

import xgboost

from sklearn.metrics import explained_variance_score

from xgboost import XGBRegressor
# Load datasets

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

train.head()
# To operate on both dataframes simulatenously

combine=[train,test]
# Check if any data is duplicated

sum(train['Id'].duplicated()),sum(test['Id'].duplicated())
# Drop Id column from train dataset

train.drop(columns=['Id'],inplace=True)
miss_cols=['MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

           'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond']

zero_list=['LotFrontage','MasVnrArea','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2',

           'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath']

drop_cols=['PoolQC','Fence','MiscFeature','GarageYrBlt','Alley']

for df in combine:

    # Fill missing values in cetgorical variables with None

    for col in miss_cols:

        df[col]=df[col].fillna('None')

    # Fill missing values in numerical variables with 0

    for col in zero_list:

        df[col]=df[col].fillna(0)

    # Drop columns with large number of missing values

    df.drop(columns=drop_cols,inplace=True)

# Fill missing value in Electrical in train dataset

index=train[train['Electrical'].isnull()].index

train.loc[index,'Electrical']=train['Electrical'].mode()[0]
# Fill missing values in categorical variables in test dataset

mode_list=['Utilities','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType','MSZoning']

for col in mode_list:

    mode=test[col].mode()

    test[col]=test[col].fillna(mode[0])
# Check if the above operations worked correctly

train.isnull().sum().max(),test.isnull().sum().max()
# Plot the correlation matrix

imp_list=['SalePrice','OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea' ,'GarageArea',

         'GarageCars','LotArea','PoolArea']

corrmat = train[imp_list].corr()

f, ax = plt.subplots(figsize=(10, 9))

sb.heatmap(corrmat, vmax=.8, square=True,cbar=True, annot=True, fmt='.2f', annot_kws={'size': 10});
# Set the base color as blue

base_color=sb.color_palette()[0]
# Univariate plot SalePrice

sb.distplot(train['SalePrice']);
# Univariate plot of SalePrice using log transform

bins=10**np.arange(0,np.log(train['SalePrice'].max())+0.05,0.05)

plt.hist(data=train,x='SalePrice',bins=bins);

plt.xscale('log');

plt.xlim(10000,1000000);

xticks=[10000,30000,100000,300000,1000000]

plt.xticks(xticks,xticks);

plt.xlabel('Sale Price in $');

plt.title('Distribution of Sale Price');
# Univariate plot of Overall Quality

sb.countplot(data=train,x='OverallQual',color=base_color);

plt.title('Distribution of Overall-Quality');
# Bivariate plot of SalePrice and OverallQual

sb.boxplot(data=train,x='OverallQual',y='SalePrice',color=base_color);

plt.title('Sale Price vs. Overall Quality');
# Bivariate plot SalePrice and YearBuilt

plt.figure(figsize=(18,6))

sb.barplot(data=train,x='YearBuilt',y='SalePrice',color=base_color);

plt.xticks(rotation=90);

plt.title('SalevPrice vs. Year Built');
# Bivariate plot of Sale Price and Basement Area 

sb.regplot(data=train,x='TotalBsmtSF',y='SalePrice',scatter_kws={'alpha':1/5});
# Removing outliers from Basement Area

index=train[train['TotalBsmtSF']>4000].index

train.drop(index,inplace=True)
# Bivariate plot of Sale Price and Basement Area 

plt.figure(figsize=(10,5))

bins_x = np.arange(0, train['TotalBsmtSF'].max()+300, 300)

bins_y = np.arange(0, train['SalePrice'].max()+50000, 50000)

h2d=plt.hist2d(data=train,x='TotalBsmtSF',y='SalePrice',cmin=0.5,cmap='viridis_r',bins=[bins_x,bins_y]);

plt.colorbar();

plt.xlabel('Basement Area in Square Feet');

plt.ylabel('Sale Price in $');

plt.title('Sale Price vs. Basement Area ');

counts = h2d[0]

# loop through the cell counts and add text annotations for each

for i in range(counts.shape[0]):

    for j in range(counts.shape[1]):

        c = counts[i,j]

        if c >= 100: # increase visibility on darkest cells

            plt.text(bins_x[i]+150, bins_y[j]+25000, int(c),

                     ha = 'center', va = 'center', color = 'white')

        elif c > 0:

            plt.text(bins_x[i]+150, bins_y[j]+25000, int(c),

                     ha = 'center', va = 'center', color = 'black')
# Bivariate plot of Sale Price and Above Ground Area

sb.regplot(data=train,x='GrLivArea',y='SalePrice',scatter_kws={'alpha':1/5});
# Remove outliers from Above Ground Area

index=train[train['GrLivArea']>4000].index

train.drop(index,inplace=True)
# Bivariate plot of Sale Price and Above Ground Area

plt.figure(figsize=(10,5))

bins_x = np.arange(0, train['GrLivArea'].max()+300, 300)

bins_y = np.arange(0, train['SalePrice'].max()+50000, 50000)

h2d=plt.hist2d(data=train,x='GrLivArea',y='SalePrice',cmin=0.5,cmap='viridis_r',bins=[bins_x,bins_y]);

plt.colorbar();

plt.xlabel('Above Ground Area in Square Feet');

plt.ylabel('Sale Price in $');

plt.title('Sale Price vs. Above Ground Area ');

counts = h2d[0]

# loop through the cell counts and add text annotations for each

for i in range(counts.shape[0]):

    for j in range(counts.shape[1]):

        c = counts[i,j]

        if c >= 100: # increase visibility on darkest cells

            plt.text(bins_x[i]+150, bins_y[j]+25000, int(c),

                     ha = 'center', va = 'center', color = 'white')

        elif c > 0:

            plt.text(bins_x[i]+150, bins_y[j]+25000, int(c),

                     ha = 'center', va = 'center', color = 'black')
sb.regplot(data=train,x='GarageArea',y='SalePrice',scatter_kws={'alpha':1/5});
# Bivariate plot of Sale Price and Garage Area

plt.figure(figsize=(10,5))

bins_x = np.arange(0, train['GarageArea'].max()+100, 100)

bins_y = np.arange(0, train['SalePrice'].max()+50000, 50000)

h2d=plt.hist2d(data=train,x='GarageArea',y='SalePrice',cmin=0.5,cmap='viridis_r',bins=[bins_x,bins_y]);

plt.colorbar();

plt.xlabel('Garage Area in Square Feet');

plt.ylabel('Sale Price in $');

plt.title('Sale Price vs. Garage Area ');

counts = h2d[0]

# loop through the cell counts and add text annotations for each

for i in range(counts.shape[0]):

    for j in range(counts.shape[1]):

        c = counts[i,j]

        if c >= 100: # increase visibility on darkest cells

            plt.text(bins_x[i]+50, bins_y[j]+25000, int(c),

                     ha = 'center', va = 'center', color = 'white')

        elif c > 0:

            plt.text(bins_x[i]+50, bins_y[j]+25000, int(c),

                     ha = 'center', va = 'center', color = 'black')
# Bivariate plot between all the areas

plt.figure(figsize=(16,10))

plt.subplot(3,3,1)

sb.regplot(data=train,x='TotalBsmtSF',y='GrLivArea',x_jitter=0.3,scatter_kws={'alpha':1/5});

plt.subplot(3,3,2)

sb.regplot(data=train,x='TotalBsmtSF',y='GarageArea',x_jitter=0.3,scatter_kws={'alpha':1/5});

plt.subplot(3,3,3)

sb.regplot(data=train,x='GrLivArea',y='GarageArea',x_jitter=0.3,scatter_kws={'alpha':1/5});
# Bivariate plot of Quality vs. Area

plt.figure(figsize=(10,7))

plt.subplot(3,1,1)

sb.boxplot(data=train,y='TotalBsmtSF',x='OverallQual',color=base_color);

plt.subplot(3,1,2)

sb.boxplot(data=train,y='GarageArea',x='OverallQual',color=base_color);

plt.subplot(3,1,3)

sb.boxplot(data=train,y='GrLivArea',x='OverallQual',color=base_color);
# Multivariate plot of SalePrice vs. YearBuilt and OverallQual

plt.figure(figsize=(18,7))

sb.pointplot(data=train,x='YearBuilt',y='SalePrice',hue='OverallQual',palette='viridis_r',linestyles="");

plt.xticks(rotation=90);

plt.title('Sale Price vs. Year Built and Overall Quality');

# Multivariate plot of SalePrice Vs. Quality and GarageQual, BsmtQual

plt.figure(figsize=(14,7))

plt.subplot(2,1,1)

plt.title('Sale Price Vs. Quality and Garage Quality, Basement Quality');

sb.pointplot(data=train,hue='OverallQual',y='SalePrice',x='GarageQual',palette='viridis_r',linestyles="",

            order=['None','Po','Fa','TA','Gd','Ex']);

plt.legend(loc=1);

plt.subplot(2,1,2)

sb.pointplot(data=train,hue='OverallQual',y='SalePrice',x='BsmtQual',palette='viridis_r',linestyles="",

            order=['None','Po','Fa','TA','Gd','Ex']);

plt.legend(loc=1);

def mean_poly(x, y, bins = 10, **kwargs):

    # set bin edges if none or int specified

    if type(bins) == int:

        bins = np.linspace(x.min(), x.max(), bins+1)

    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2



    # compute counts

    data_bins = pd.cut(x, bins, right = False,

                       include_lowest = True)

    means = y.groupby(data_bins).mean()



    # create plot

    plt.errorbar(x = bin_centers, y = means, **kwargs)
# Lineplot of SalePrice and TotalBsmtSF using OverallQual as hue

bin_edges = np.arange(0, train['TotalBsmtSF'].max()+250, 250)

g = sb.FacetGrid(data = train, hue = 'OverallQual',size=7,palette='viridis_r');

g.map(mean_poly, "TotalBsmtSF", "SalePrice", bins = bin_edges);

g.set_ylabels('mean(SalePrice)');

g.add_legend();

plt.title('SalePrice vs. Basement Area by Overall Quality');
# Lineplot of SalePrice and GrLivArea using OverallQual as hue

bin_edges = np.arange(0, train['GrLivArea'].max()+250, 250)

g = sb.FacetGrid(data = train, hue = 'OverallQual',size=7,palette='viridis_r');

g.map(mean_poly, "GrLivArea", "SalePrice", bins = bin_edges);

g.set_ylabels('mean(SalePrice)');

g.add_legend();

plt.title('SalePrice vs. Above Ground Area by Overall Quality');
# Lineplot of SalePrice and GarageArea using OverallQual as hue

bin_edges = np.arange(0, train['GarageArea'].max()+150, 150)

g = sb.FacetGrid(data = train, hue = 'OverallQual',size=7,palette='viridis_r');

g.map(mean_poly, "GarageArea", "SalePrice", bins = bin_edges);

g.set_ylabels('mean(SalePrice)');

g.add_legend();

plt.title('SalePrice vs. Garage Area by Overall Quality');
# Drop object features

'''

for df in combine:

    df.drop(columns=['MSZoning','SaleCondition','SaleType','PavedDrive','GarageCond','GarageQual','GarageFinish',

                    'GarageType','FireplaceQu','Functional','KitchenQual','Heating','HeatingQC','CentralAir',

                     'Electrical','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure',

                     'BsmtFinType1','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Street',

                     'LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1',

                     'Condition2','BldgType','HouseStyle','BsmtFinType2' ],inplace=True)

'''
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
'''

# Apply Random Forest

random_forest = RandomForestClassifier(n_estimators=1000)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest

'''
# Apply XGBRegressor

xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,

                           colsample_bytree=1, max_depth=7)

xgb.fit(X_train,Y_train)

Y_pred = xgb.predict(X_test)
final_df = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": Y_pred

    })

# Save the dataframe to a csv file

final_df.to_csv('submission.csv',index=False)