import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for visualization

%matplotlib inline

import seaborn as sns # also for visualization

from scipy import stats # general statistical functions



import warnings

warnings.filterwarnings('ignore') # ignore warnings from the different libraries





pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points



import os

print(os.listdir("../input")) # check directory contents
# Import and put the train and test datasets in pandas dataframes



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Drop the 'Id' colum since it's unnecessary for prediction process

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
# Distribution plot for SalePrice

from scipy.stats import norm

sns.distplot(train['SalePrice'] , fit=norm)



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train['SalePrice'])



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')
# Determine which kind of variables are present



train.dtypes.unique()
# Plotting function



def explore_variables(target_name, dt):

    for col in dt.drop(target_name, 1).columns:

        if train.dtypes[train.columns.get_loc(col)] == 'O': # categorical variable

            f, ax = plt.subplots()

            fig = sns.boxplot(x=col, y=target_name, data=dt)

            ax = sns.swarmplot(x=col, y=target_name, data=dt, color=".25", alpha=0.2)

            fig.axis(ymin=0, ymax=800000)

        else: # numerical variable

            fig, ax = plt.subplots()

            ax.scatter(x=dt[col], y=dt[target_name])

            plt.ylabel(target_name, fontsize=13)

            plt.xlabel(col, fontsize=13)

            plt.show()
explore_variables('SalePrice', train)
print("Original size: {}".format(train.shape))



# Drop extreme observations

conditions = [train['LotFrontage'] > 250,

             train['LotArea'] > 100000,

             train['BsmtFinSF1'] > 4000,

             train['TotalBsmtSF'] > 5000,

             train['1stFlrSF'] > 4000,

             np.logical_and(train['GrLivArea'] > 4000, train['SalePrice'] < 300000)]



print("Outliers: {}".format(sum(np.logical_or.reduce(conditions))))
# drop outliers

train = train[np.logical_or.reduce(conditions)==False]
# drop useless variables

train.drop(labels=['PoolArea', 

                   'PoolQC', 

                   'Street', 

                   'Condition2', 

                   'RoofMatl', 

                   'Heating', 

                   'MiscFeature', 

                   'MiscVal', 

                   'Utilities'], axis=1, inplace=True);
#missing data

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
train['Alley'].value_counts()
train.drop('Alley', axis=1, inplace=True)
train['Fence'].value_counts()
# replace NaNs



values = {'Fence': 'NoFence'}

train.fillna(value=values, inplace=True)
# combine other values into one



train.loc[train['Fence'] != 'NoFence', 'Fence'] = 'Fence'
train['FireplaceQu'].value_counts()
# replace NaNs



values = {'FireplaceQu': 'NoFireplace'}

train.fillna(value=values, inplace=True)
# drop 'Fireplaces'



train.drop('Fireplaces', axis=1, inplace=True)
sns.distplot(train['LotFrontage'].dropna())



print("Mean: {}. Median: {}".format(np.mean(train['LotFrontage']), np.median(train['LotFrontage'].dropna())))
# replace NaNs



values = {'LotFrontage': np.median(train['LotFrontage'].dropna())}

train.fillna(value=values, inplace=True)
# drop 'GarageYrBlt'



train.drop('GarageYrBlt', axis=1, inplace=True)
# replace NaNs



values = {var:'NoGarage' for var in ['GarageCond', 'GarageFinish', 'GarageQual', 'GarageType']}

train.fillna(value=values, inplace=True)
# replace NaNs



values = {var:'NoBsmt' for var in ['BsmtFinType2', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']}

train.fillna(value=values, inplace=True)
# replace NaNs



values = {'MasVnrType': 'None', 'MasVnrArea':0}

train.fillna(value=values, inplace=True)
train['Electrical'].value_counts()
# replace NaNs



values = {'Electrical': 'SBrkr'}

train.fillna(value=values, inplace=True)
# combine other values into one



train.loc[train['Electrical'] != 'SBrkr', 'Electrical'] = 'Fusebox'
#missing data

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
explore_variables('SalePrice', train.loc[:, ['SalePrice', 'LotShape'] ])
# Turn irregular lot shapes into one class



train.loc[train['LotShape'] != 'Reg', 'LotShape'] = 'Irregular'
explore_variables('SalePrice', train.loc[:, ['SalePrice', 'LandContour'] ])
# Turn non leveled land contours into one class



train.loc[train['LandContour'] != 'Lvl', 'LandContour'] = 'Unleveled'
explore_variables('SalePrice', train.loc[:, ['SalePrice', 'LotConfig'] ])
# Turn FR* observations into one class



train.loc[np.logical_or(train['LotConfig'] == 'FR2', train['LotConfig'] == 'FR3'), 'LotConfig'] = 'FR'
explore_variables('SalePrice', train.loc[:, ['SalePrice', 'LandSlope'] ])
# Turn Severe slopes into Moderate (there were too few)



train.loc[train['LandSlope'] == 'Sev', 'LandSlope'] = 'Mod'
explore_variables('SalePrice', train.loc[:, ['SalePrice', 'MasVnrType'] ])
# Turn brick veneer types into one class (for example BrkCmn)



train.loc[train['MasVnrType'] == 'BrkFace', 'MasVnrType'] = 'BrkCmn'
explore_variables('SalePrice', train.loc[:, ['SalePrice', 'ExterCond'] ])
# Put poor and excellent external conditions into other conditions



train.loc[train['ExterCond'] == 'Po', 'ExterCond'] = 'Fa'

train.loc[train['ExterCond'] == 'Ex', 'ExterCond'] = 'Gd'
# incorporate log(1+x) transformation !NOT ANYMORE!



# train["SalePrice"] = np.log1p(train["SalePrice"])
# Transform nominal variables that were read as numeric back into nominal



#MSSubClass=The building class

train['MSSubClass'] = train['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

train['OverallCond'] = train['OverallCond'].astype(str)





#Month sold transformed into categorical feature.

train['MoSold'] = train['MoSold'].astype(str)
# transform year variables and drop not useful ones



train['SoldYrAgo'] = 2019 - train['YrSold']

train.drop('YrSold', axis=1, inplace=True)



train['BuiltYrAgo'] = 2019 - train['YearBuilt']

train.drop('YearBuilt', axis=1, inplace=True)



train.drop(labels=['YearRemodAdd', 'GarageFinish'], axis=1, inplace=True)
#Correlation map to see how features are correlated with SalePrice

corrmat = train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True);
# Lets see which correlations are very large in absolute value



corrmat = train.corr()

corrmat[abs(corrmat) > 0.7] = 1

corrmat[abs(corrmat) <= 0.7] = 0



plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True);
# drop GarageArea

train.drop('GarageArea', axis=1, inplace=True)
# drop TotalBsmtSF

train.drop('TotalBsmtSF', axis=1, inplace=True)
# create AreaPerRoom

train['AreaPerRoom'] = train['GrLivArea'] / train['TotRmsAbvGrd']
# dummy coding



train = pd.get_dummies(train)
train.shape
train.to_csv('preprocessed_training.csv', index=False)