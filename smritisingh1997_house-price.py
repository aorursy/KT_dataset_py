import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from scipy import stats

import warnings

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from scipy import stats

from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

warnings.filterwarnings('ignore')

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.options.display.max_columns = None

pd.options.display.max_rows = None
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

train.head()
test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test.head()
print(f"Train dataset has {train.shape[0]} rows and {train.shape[1]} columns")

print(f"Test dataset has {test.shape[0]} rows and {test.shape[1]} columns")
#Gives statistical information about numerical variables

train.describe().T
#Gives information about the features (like data type etc.)

train.info()
#Gives count of different data types

train.get_dtype_counts()
#Checking for missing values

def missing_percentage(df):

    total = df.isnull().sum().sort_values(ascending=False)[df.isnull().sum().sort_values(ascending=False)!=0]

    percent = (((df.isnull().sum().sort_values(ascending=False)) / (df.shape[0])) * 100)[((df.isnull().sum().sort_values(ascending=False)) / (df.shape[0])) * 100 != 0]

    return(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']))

missing_percentage(train)
missing_percentage(test)
fig, ax =plt.subplots(2,2, figsize=(12, 8))

sns.distplot(train["SalePrice"], ax=ax[0][0])

sns.boxplot(train["SalePrice"], ax=ax[0][1])

fig.delaxes(ax[1][0])

fig.delaxes(ax[1][1])

fig.tight_layout(pad=3.0)
print(f"Skewness value of SalePrice is {train['SalePrice'].skew()}")

print(f"Kurtosis value of SalePrice is {train['SalePrice'].kurt()}")
#Correlation value between target (SalePrice) and other numerical variables

(train.corr())['SalePrice'].sort_values(ascending=False)[1:]
len(train.corr()['SalePrice'])
fig, ax =plt.subplots(figsize=(12, 8))

sns.scatterplot(x='OverallQual', y='SalePrice', data=train)
fig, ax =plt.subplots(figsize=(12, 8))

sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)
# fig, ax =plt.subplots(figsize=(12, 8))

# sns.scatterplot(x='GarageCars', y='SalePrice', data=train)
fig, ax =plt.subplots(figsize=(12, 8))

sns.scatterplot(x='GarageArea', y='SalePrice', data=train)
fig, ax =plt.subplots(figsize=(12, 8))

sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=train)
fig, ax =plt.subplots(figsize=(12, 8))

sns.scatterplot(x='1stFlrSF', y='SalePrice', data=train)
# fig, ax =plt.subplots(figsize=(12, 8))

# sns.scatterplot(x='FullBath', y='SalePrice', data=train)
# fig, ax =plt.subplots(figsize=(12, 8))

# sns.scatterplot(x='TotRmsAbvGrd', y='SalePrice', data=train)
# fig, ax =plt.subplots(figsize=(12, 8))

# sns.scatterplot(x='YearBuilt', y='SalePrice', data=train)
# fig, ax =plt.subplots(figsize=(12, 8))

# sns.scatterplot(x='YearRemodAdd', y='SalePrice', data=train)
# fig, ax =plt.subplots(figsize=(12, 8))

# sns.scatterplot(x='GarageYrBlt', y='SalePrice', data=train)
fig, ax =plt.subplots(figsize=(12, 8))

sns.scatterplot(x='MasVnrArea', y='SalePrice', data=train)
# fig, ax =plt.subplots(figsize=(12, 8))

# sns.scatterplot(x='Fireplaces', y='SalePrice', data=train)
#Deleting the outliers from the given dataset, outliers found using plot of GrLivArea vs SalePrice column



train = train[train.GrLivArea < 4500]

train.reset_index(drop=True, inplace=True)



#save previous train dataset

previous_train = train.copy()
fig, (ax1, ax2) =plt.subplots(figsize=(12, 8), ncols=2,sharey=False)

# Scatter plot between GrLivArea and SalePrice

sns.scatterplot(x='GrLivArea', y='SalePrice', data=train, ax=ax1)

# Putting a regression line plot between GrLivArea and SalePrice in the above scatter plot

sns.regplot(x='GrLivArea', y='SalePrice', data=train, ax=ax1)

# Scatter plot between MasVnrArea and SalePrice

sns.scatterplot(x='MasVnrArea', y='SalePrice', data=train, ax=ax2)

# Putting a regression line plot between MasVnrArea and SalePrice in the above scatter plot

sns.regplot(x='MasVnrArea', y='SalePrice', data=train, ax=ax2)
#Residual plot between GrLivArea and SalePrice

plt.subplots(figsize = (12,8))

sns.residplot(train.GrLivArea, train.SalePrice)
fig, (ax1, ax2) =plt.subplots(figsize=(14, 8), ncols=2,sharey=False)

sns.distplot(train['SalePrice'], ax=ax1)

sns.boxplot(train['SalePrice'], ax=ax2)
#Applying numpy's log1p transformation, log1p is aaplied to handle the situation of any zero value present in SalePrice column

train['SalePrice'] = np.log1p(train['SalePrice'])
#Again plotting SalePrice to check whether it is normally distributed now or not

fig, (ax1, ax2) =plt.subplots(figsize=(14, 8), ncols=2,sharey=False)

sns.distplot(train['SalePrice'], ax=ax1)

sns.boxplot(train['SalePrice'], ax=ax2)
#Plotting residual plot between GrLivArea and SalePrice to find whether heteroscedasticity is resolved or not

fig, (ax1, ax2) =plt.subplots(figsize=(14, 8), ncols=2,sharey=False)

#Residual plot, without transforming the target column SalePrice

sns.residplot(previous_train.GrLivArea, previous_train.SalePrice, ax=ax1)

#Residual plot, after transforming the target column SalePrice

sns.residplot(train.GrLivArea, train.SalePrice, ax=ax2)
sns.set_style('whitegrid')

plt.subplots(figsize = (30,20))



#Generate a mask for the upper triangle

mask = np.zeros_like(train.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True





sns.heatmap(train.corr(), cmap=sns.diverging_palette(20, 220, n=200), mask = mask, annot=True, center = 0);



#Give title. 

plt.title("Heatmap of all the Features", fontsize = 30);
#Dropping Id column from both train and test dataset

train.drop('Id', axis=1, inplace=True)

test.drop('Id', axis=1, inplace=True)



#Storing the target variable in y

y = train['SalePrice'].reset_index(drop=True)



previous_train = train.copy()
#Combining train and test dataset together

all_data = pd.concat((train, test)).reset_index(drop=True)
l = []

for i in (all_data.select_dtypes(include ='object').columns):

    if(i != 'SalePrice'):

        data_crosstab = pd.crosstab(all_data[i], all_data['SalePrice'], margins = False)

        stat, p, dof, expected = stats.chi2_contingency(data_crosstab)

        prob=0.95

        alpha = 1.0 - prob

        if p <= alpha:

            print(i, ' : Dependent (reject H0)')

        else:

            l.append(i)

            print(i, ' : Independent (fail to reject H0)')
for i in all_data[l].columns:

    print(f'Plot between {i} and SalePrice')

    sns.boxplot(x=i, y=all_data['SalePrice'], data=all_data)

    plt.show()
all_data['Utilities'] = all_data['Utilities'].replace(['NoSewr', 'NoSeWa', 'ELO'], 'NoSeWa')
sns.boxplot(x='Utilities', y='SalePrice', data=all_data)
all_data['Utilities'].value_counts()
sns.boxplot(x='Street', y='SalePrice', data=all_data)
all_data['YrSold'] = all_data['YrSold'].astype('int64')

all_data['MoSold'] = all_data['MoSold'].astype('int64')

all_data['YearRemodAdd'] = all_data['YearRemodAdd'].astype('int64')
all_data['Old'] = all_data['YrSold'] - all_data['YearRemodAdd']
#From following correlation, we can say that Old column is effecting target column SalePrice, as correlation value is 0.57 (negative)

df_corr_num_num=all_data.loc[:,["SalePrice","Old"]]

sns.heatmap(df_corr_num_num.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
#Finding correlation of newly created variable Old, with the variables using which it is created

print(all_data[['Old', 'YrSold']].corr())

print(all_data[['Old','YearRemodAdd']].corr())
#Dropping YearRemodAdd and MoSold

all_data.drop(['YearRemodAdd', 'MoSold'], axis=1, inplace=True)
missing_percentage(all_data)
#Replacing null values with None as in this case null values means that particular feature is not present in the house

missing_val_col = ["Alley", 

                   "PoolQC", 

                   "MiscFeature",

                   "Fence",

                   "FireplaceQu",

                   "GarageType",

                   "GarageFinish",

                   "GarageQual",

                   "GarageCond",

                   'BsmtQual',

                   'BsmtCond',

                   'BsmtExposure',

                   'BsmtFinType1',

                   'BsmtFinType2',

                   'MasVnrType']



for i in missing_val_col:

    all_data[i] = all_data[i].fillna('None')
#The following features also contains null value for a reason, implies that area or square feet is zero, so replacing with 0 value

missing_val_col2 = ['BsmtFinSF1',

                    'BsmtFinSF2',

                    'BsmtUnfSF',

                    'TotalBsmtSF',

                    'BsmtFullBath', 

                    'BsmtHalfBath', 

                    'GarageYrBlt',

                    'GarageArea',

                    'GarageCars',

                    'MasVnrArea']



for i in missing_val_col2:

    all_data[i] = all_data[i].fillna(0)
# Replaced all missing values in LotFrontage by imputing the mean value of each neighborhood

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: 

                                                                                   x.fillna(x.mean()))
# Replaced all missing values in MSZoning by imputing the mode value of each MSSubClass

#Converting MSSubClass to categorical data type 

all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: 

                                                                            x.fillna(x.mode()[0]))
#Converting YrSold, MoSold to categorical data type 

# all_data['YrSold'] = all_data['YrSold'].astype(str)

# all_data['MoSold'] = all_data['MoSold'].astype(str)
#Replace the remaining caegorical columns with their respective mode values

all_data['Functional'] = all_data['Functional'].fillna(all_data['Functional'].mode()[0]) 

all_data['Utilities'] = all_data['Utilities'].fillna(all_data['Utilities'].mode()[0]) 

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0]) 

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0]) 

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
missing_percentage(all_data)
numeric_feats = all_data.dtypes[all_data.dtypes != object].index
skew_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

skew_feats
def fixing_skewness(df):

    numeric_feats = df.dtypes[df.dtypes != object].index

    

    skew_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

    

    high_skew = skew_feats[abs(skew_feats) > 0.5].index

    

    for i in high_skew:

        df[i] = boxcox1p(df[i], boxcox_normmax(df[i] + 1))

        

fixing_skewness(all_data)
all_data['TotalSF'] = (all_data['TotalBsmtSF']

                      + all_data['1stFlrSF']

                      +all_data['2ndFlrSF'])



# all_data['YrBltAndRemod'] = (all_data['YearBuilt'] 

#                              + all_data['YearRemodAdd'])



all_data['Total_porch_sf'] = (all_data['OpenPorchSF']

                             + all_data['WoodDeckSF']

                             + all_data['3SsnPorch']

                             + all_data['EnclosedPorch']

                             + all_data['ScreenPorch'])



all_data['Total_Bathrooms'] = (all_data['FullBath']

                              + 0.5 * all_data['HalfBath']

                              + all_data['BsmtFullBath']

                              + 0.5 * all_data['BsmtHalfBath'])



all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] 

                                 + all_data['BsmtFinSF2'] 

                                 + all_data['1stFlrSF'] 

                                 + all_data['2ndFlrSF']

                                )
all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['has1stfloor'] = all_data['1stFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data.shape
#Finding correlation with newly created variable TotalSF

df_corr_num_num=all_data.loc[:,["SalePrice","TotalSF"]]

sns.heatmap(df_corr_num_num.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
#Finding correlation of newly created variable 'TotalSF' with the variable using which it is created

print(all_data[['TotalSF', 'TotalBsmtSF']].corr())

print(all_data[['TotalSF', '1stFlrSF']].corr())

print(all_data[['TotalSF','2ndFlrSF']].corr())
print(all_data[['2ndFlrSF', 'SalePrice']].corr())
#Finding correlation with newly created variable Total_porch_sf

df_corr_num_num=all_data.loc[:,["SalePrice","Total_porch_sf"]]

sns.heatmap(df_corr_num_num.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
print(all_data[['Total_porch_sf', 'OpenPorchSF']].corr())

print(all_data[['Total_porch_sf', 'WoodDeckSF']].corr())

print(all_data[['Total_porch_sf','3SsnPorch']].corr())

print(all_data[['Total_porch_sf','ScreenPorch']].corr())

print(all_data[['Total_porch_sf','EnclosedPorch']].corr())
print(all_data[['WoodDeckSF', 'SalePrice']].corr())
#Dropping WoodDeckSF column, as it is less correlated with target column as compare to newly created variable, and highly correlated with target variable

all_data.drop('WoodDeckSF', axis=1, inplace=True)
#Finding correlation with newly created variable Total_Bathrooms

df_corr_num_num=all_data.loc[:,["SalePrice","Total_Bathrooms"]]

sns.heatmap(df_corr_num_num.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
print(all_data[['Total_Bathrooms', 'FullBath']].corr())

print(all_data[['Total_Bathrooms', 'HalfBath']].corr())

print(all_data[['Total_Bathrooms','BsmtFullBath']].corr())

print(all_data[['Total_Bathrooms','BsmtHalfBath']].corr())
#Finding correlation with newly created variable Total_sqr_footage

df_corr_num_num=all_data.loc[:,["SalePrice","Total_sqr_footage"]]

sns.heatmap(df_corr_num_num.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
print(all_data[['Total_sqr_footage', 'BsmtFinSF1']].corr())

print(all_data[['Total_sqr_footage', 'BsmtFinSF2']].corr())

print(all_data[['Total_sqr_footage','1stFlrSF']].corr())

print(all_data[['Total_sqr_footage','2ndFlrSF']].corr())
print(all_data[['SalePrice', '2ndFlrSF']].corr())
#Dropping 2ndFlrSF column, as it is highly correlated with the newly created variable, and effecting less target variable as compare to the other newly created variables

all_data.drop('2ndFlrSF', axis=1, inplace=True)
#Finding correlation with newly created variable haspool

df_corr_num_num=all_data.loc[:,["SalePrice","haspool"]]

sns.heatmap(df_corr_num_num.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
#Dropping haspool, as it is correlated with target column but the value is very less 0.077

all_data.drop('haspool', axis=1, inplace=True)
#Finding correlation with newly created variable hasgarage

df_corr_num_num=all_data.loc[:,["SalePrice","hasgarage"]]

sns.heatmap(df_corr_num_num.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
print(all_data[['hasgarage', 'GarageArea']].corr())

print(all_data[['SalePrice', 'GarageArea']].corr())
#Finding correlation with newly created variable hasfireplace

df_corr_num_num=all_data.loc[:,["SalePrice","hasfireplace"]]

sns.heatmap(df_corr_num_num.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
print(all_data[['hasfireplace', 'Fireplaces']].corr())

print(all_data[['SalePrice', 'Fireplaces']].corr())
#Dropping Fireplaces column as it is highly correlated with the newly created variable, and also it is less correlated to the target column as compare to the newly created variable

all_data.drop('Fireplaces', axis=1, inplace=True)
#Finding correlation with newly created variable hasbsmt

df_corr_num_num=all_data.loc[:,["SalePrice","hasbsmt"]]

sns.heatmap(df_corr_num_num.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
#Dropping hasbsmt column as it is correlated to the target column but the correlation value is very small

all_data.drop('hasbsmt', axis=1, inplace=True)
#Finding correlation with newly created variable has2ndfloor

df_corr_num_num=all_data.loc[:,["SalePrice","has2ndfloor"]]

sns.heatmap(df_corr_num_num.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
#Dropping has2ndfloor column as it is correlated to the target column but the correlation value is very small

all_data.drop('has2ndfloor', axis=1, inplace=True)
#Finding correlation with newly created variable has1stfloor

df_corr_num_num=all_data.loc[:,["SalePrice","has1stfloor"]]

sns.heatmap(df_corr_num_num.corr(), annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')
#Dropping has1stfloor column as it is correlated to the target column but the correlation value is very small

all_data.drop('has1stfloor', axis=1, inplace=True)
#Dropping target variable SalePrice from the whole data

all_data.drop('SalePrice', axis=1, inplace=True)
#We are dropping following columns as in 'PoolQC' only 'NA' is present, in 'Street' only 'Pave' is present, in 'Utilities' only 'AllPub' is present

#We are dropping 'YearBuilt', 'GarageYrBlt' columns because these are correlated with 'YearRemodAdd' with 83% correlation value

all_data = all_data.drop(['Utilities', 'Street', 'PoolQC','YearBuilt', 'GarageYrBlt'], axis=1)
#As only two value is present in CentralAir column, so replacing those with 0 and 1, and converting column data type to integer

all_data['CentralAir'] = all_data['CentralAir'].replace({'Y':1, 'N':0})

all_data['CentralAir'] = all_data['CentralAir'].astype('int64')
#Creating dummy variable

final_features = pd.get_dummies(all_data).reset_index(drop=True)

final_features.shape
#Separating the train and test dataset

X = final_features.iloc[:len(y), :]

X_sub = final_features.iloc[len(y):, :]
#Dropping outliers ----research required

outliers = [30, 88, 462, 631, 1322]

X = X.drop(X.index[outliers])

y = y.drop(y.index[outliers])
for i in X.columns:

    counts = X[i].value_counts()

    print (counts)
#Function to determine overfitted features, that is feature containing only one value in more than 99.94 cases

def overfit_reducer(df):

    overfit = []

    for i in df.columns:

        count = df[i].value_counts()

        zero_index_value = count.iloc[0]

        

        if (((zero_index_value / len(df)) * 100) > 99.94):

            overfit.append(i)

            

    overfit = list(overfit)

    return overfit
#Finding the list of overfitted features using above user-defined function

overfitted_features = overfit_reducer(X)

#Dropping the overfitted columns from the final dataframes

X.drop(overfitted_features, axis=1, inplace=True)

X_sub.drop(overfitted_features, axis=1, inplace=True)
X.shape, X_sub.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
lin_reg = LinearRegression(normalize=True, n_jobs=-1)
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
print((mean_squared_error(y_test, y_pred)))
from sklearn.linear_model import Ridge

#Assiging different sets of alpha values to explore which can be the best fit for the model

alpha_ridge = [-3,-2,-1,1e-15, 1e-10, 1e-8,1e-5,1e-4, 1e-3,1e-2,0.5,1,1.5, 2,3,4, 5, 10, 20, 30, 40]

temp_rss = {}

temp_mse = {}



for i in alpha_ridge:

    #Assigin each model

    ridge = Ridge(alpha=i, normalize=True)

    # fit the model

    ridge.fit(X_train, y_train)

    #Predicting the target value based on X_test

    y_pred = ridge.predict(X_test)

    

    mse = mean_squared_error(y_test, y_pred)

    rss = sum((y_test - y_pred) ** 2)

    

    temp_rss[i] = rss

    temp_mse[i] = mse

    
for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):

    print("%s: %s" % (key, value))
for key, value in sorted(temp_rss.items(), key=lambda item: item[1]):

    print("%s: %s" % (key, value))
from sklearn.linear_model import Lasso 

temp_rss = {}

temp_mse = {}

for i in alpha_ridge:

    #Assigin each model. 

    lasso_reg = Lasso(alpha= i, normalize=True)

    #fit the model. 

    lasso_reg.fit(X_train, y_train)

    #Predicting the target value based on "X_test"

    y_pred = lasso_reg.predict(X_test)



    mse = mean_squared_error(y_test, y_pred)

    rss = sum((y_pred-y_test)**2)

    temp_mse[i] = mse

    temp_rss[i] = rss
for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):

    print("%s: %s" % (key, value))
for key, value in sorted(temp_rss.items(), key=lambda item: item[1]):

    print("%s: %s" % (key, value))
from sklearn.linear_model import ElasticNet

temp_rss = {}

temp_mse = {}

for i in alpha_ridge:

    #Assigin each model. 

    lasso_reg = ElasticNet(alpha= i, normalize=True)

    #fit the model. 

    lasso_reg.fit(X_train, y_train)

    #Predicting the target value based on "X_test"

    y_pred = lasso_reg.predict(X_test)



    mse = mean_squared_error(y_test, y_pred)

    rss = sum((y_pred-y_test)**2)

    temp_mse[i] = mse

    temp_rss[i] = rss
for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):

    print("%s: %s" % (key, value))
for key, value in sorted(temp_rss.items(), key=lambda item: item[1]):

    print("%s: %s" % (key, value))
corr_matrix = X.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
X = X.drop(X[to_drop], axis=1)

X_sub = X_sub.drop(X_sub[to_drop], axis=1)
from scipy.special import inv_boxcox1p
from sklearn.ensemble import GradientBoostingRegressor 

gbr = GradientBoostingRegressor(n_estimators=3000, 

                                learning_rate=0.05, 

                                max_depth=4, 

                                max_features='sqrt', 

                                min_samples_leaf=15, 

                                min_samples_split=10, 

                                loss='huber', 

                                random_state =42)
from xgboost import XGBRegressor

xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)
rf = RandomForestRegressor(n_estimators=20)
xgb_model_full_data = xgboost.fit(X, y)
gbr_model_full_data = gbr.fit(X, y)
rf_fit = rf.fit(X,y)
y_xgb_pred = xgb_model_full_data.predict(X_sub)
y_gbr_pred = gbr_model_full_data.predict(X_sub)
y_rf_pred = rf_fit.predict(X_sub)
y_xgb_pred = inv_boxcox1p(y_xgb_pred, 0)

y_gbr_pred = inv_boxcox1p(y_gbr_pred, 0)

y_rf_pred = inv_boxcox1p(y_rf_pred, 0)
submission_df_xgb = pd.DataFrame(y_xgb_pred,columns=['SalePrice'])

submission_df_gbr = pd.DataFrame(y_gbr_pred,columns=['SalePrice'])

submission_df_rf = pd.DataFrame(y_rf_pred,columns=['SalePrice'])
test_sub = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_sub.head()
submission_df_xgb['Id'] = test_sub['Id']

submission_df_xgb = submission_df_xgb[['Id', 'SalePrice']]



submission_df_gbr['Id'] = test_sub['Id']

submission_df_gbr = submission_df_gbr[['Id', 'SalePrice']]



submission_df_rf['Id'] = test_sub['Id']

submission_df_rf = submission_df_rf[['Id', 'SalePrice']]
submission_df_xgb.to_csv('/kaggle/working/submission_xgb.csv', index=False)

submission_df_gbr.to_csv('/kaggle/working/submission_gbr.csv', index=False)

submission_df_rf.to_csv('/kaggle/working/submission_rf.csv', index=False)