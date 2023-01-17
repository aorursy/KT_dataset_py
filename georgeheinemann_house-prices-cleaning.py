# import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import skew

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.preprocessing import PowerTransformer

import os

%matplotlib inline
# Limits floats output to 3 decimal points

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) 
# Files in directory

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#import data

train = pd.read_csv('/kaggle/input/train.csv',index_col=0)

test = pd.read_csv('/kaggle/input/test.csv',index_col=0)



#Check size and look

print(train.shape)

train.head()
#Check size and look

print(test.shape)

test.head()
# Combine train and test for pre-processing

df_all = pd.concat([train[train.columns[:-1]],test])

df_all.head(5)
# Save training observations for later

y = train.SalePrice
#Number and types of columns

df_all.info()
# Looking at distribution of house prices

plt.figure(figsize=[20,5])



# Histogram plot

plt.subplot(1,2,1)

sns.distplot(y)

plt.title('Standard')



# Skewness and kurtosis

print("Skewness: %f" % y.skew())

print("Kurtosis: %f" % y.kurt())



# Due to skew (>1), we'll log it and show it now better approximates a normal distribution

plt.subplot(1,2,2)

sns.distplot(np.log(y))

plt.title('Log transformation')
# Convert y into log(y)

y = np.log(y)
# Look for missing data

plt.figure(figsize=[20,5])

sns.heatmap(df_all.isnull(),yticklabels=False,cbar=False)
# Dropping data that is heavily missing (will deal with partially missing later)

df_all.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
# Values for feature 'MSSubClass'

df_all.MSSubClass.unique()
# Use dictionaries to convert across

df_all = df_all.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 

                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 

                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 

                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"}})
# Values for feature 'MoSold'

df_all.MoSold.unique()
# Use dictionaries to convert into month strings

df_all = df_all.replace({"MoSold" : {1 : "January", 2 : "February", 3 : "March", 4 : "April", 

                                       5 : "May", 6 : "June", 7 : "July", 8 : "August", 

                                       9 : "September", 10 : "October", 11 : "November", 12 : "December"}})
# Check it worked as expected

df_all[['MSSubClass','MoSold']].head(5)
# No. of categoric variables

cat_feats = df_all.dtypes[df_all.dtypes == "object"].index.tolist()

print(str(len(cat_feats)) + ' categoric features')
# No. of numerical variables

num_feats = df_all.dtypes[df_all.dtypes != "object"].index.tolist()

print(str(len(num_feats)) + ' numeric features')
# Return list of categoric columns with missing variables

cat_feats_missing = df_all[cat_feats].columns[df_all[cat_feats].isna().any()].tolist()

cat_feats_missing
# Show value occurences to determine appropriate NaN replacement

for i in cat_feats_missing:

    print(i)

    print(df_all[i].value_counts(dropna=False))

    print("")
# Make replacements into most likely value



# Likely RL

df_all.MSZoning.fillna('RL', inplace = True)

# Drop utilities as only 1 is different which doesn't help

df_all.drop('Utilities',axis=1,inplace=True)

# Likely VinylSd

df_all.Exterior1st.fillna('VinylSd', inplace = True)

# Likely VinylSd

df_all.Exterior2nd.fillna('VinylSd', inplace = True)

# Likely no masonary

df_all.MasVnrType.fillna('None', inplace = True)

# Likely no basement

df_all.BsmtQual.fillna('No basement',inplace = True)

# Likely no basement

df_all.BsmtCond.fillna('No basement',inplace = True)

# Likely no basement

df_all.BsmtExposure.fillna('No basement',inplace = True)

# Likely no basement

df_all.BsmtFinType1.fillna('No basement',inplace = True)

# Likely no basement

df_all.BsmtFinType2.fillna('No basement',inplace = True)

# Likely standard electrical

df_all.Electrical.fillna('SBrkr',inplace = True)

# Likely typical kitchen

df_all.KitchenQual.fillna('TA',inplace = True)

# Likely typical functionality

df_all.Functional.fillna('Typ',inplace = True)

# Likely no garage

df_all.GarageType.fillna('No garage',inplace = True)

# Likely no garage

df_all.GarageFinish.fillna('No garage',inplace = True)

# Likely no garage

df_all.GarageQual.fillna('No garage',inplace = True)

# Likely no garage

df_all.GarageCond.fillna('No garage',inplace = True)

# Likely typical sale type

df_all.SaleType.fillna('WD',inplace = True)



# Check it worked correctly

df_all.head(5)
# Using descriptions in 'About this file' we can order some categoric variables

df_all = df_all.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},

                       "BsmtCond" : {"No basement" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "BsmtExposure" : {"No basement" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},

                       "BsmtFinType1" : {"No basement" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtFinType2" : {"No basement" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtQual" : {"No basement" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 

                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},

                       "GarageCond" : {"No garage" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "GarageQual" : {"No garage" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

                       "Street" : {"Grvl" : 1, "Pave" : 2},

                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}

                     )
# Checking for collinearity between similar variables

basement_feats = ['BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual']

exterior_feats = ['ExterCond','ExterQual']

garage_feats = ['GarageCond','GarageQual']





plt.figure(figsize=[20,5])



# basement_feats plot

plt.subplot(1,3,1)

sns.heatmap(df_all[basement_feats].corr(), vmin = -1, vmax=1, annot=True, cmap="coolwarm")

plt.title('basement_feats')



# exterior_feats plot

plt.subplot(1,3,2)

sns.heatmap(df_all[exterior_feats].corr(), vmin = -1, vmax=1, annot=True, cmap="coolwarm")

plt.title('exterior_feats')



# garage_feats plot

plt.subplot(1,3,3)

sns.heatmap(df_all[garage_feats].corr(), vmin = -1, vmax=1, annot=True, cmap="coolwarm")

plt.title('garage_feats')
# Although 'BsmtQual' and 'ExterQual' are highly correlated, they should be independant of each other so will both stay.



# 'GarageCond'and 'GarageQual' are highly correlated. Check what's more correlated to SalePrice

train_temp = pd.concat([df_all[:train.shape[0]],y],axis=1)

train_temp[basement_feats+garage_feats + ['SalePrice']].corr()
# We'll keep 'BsmtQual' as it's more related to 'SalePrice'

df_all.drop('BsmtCond',axis=1,inplace=True)



# We'll keep 'GarageQual' as it's more related to 'SalePrice'

df_all.drop('GarageCond',axis=1,inplace=True)
# No. of categoric variables

cat_feats = df_all.dtypes[df_all.dtypes == "object"].index.tolist()

print(str(len(cat_feats)) + ' categoric features')

print("")

for i in cat_feats:

    print(i)

    print(df_all[i].value_counts(dropna=False))

    print("")
# No. of categoric variables

cat_feats = df_all.dtypes[df_all.dtypes == "object"].index.tolist()

print(str(len(cat_feats)) + ' categoric features')

df_all = pd.get_dummies(df_all)

df_all.head(5)
null_columns=df_all.columns[df_all.isnull().any()].tolist()

df_all[null_columns].describe()
df_all[null_columns].isnull().sum()
# Some (to me) make sense to likely be 0

to_set_to_zero = ['MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageCars','GarageArea']

df_all[to_set_to_zero] = df_all[to_set_to_zero].fillna(0)



# Drop GarageYrBlt as unlikely to be able to fill years, correlated to YearBuilt & has a value 2207 so question marks on it's validity...

df_all.drop('GarageYrBlt',axis=1,inplace = True)



# Replace the rest with the median of the column:

df_all = df_all.fillna(df_all.median())
# Using original list of numerical features (with 'GarageYrBuilt' removed), check the skewness

num_feats.remove('GarageYrBlt')
# Calculate skewness

skewed_feats = df_all[num_feats].apply(lambda x: skew(x)) 

skewed_feats.sort_values()
# Return columns with high skewness

skewed_feats = skewed_feats[skewed_feats > 0.75].index

skewed_feats
# Annoyingly, Box-Cox is breaking for 3 features due to a precision issue (I think, see https://github.com/scipy/scipy/issues/7534)

# Therefore need to remove these 3 from list to transform

skewed_feats=skewed_feats.tolist()

skewed_feats = [e for e in skewed_feats if e not in ['GrLivArea','LotArea','1stFlrSF']]
# Transform numerical variables through the box-cox method (optimises transformation to a gaussian distribution)

# Requires a +1 as inputs need to be strictly positive

pt = PowerTransformer('yeo-johnson',standardize=False)

print(pt.fit(df_all[skewed_feats])) 

print("")



# Show lambdas to see what transformation was applied

print(pt.lambdas_)
# Insert these back into the dataframe

df_all[skewed_feats] = pt.transform(df_all[skewed_feats])



# Log the failed features

df_all[['GrLivArea','LotArea','1stFlrSF']]=df_all[['GrLivArea','LotArea','1stFlrSF']].apply(np.log)



# Readd to list

skewed_feats = skewed_feats + ['GrLivArea','LotArea','1stFlrSF']



# Check the new skews (still could be improved, but a lot better than before!)

df_all[skewed_feats].skew().sort_values()
train = df_all[:train.shape[0]]

test = df_all[train.shape[0]:]



# Check they are the same shape as started. They are, great.

print(train.shape)

print(test.shape)
# Correlations with SalePrice (look to make sense)

pd.concat([train,y],axis=1).corr().iloc[:,-1].sort_values(ascending=False).head(10)
# Look for outliers

plt.figure(figsize=[20,5])



# 'OverallQual' plot

plt.subplot(1,2,1)

sns.scatterplot(x = train['OverallQual'], y = y)

plt.title('OverallQual')

plt.ylabel('SalePrice')

plt.xlabel('OverallQual')



# 'GrLivArea' plot

plt.subplot(1,2,2)

sns.scatterplot(x = train['GrLivArea'], y = y)

plt.title('GrLivArea')

plt.ylabel('SalePrice')

plt.xlabel('GrLivArea')

plt.grid(b=bool, which='major', axis='both')
# Will remove two values that don't match distribution on 'GrLivArea'. Turns out this is also the two outliers on OverallQual = 10 (see below)

index_to_drop = train[(train['GrLivArea']>8.3) & (y<12.5)].index.tolist()

# Remove from training feature set

train = train.drop(index_to_drop,axis=0)

# Remove from training observation set

y = y.drop(index_to_drop)
# As above, checking they're gone

plt.figure(figsize=[20,5])



# 'OverallQual' plot

plt.subplot(1,2,1)

sns.scatterplot(x = train['OverallQual'], y = y)

plt.title('OverallQual')

plt.ylabel('SalePrice')

plt.xlabel('OverallQual')



# 'GrLivArea' plot

plt.subplot(1,2,2)

sns.scatterplot(x = train['GrLivArea'], y = y)

plt.title('GrLivArea')

plt.ylabel('SalePrice')

plt.xlabel('GrLivArea')

plt.grid(b=bool, which='major', axis='both')
# Check still the same

print(train.shape)

print(y.shape)
# Define error measure for official scoring : RMSE

scorer = make_scorer(mean_squared_error, greater_is_better = False)



def rmse_cv_train(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))

    return(rmse)



def rmse_cv_test(model):

    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))

    return(rmse)          