# Section for all imports

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
# Section for defining functions

def scatterPlot(df, x, y):

    plt.scatter(df[x], df[y], c = "blue", marker = "s")

    plt.xlabel(x)

    plt.ylabel(y)

    plt.show()

    

    

def findMissingValues(df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing_data_stats = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data_stats



def plotCorrMatrixAndReturnColumns(df, topN): 

    corrmat = df.corr()

    f, ax = plt.subplots(figsize=(12, 9))

    highcorrelation_cols = corrmat.nlargest(topN, 'SalePrice')['SalePrice'].index

    cm = np.corrcoef(df[highcorrelation_cols].values.T)

    sns.set(font_scale=1.25)

    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=highcorrelation_cols.values, xticklabels=highcorrelation_cols.values)

    plt.show()

    return highcorrelation_cols



def computeAccuracy(scores):

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    

def buildModelAndFit(clf, train, train_target):

    # Compute accuracy

    clf_score = cross_val_score(clf, train, train_target, cv=5)

    computeAccuracy(clf_score)

    # Fit the model

    return clf.fit(train, train_target)
# Load the train, test data

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# Intuition tells us that the SalePrice should increase with GrLivArea

# Let's plot this to see if this intuition is true in our dataset

scatterPlot(df_train, "GrLivArea", "SalePrice")
# The above plot overall agrees with our intuition, there are few rows that don't agree, let's eliminate them.

# Eliminiating Outliers from the dataset

df_train = df_train[df_train.GrLivArea < 4500]

# Plotting again - Looks much better!

scatterPlot(df_train, "GrLivArea", "SalePrice")
# Combine train and test so we can apply data clean up, feature engineering.

df_all = pd.concat([df_train, df_test], keys=['train', 'test'])
missing_data_stats = findMissingValues(df_all).drop("SalePrice")

# Let's look at the columns that 

print("Total number of columns with missing values : " + str(len(missing_data_stats[missing_data_stats.Total > 0])))

missing_data_stats[missing_data_stats.Total > 0]
# Handling features with fewer missing values

# You can see that BsmtFullBath and BsmtHalfBath when TotalBsmtSF is 0.0

df_all[pd.isnull(df_all['BsmtFullBath'])][["BsmtFullBath", "BsmtHalfBath", "TotalBsmtSF"]]



df_all.loc[:, "BsmtFullBath"] = df_all.loc[:, "BsmtFullBath"].fillna(0.0)

df_all.loc[:, "BsmtHalfBath"] = df_all.loc[:, "BsmtHalfBath"].fillna(0.0)



# Same logic for "BsmtUnfSF", "BsmtFinSF2", "BsmtFinSF1"

df_all[pd.isnull(df_all['BsmtUnfSF'])][["TotalBsmtSF","BsmtUnfSF", "BsmtFinSF2", "BsmtFinSF1"]]

df_all.loc[:, "BsmtUnfSF"] = df_all.loc[:, "BsmtUnfSF"].fillna(0.0)

df_all.loc[:, "BsmtFinSF2"] = df_all.loc[:, "BsmtFinSF2"].fillna(0.0)

df_all.loc[:, "BsmtFinSF1"] = df_all.loc[:, "BsmtFinSF1"].fillna(0.0)





# Handling missing value in GarageCars

#  GarageCars	1	0.000343

df_all[pd.isnull(df_all['GarageCars'])][["GarageCars","GarageArea"]]



# GarageArea is 0, so GarageCars -> 0.0

df_all.loc[:, "GarageCars"] = df_all.loc[:, "GarageCars"].fillna(0.0)
# Handling missing data in PoolQC: 

#   PoolQC	2906	0.997255



# Number of house records with no pool (pool area = 0) = 2903

print("Number of house records with no pool (pool area = 0) = " + str(len(df_all[df_all.PoolArea == 0])))



# This almost accounts for the number of records with a missing PoolQC.

#   Here are the possible values for PoolQC from data_description.txt : 

#   	 Ex	Excellent

#   	 Gd	Good

#   	 TA	Average/Typical

#   	 Fa	Fair

#   	 NA	No Pool



# When you plot the distribution of PoolQC, you will see that very few items have this.

df_all.PoolQC.value_counts()

#  Gd    3

#  Ex    3

#  Fa    2



# So, we can set these missing values should be set to "NA"

df_all.loc[:, "PoolQC"] = df_all.loc[:, "PoolQC"].fillna("NA")



# Create a new feature HasPool

df_all.ix[df_all[(df_all['PoolArea'] != 0) ].index, 'HasPool'] = 1

df_all.ix[df_all[(df_all['PoolArea'] == 0) ].index, 'HasPool'] = 0
# Handling missing data in FireplaceQu: 

# 	 FireplaceQu	1420	0.487303



print("\nNumber of house records with no fireplace (Fireplaces = 0) = " + str(len(df_all[df_all.Fireplaces == 0])))

# Number of house records with no fireplace (Fireplaces = 0) = 1420



# Number of entries with missing FireplaceQu match the number of entries where there aren't any Fireplaces!





# Here are the possible values for FireplaceQu from data_description.txt : 

# 	 Ex   Excellent - Exceptional Masonry Fireplace

# 	 Gd   Good - Masonry Fireplace in main level

# 	 TA   Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement

# 	 Fa   Fair - Prefabricated Fireplace in basement

# 	 Po   Poor - Ben Franklin Stove

# 	 NA   No Fireplace



# So, these missing values should be set to "NA"

df_all.loc[:, "FireplaceQu"] = df_all.loc[:, "FireplaceQu"].fillna("NA")
# Handling missing data in MiscFeature: 

#   MiscFeature	2809	0.963967



# Here are the possible values for MiscFeature from data_description.txt : 

# 	 Elev Elevator

# 	 Gar2 2nd Garage (if not described in garage section)

# 	 Othr Other

# 	 Shed Shed (over 100 SF)

# 	 TenC Tennis Court

# 	 NA   None



# So, these missing values should be set to "NA"

df_all.loc[:, "MiscFeature"] = df_all.loc[:, "MiscFeature"].fillna("NA")
# 	 Alley	2716	0.932052

# Here are the possible values for Alley from data_description.txt : 

# 	 Grvl Gravel

# 	 Pave Paved

# 	 NA   No alley access



#  So, these missing values should be set to "NA"

df_all.loc[:, "Alley"] = df_all.loc[:, "Alley"].fillna("NA")



# TODO: Check if adding a boolean HasAlley (derived from Alley might be useful)
# Handling missing data in Fence: 

#  Fence	2344	0.804393

# Here are the possible values for Fence from data_description.txt : 

# 	 GdPrv    Good Privacy

# 	 MnPrv    Minimum Privacy

# 	 GdWo     Good Wood

# 	 MnWw     Minimum Wood/Wire

# 	 NA   No Fence



# So, these missing values should be set to "NA"

df_all.loc[:, "Fence"] = df_all.loc[:, "Fence"].fillna("NA")
# Handling missing data in GarageFinish, GarageCond, GarageQual, GarageYrBlt, GarageType

#   GarageFinish	159	0.054564

#   GarageCond	159	0.054564

#   GarageQual	159	0.054564

#   GarageYrBlt	159	0.054564

#   GarageType	157	0.053878



# One thing that's obvious from these fields is that they are all related to "Garage" 



# Let's do a quick sanity check to see how many records happen have no garage.

print("\nNumber of house records without a garage (GarageArea = 0) = " + str(len(df_all[df_all.GarageArea == 0])))

# Number of house records with no garage (GarageArea = 0) = 157



# Of the other 2, GarageArea for one of the houses is null, we can set that to 0.

df_all.loc[:, "GarageArea"] = df_all.loc[:, "GarageArea"].fillna(0.0)



# So, all these missing Garage* fields should be set to "NA" - No Garage.

df_all.loc[:, "GarageCond"] = df_all.loc[:, "GarageCond"].fillna("NA")

df_all.loc[:, "GarageType"] = df_all.loc[:, "GarageType"].fillna("NA")

df_all.loc[:, "GarageFinish"] = df_all.loc[:, "GarageFinish"].fillna("NA")

df_all.loc[:, "GarageQual"] = df_all.loc[:, "GarageQual"].fillna("NA")



# Set to 0, may be set this to the year the house was built.

df_all.loc[:, "GarageYrBlt"] = df_all.loc[:, "GarageYrBlt"].fillna(0.0)
# Handling missing data in BsmtExposure, BsmtFinType2, BsmtFinType1, BsmtCond, BsmtQual

#   BsmtCond	82	0.028140

#   BsmtExposure	82	0.028140

#   BsmtQual	81	0.027797

#   BsmtFinType2	80	0.027454

#   BsmtFinType1	79	0.027111



# We know that there is one house with TotalBsmtSF as null. Let's fix that right away.

df_all.loc[:, "TotalBsmtSF"] = df_all.loc[:, "TotalBsmtSF"].fillna(0.0)



# Of all of the above rows with missing Bsmt* features, about 79 have these fields missing because they don't have any basement (TotalBsmtSF ==0)

len(df_all[(df_all['TotalBsmtSF'] == 0) & (((pd.isnull(df_all['BsmtCond'])) | (pd.isnull(df_all['BsmtExposure'])) | (pd.isnull(df_all['BsmtQual'])) | (pd.isnull(df_all['BsmtFinType1'])) | (pd.isnull(df_all['BsmtFinType2']))))][['TotalBsmtSF', 'BsmtUnfSF', 'BsmtExposure','BsmtFinType2', 'BsmtFinType1','BsmtCond','BsmtQual']])
# The remaining is about 9 rows.

df_all[(df_all['TotalBsmtSF'] != 0) & (((pd.isnull(df_all['BsmtCond'])) | (pd.isnull(df_all['BsmtExposure'])) | (pd.isnull(df_all['BsmtQual'])) | (pd.isnull(df_all['BsmtFinType1'])) | (pd.isnull(df_all['BsmtFinType2']))))][['TotalBsmtSF', 'BsmtUnfSF', 'BsmtExposure','BsmtFinType2', 'BsmtFinType1','BsmtCond','BsmtQual']]
df_all[(df_all['TotalBsmtSF'] != 0) & (pd.isnull(df_all['BsmtCond']))][['TotalBsmtSF', 'BsmtUnfSF', 'BsmtExposure','BsmtFinType2', 'BsmtFinType1','BsmtCond','BsmtQual']]

df_all.BsmtCond.value_counts()

# Most common BsmtCond is "TA"

df_all.ix[df_all[(df_all['TotalBsmtSF'] != 0) & (pd.isnull(df_all['BsmtCond']))].index, 'BsmtCond'] = 'TA'
# We notice that when BsmtExposure is null, and TotalBsmtSF > 0, the rows have the same value for BsmtFinType2	BsmtFinType1	BsmtCond	BsmtQual

# Let's focus on this subset.

df_all[(df_all['TotalBsmtSF'] != 0) & (pd.notnull(df_all['BsmtExposure'])) & (df_all.BsmtQual == 'Gd') & (df_all.BsmtCond == 'TA') & (df_all.BsmtFinType1 == 'Unf') & (df_all.BsmtFinType2 == 'Unf') ].BsmtExposure.value_counts()



# This shows the most common BsmtExposure is "No"

df_all.ix[df_all[(df_all['TotalBsmtSF'] != 0) & (pd.isnull(df_all['BsmtExposure']))].index, 'BsmtExposure'] = 'No'
df_all[(df_all['TotalBsmtSF'] != 0) & (df_all['TotalBsmtSF'] == df_all['BsmtUnfSF'])].BsmtQual.value_counts()



# This shows the most common BsmtExposure is "TA"

df_all.ix[df_all[(df_all['TotalBsmtSF'] != 0) & (pd.isnull(df_all['BsmtQual']))].index, 'BsmtQual'] = 'TA'
df_all[(df_all['TotalBsmtSF'] != 0)].BsmtFinType2.value_counts()



# This shows the most common BsmtFinType2 is "Unf"

df_all.ix[df_all[(df_all['TotalBsmtSF'] != 0) & (pd.isnull(df_all['BsmtFinType2']))].index, 'BsmtFinType2'] = 'Unf'
# Handling the other 79 Basement related rows with missing values

# Setting BsmtExposure, BsmtFinType2, BsmtFinType1, BsmtCond and BsmtQual to "NA".

df_all.loc[:, "BsmtExposure"] = df_all.loc[:, "BsmtExposure"].fillna("NA")

df_all.loc[:, "BsmtFinType2"] = df_all.loc[:, "BsmtFinType2"].fillna("NA")

df_all.loc[:, "BsmtFinType1"] = df_all.loc[:, "BsmtFinType1"].fillna("NA")

df_all.loc[:, "BsmtCond"] = df_all.loc[:, "BsmtCond"].fillna("NA")

df_all.loc[:, "BsmtQual"] = df_all.loc[:, "BsmtQual"].fillna("NA")
# Handling missing value in Electrical

# Electrical	1	0.000343

df_all.Electrical.value_counts()



# We can see that "SBrkr" is the most common value for Electrical, let's use that for the missing value.

df_all.loc[:, "Electrical"] = df_all.loc[:, "Electrical"].fillna("SBrkr")



# Utilities	2	0.000686

# AllPub is the most common one.

df_all.loc[:, "Utilities"] = df_all.loc[:, "Utilities"].fillna("AllPub")



# Functional	2	0.000686

# Typ is the most commone one.

df_all.loc[:, "Functional"] = df_all.loc[:, "Functional"].fillna("Typ")



# MSZoning	4	0.001373

df_all[pd.isnull(df_all['MSZoning'])]



# df_all.MSZoning.value_counts() shows that "RL" is the most common value for MSZoning

df_all.loc[:, "MSZoning"] = df_all.loc[:, "MSZoning"].fillna("RL")



# Looking at missing value in Exterior1st and Exterior2nd, they are in the same row.

#    Exterior1st	1	0.000343

#    Exterior2nd	1	0.000343

df_all[pd.isnull(df_all['Exterior1st'])][["Exterior1st", "Exterior2nd", "ExterQual", "ExterCond"]]

# From looking at the most common, "HdBoard" is the winner.

df_all[(df_all['ExterQual'] == "TA") & (df_all['ExterCond'] == "TA")].Exterior1st.value_counts()

df_all[(df_all['ExterQual'] == "TA") & (df_all['ExterCond'] == "TA")].Exterior2nd.value_counts()



df_all.loc[:, "Exterior1st"] = df_all.loc[:, "Exterior1st"].fillna("HdBoard")

df_all.loc[:, "Exterior2nd"] = df_all.loc[:, "Exterior2nd"].fillna("HdBoard")



# KitchenQual	1	0.000343

df_all[pd.isnull(df_all['KitchenQual'])][["KitchenAbvGr"]]

df_all[(df_all['KitchenAbvGr'] == 1)].KitchenQual.value_counts()

# Most common KitchenQual is TA

df_all.loc[:, "KitchenQual"] = df_all.loc[:, "KitchenQual"].fillna("TA")



# SaleType	1	0.000343

df_all[pd.isnull(df_all['SaleType'])][["SaleType", "SaleCondition"]]

df_all[(df_all['SaleCondition'] == "Normal")].SaleType.value_counts()

# Common value is WD

df_all.loc[:, "SaleType"] = df_all.loc[:, "SaleType"].fillna("WD")
# Handling missing values in MasVnrType and MasVnrArea

#  MasVnrType	24	0.008236

#  MasVnrArea	23	0.007893



# We can notice that the missing fields are in the same rows. 

df_all[['MasVnrType','MasVnrArea']][df_all['MasVnrType'].isnull()==True]



# Find the distribution of MasVnrType

df_all.MasVnrType.value_counts()



# You can see that None is the most common value, let's use that.

df_all.loc[:, "MasVnrType"] = df_all.loc[:, "MasVnrType"].fillna("None")



# Find the distribution of MasVnrArea

df_all.MasVnrArea.value_counts()



# You can see that 0.0 is the most common value, let's use that.

df_all.loc[:, "MasVnrArea"] = df_all.loc[:, "MasVnrArea"].fillna(0.0)
# Handling missing data in LotFrontage: 

# 	 LotFrontage	486	0.166781



# From data_descriptions.txt we have LotFrontage: Linear feet of street connected to property 



df_all[(pd.isnull(df_all['LotFrontage']))][["Street", "LotFrontage", "GrLivArea", "LotArea", "LotShape", "LotConfig"]]



# Let's look at the correlation between LotFrontage and LotArea 

df_all['LotFrontage'].corr(df_all['LotArea'])

# 0.46601522801969764 - doesn't look too promising.



# Let's compute the Sqrt of LotArea and find it's correlation with LotFrontage

df_all['SqrtLotArea']=np.sqrt(df_all['LotArea'])

df_all['LotFrontage'].corr(df_all['SqrtLotArea'])

# 0.63639701180660091 - looks much better, let's use that for filling the null entries.



df_all.LotFrontage.fillna(df_all.SqrtLotArea, inplace=True)

del df_all['SqrtLotArea']
highcorrelation_cols = plotCorrMatrixAndReturnColumns(df_all, 30)
# Looking at the correlation matrix, it's evident that the columns with high % of missing values aren't in the top 37 features.

lowCorrMissingFeaturesToDrop=["PoolQC",	"MiscFeature", "Alley", "Fence"]



for x in lowCorrMissingFeaturesToDrop:

    print(x in highcorrelation_cols)

    

df_all.drop(lowCorrMissingFeaturesToDrop, axis=1, inplace=True)
# Other features we can drop after checking the correlation matrix. 

# 'GarageYrBlt',

otherLowCorrelationNumericFeaturesToConsider = ['LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'MiscVal']

for x in otherLowCorrelationNumericFeaturesToConsider: 

    print(x + "\'s correlation with SalePrice : " + str(df_all['SalePrice'].corr(df_all[x])))

    scatterPlot(df_all, 'SalePrice', x)

    

# As you can see these features are mostly having a value of 0 and have very low correlation

df_all.drop(otherLowCorrelationNumericFeaturesToConsider, axis=1, inplace=True)
df_all["MSSubClass"] = df_all["MSSubClass"].astype(str)

df_all["OverallCond"] = df_all["OverallCond"].astype(str)



df_all["GarageYrBlt"] = df_all["GarageYrBlt"].astype(str)

df_all["YrSold"] = df_all["YrSold"].astype(str)

df_all["MoSold"] = df_all["MoSold"].astype(str)
# The following "Categorical" features are ordinal features. 

# Let's convert them into ordered numbers so there is some order information in them.



# BsmtCond

df_all = df_all.replace({"BsmtCond" : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})



# BsmtExposure

df_all = df_all.replace({"BsmtExposure" : {"NA" : 0, "No" : 1, "Mn" : 2, "Av": 3, "Gd" : 4}})



# BsmtFinType1 and BsmtFinType2

df_all = df_all.replace({"BsmtFinType1" : {"NA" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6}})

df_all = df_all.replace({"BsmtFinType2" : {"NA" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6}})



# BsmtQual

df_all = df_all.replace({"BsmtQual" : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5}})



# ExterCond and ExterQual

df_all = df_all.replace({"ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5}})

df_all = df_all.replace({"ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5}}) 



# FireplaceQu

df_all = df_all.replace({"FireplaceQu" : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})

# Functional

df_all = df_all.replace({"Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8}})

# GarageCond

df_all = df_all.replace({"GarageCond" : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})

# GarageQual

df_all = df_all.replace({"GarageQual" : {"NA" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})

# HeatingQC

df_all = df_all.replace({"HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})

# KitchenQual

df_all = df_all.replace({"KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})

# Utilities

df_all = df_all.replace({"Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}})
# Combine existing features to create new ones.

df_all["TotalNumOfBathsAbvGrade"] = df_all["FullBath"] + (0.5 * df_all["HalfBath"])

df_all["TotalAreaSf"] = df_all["GrLivArea"] + df_all["TotalBsmtSF"]

df_all["TotalFlrSf"] = df_all["1stFlrSF"] + df_all["2ndFlrSF"]

df_all["TotalPorchSF"] = df_all["OpenPorchSF"] + df_all["EnclosedPorch"] + df_all["3SsnPorch"] + df_all["ScreenPorch"]
plotCorrMatrixAndReturnColumns(df_all, 30)
categorical_features = df_all.select_dtypes(include = ["object"]).columns

numerical_features = df_all.select_dtypes(exclude = ["object"]).columns

numerical_features = numerical_features.drop("SalePrice").drop("Id")

df_all_num = df_all[numerical_features]

df_all_cat = df_all[categorical_features]

print("Total number of features : " + str(len(df_all.columns)))

print("\nNumber of Numerical features : " + str(len(numerical_features)))

print(numerical_features)



print("\nNumber of non-numeric features : " + str(len(categorical_features)))

print(categorical_features)
from scipy.stats import skew

skewness = df_all_num.apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.5]

print(str(skewness.shape[0]) + " skewed numerical features to log transform")



skewed_features = skewness.index

df_all_num[skewed_features] = np.log1p(df_all_num[skewed_features])
train_target = df_all.pop("SalePrice").xs('train')

# Let's look at the distribution of SalePrice. We can see that it's skewed to the right.

sns.distplot(train_target);

# Applying log transformation usually helps with the skew.

train_target = np.log1p(train_target)



# Let's plot this again to check -> Looks much better now.

sns.distplot(train_target);
# Convert cat features using dummies

df_all_cat = pd.get_dummies(df_all_cat)



# Scale the numeric features

numericScaler = StandardScaler()

df_all_num.loc[:, numerical_features] = numericScaler.fit_transform(df_all_num.loc[:, numerical_features])



df_all = pd.concat([df_all_num, df_all_cat], axis = 1)



# Seperate the train and test data

train = df_all.xs('train')

test = df_all.xs('test')
gbr_100 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')

gbr_100 = buildModelAndFit(gbr_100, train, train_target)

gbr_100_labels = np.exp(gbr_100.predict(test))
gbr_3000 = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',

                                               min_samples_leaf=15, min_samples_split=10, loss='huber')

gbr_3000 = buildModelAndFit(gbr_3000, train, train_target)

gbr_3000_labels = np.exp(gbr_3000.predict(test))



predictions = pd.DataFrame({'Id': df_test.Id, 'SalePrice': gbr_3000_labels})

predictions.to_csv('attempt3_gbm3000.csv', index =False)
gbr_400 = GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,

          learning_rate = 0.1, loss = 'ls')

gbr_400 = buildModelAndFit(gbr_400, train, train_target)

gbr_400_labels = np.exp(gbr_400.predict(test))



predictions = pd.DataFrame({'Id': df_test.Id, 'SalePrice': gbr_400_labels})

predictions.to_csv('attempt4_gbm400.csv', index =False)
gbr_3000_v2 = GradientBoostingRegressor(n_estimators = 3000, max_depth = 5, min_samples_split = 2,

          learning_rate = 0.1, loss = 'ls')

gbr_3000_v2 = buildModelAndFit(gbr_3000_v2, train, train_target)

gbr_3000_v2_labels = np.exp(gbr_3000_v2.predict(test))



predictions = pd.DataFrame({'Id': df_test.Id, 'SalePrice': gbr_3000_v2_labels})

predictions.to_csv('attempt4_gbm3000_v2.csv', index =False)
gbr_3000_v3 = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=5, max_features='auto',

                                               min_samples_leaf=15, min_samples_split=10, loss='huber')

gbr_3000_v3 = buildModelAndFit(gbr_3000_v3, train, train_target)

gbr_3000_v3_labels = np.exp(gbr_3000_v3.predict(test))



predictions = pd.DataFrame({'Id': df_test.Id, 'SalePrice': gbr_3000_v3_labels})

predictions.to_csv('attempt3_gbm3000_v3.csv', index =False)
from sklearn.ensemble import RandomForestRegressor



rmfClf = RandomForestRegressor()

rmfClf = buildModelAndFit(rmfClf, train, train_target)

rmfClf_labels = np.exp(rmfClf.predict(test))



predictions = pd.DataFrame({'Id': df_test.Id, 'SalePrice': rmfClf_labels})

predictions.to_csv('attempt5_rmfClf.csv', index =False)

rmfClf_3000 = RandomForestRegressor(n_estimators=3000, max_depth=3, max_features='sqrt',

                                               min_samples_leaf=15, min_samples_split=10)



rmfClf_3000 = buildModelAndFit(rmfClf_3000, train, train_target)

rmfClf_3000_labels = np.exp(rmfClf_3000.predict(test))



predictions = pd.DataFrame({'Id': df_test.Id, 'SalePrice': rmfClf_3000_labels})

predictions.to_csv('attempt5_rmfClf_3000.csv', index =False)