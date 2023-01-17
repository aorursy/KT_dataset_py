# I started my work by taking tips from https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

# Importing libraries

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.model_selection import KFold, cross_val_score

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings('ignore')
# bring the numbers in

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

print('Size of train data set is :',df_train.shape)

print('Size of test data set is :',df_test.shape)
#Saving Ids

train_ID = df_train['Id']

test_ID = df_test['Id']



#Dropping Ids

df_train.drop("Id", axis = 1, inplace = True)

df_test.drop("Id", axis = 1, inplace = True)
fig, ax = plt.subplots()

ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'], c = "skyblue")

plt.ylabel('SalePrice', fontsize=6)

plt.xlabel('GrLivArea', fontsize=6)

plt.show()



df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000)].index)



fig, ax = plt.subplots()

ax.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'], c = "skyblue")

plt.ylabel('SalePrice', fontsize=6)

plt.xlabel('GrLivArea', fontsize=6)

plt.show()
#Correlation matrix

corrmat = df_train.corr()

mask = np.zeros_like(corrmat)

mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, mask=mask, linewidths=.5, vmax=0.7, square=True, cmap="YlGnBu")
#stat summary

df_train['SalePrice'].describe()



#get distribution & QQ Plot

sns.distplot(df_train['SalePrice'], 

             kde_kws={"color": "coral", "lw": 1, "label": "KDE"}, 

             hist_kws={"histtype": "stepfilled", "linewidth": 3, "alpha": 0.8, "color": "skyblue"});





fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()
df_train['SalePrice_Log'] = np.log(df_train['SalePrice'])



sns.distplot(df_train['SalePrice_Log'], 

             kde_kws={"color": "coral", "lw": 1, "label": "KDE"}, 

             hist_kws={"histtype": "stepfilled", "linewidth": 3, "alpha": 1, "color": "skyblue"});

 

# skewness and kurtosis

print("Skewness: %f" % df_train['SalePrice_Log'].skew())

print("Kurtosis: %f" % df_train['SalePrice_Log'].kurt())



fig = plt.figure()

res = stats.probplot(df_train['SalePrice_Log'], plot=plt)

plt.show()



# dropping SalePrice

df_train.drop('SalePrice', axis= 1, inplace=True)
#Saving sets sizes, concat sets and dropping target variable

size_df_train = df_train.shape[0]

size_df_test = df_test.shape[0]

target_variable = df_train.SalePrice_Log.values

data = pd.concat((df_train, df_test)).reset_index(drop=True)

data.drop(['SalePrice_Log'], axis=1, inplace=True)



# Lets check if the ammount of null values in data

data.count().sort_values()
features_fill_na_none = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu',

               'GarageQual','GarageCond','GarageFinish','GarageType',

               'BsmtExposure','BsmtCond','BsmtQual','BsmtFinType1','BsmtFinType2',

               'MasVnrType']



for feature_none in features_fill_na_none:

    data[feature_none].fillna('None',inplace=True)

    

features_fill_na_0 = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea',

                      'BsmtFullBath','BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 

                      'BsmtUnfSF', 'TotalBsmtSF']



for feature_0 in features_fill_na_0:

    data[feature_0].fillna('None',inplace=True)



#LotFrontage

#We'll fill missing values by the median of observation's Neighborhood

data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



#MSzoning - 4 missing values

#We'll fill missing values with most common value

data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])



# Utilities

# All records have "AllPub", but 3. From those 3, 2 are NA and one is "NoSeWa" is 

# in the training set.

# We may proceed to drop this column

data = data.drop(columns=['Utilities'],axis=1)



#Functional: NA means "Typ"

data["Functional"] = data["Functional"].fillna("Typ")



# Electrical - 91% of observations have Electrical = SBrkr

# We'll fill missing values with SBrkr

data['Electrical'] = data['Electrical'].fillna("SBrkr")



#Exterior1st and Exterior2nd, one missing value, same observation

#We'll substitute it with the most common value

data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])

data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])



#KitchenQual, ony one missing value

#We'll subsitute it with the most common value

data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])



#Saletype, ony one missing value

#We'll subsitute it with the most common value

data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
data.count().sort_values()
#getting numerical features and categorical features

numerical_features = data.dtypes[data.dtypes != "object"].index

categorical_features = data.dtypes[data.dtypes == "object"].index



print("We have: ", len(numerical_features), 'Numerical Features')

print("We have: ", len(categorical_features), 'Categorical Features')



numerical_features

categorical_features
features_to_transform = ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']



for feature in features_to_transform:

    data[feature] = data[feature].apply(str)



#Let's check how our features stand now

numerical_features = data.dtypes[data.dtypes != "object"].index

categorical_features = data.dtypes[data.dtypes == "object"].index



print("We have: ", len(numerical_features), 'Numerical Features')

print("We have: ", len(categorical_features), 'Categorical Features')



numerical_features

categorical_features



#Let's encode categorical variables

encode_cat_variables = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



for variable in encode_cat_variables:

    lbl = LabelEncoder() 

    lbl.fit(list(data[variable].values)) 

    data[variable] = lbl.transform(list(data[variable].values))
#Boxplot for numerical_features

sns.set_style("whitegrid")

f, ax = plt.subplots(figsize=(20, 20))

ax.set_xscale("log")

ax = sns.boxplot(data=data[numerical_features] , orient="h", palette="ch:2.5,-.2,dark=.5")

ax.set(ylabel="Features")

ax.set(xlabel="Value")

ax.set(title="Distribution")

sns.despine(trim=True, left=True)
skewed_features = data[numerical_features].apply(lambda x: skew(x)).sort_values(ascending=False)



norm_target_features = skewed_features[skewed_features > 0.5]

norm_target_index = norm_target_features.index

print("#{} numerical features need normalization; :".format(norm_target_features.shape[0]))

skewness = pd.DataFrame({'Skew' :norm_target_features})

norm_target_features

#Normalizing with Box Cox Transformation

for i in norm_target_index:

    data[i] = boxcox1p(data[i], boxcox_normmax(data[i] + 1))



#Let's look how the transformed features are standing now

sns.set_style("whitegrid")

f, ax = plt.subplots(figsize=(15, 15))

ax.set_xscale("log")

ax = sns.boxplot(data=data[norm_target_index] , orient="h", palette="ch:2.5,-.2,dark=.3")

ax.set(ylabel="Features")

ax.set(xlabel="Value")

ax.set(title="Distribution")

sns.despine(trim=True, left=True)
#back to train and test sets

df_train = data[:size_df_train]

df_test = data[size_df_train:]
#Cross Validation

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(df_train.values)

    rmse= np.sqrt(-cross_val_score(model, df_train.values, target_variable, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
