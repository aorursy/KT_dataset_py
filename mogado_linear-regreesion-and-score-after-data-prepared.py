import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from scipy import stats

import warnings

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')









#**1-Identification of variables and data types





X_train = df_train.shape[0]

X_test = df_test.shape[0]

y_train = df_train.SalePrice.values

Alldata = pd.concat((df_train, df_test)).reset_index(drop=True)

Alldata.drop(['SalePrice'], axis=1, inplace=True)

print(Alldata.shape)
#Save the 'Id' column

X_train_ID = df_train['Id']

X_test_ID = df_test['Id']

#9-Correlation Analysis

corrmat=df_train.corr()

plt.show(sns.heatmap(corrmat,vmax=.8,square=True))
k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice').index

cm=np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

plt.show(sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values))





sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

plt.show(sns.pairplot(df_train[cols],size=1.5))

#Graphical Univariate Analysis*

var='OverallQual'

data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)

f,ax=plt.subplots(figsize=(8,6))

plt.show(sns.boxplot(x=var,y="SalePrice",data=data))
var='GrLivArea'

data=pd.concat([df_train['SalePrice'],df_train[var]],axis=1)

plt.scatter(x=df_train[var],y=df_train['SalePrice'])

plt.xlabel(var,fontsize=20)

plt.ylabel('SalePrice',fontsize=20)

#plt.show()
#Missing value treatment



Alldata_na = (Alldata.isnull().sum() / len(Alldata)) * 100

Alldata_na = Alldata_na.drop(Alldata_na[Alldata_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :Alldata_na})

print(missing_data.head(20))
f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='90')

sns.barplot(x=Alldata_na.index, y=Alldata_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
##Data Description says NA which means No Pool, No Misc Feature etc etc etc.



Alldata["PoolQC"] = Alldata["PoolQC"].fillna("None")

Alldata["MiscFeature"] = Alldata["MiscFeature"].fillna("None")

Alldata["MiscFeature"] = Alldata["MiscFeature"].fillna("None")

Alldata["Alley"] = Alldata["Alley"].fillna("None")

Alldata["Fence"] = Alldata["Fence"].fillna("None")

Alldata["FireplaceQu"] = Alldata["FireplaceQu"].fillna("None")
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

Alldata["LotFrontage"] = Alldata.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    Alldata[col] = Alldata[col].fillna('None')



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    Alldata[col] = Alldata[col].fillna(0)



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    Alldata[col] = Alldata[col].fillna(0)



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    Alldata[col] = Alldata[col].fillna('None')

Alldata["MasVnrType"] = Alldata["MasVnrType"].fillna("None")

Alldata["MasVnrArea"] = Alldata["MasVnrArea"].fillna(0)

Alldata['MSZoning'] = Alldata['MSZoning'].fillna(Alldata['MSZoning'].mode()[0])

Alldata = Alldata.drop(columns=['Utilities'], axis=1)

Alldata["Functional"] = Alldata["Functional"].fillna("Typ")

Alldata['Electrical'] = Alldata['Electrical'].fillna(Alldata['Electrical'].mode()[0])

Alldata['KitchenQual'] = Alldata['KitchenQual'].fillna(Alldata['KitchenQual'].mode()[0])

Alldata['Exterior1st'] = Alldata['Exterior1st'].fillna(Alldata['Exterior1st'].mode()[0])

Alldata['Exterior2nd'] = Alldata['Exterior2nd'].fillna(Alldata['Exterior2nd'].mode()[0])

Alldata['SaleType'] = Alldata['SaleType'].fillna(Alldata['SaleType'].mode()[0])

Alldata['MSSubClass'] = Alldata['MSSubClass'].fillna("None")

Alldata['MSSubClass'] = Alldata['MSSubClass'].fillna("None")

#Check remaining missing values if any

Alldata_na = (Alldata.isnull().sum() / len(Alldata)) * 100

Alldata_na = Alldata_na.drop(Alldata_na[Alldata_na == 0].index).sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Ratio' :Alldata_na})

missing_data.head()
#transformin some numerical value that are categoricaL

#MSSubClass=The building class

Alldata['MSSubClass'] = Alldata['MSSubClass'].apply(str)

print(Alldata['MSSubClass'])
#Changing OverallCond into a categorical variable

Alldata['OverallCond'] = Alldata['OverallCond'].astype(str)
#Year and month sold are transformed into categorical features.

Alldata['YrSold'] = Alldata['YrSold'].astype(str)

Alldata['MoSold'] = Alldata['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',

        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features

for c in cols:

    lbl = LabelEncoder()

    lbl.fit(list(Alldata[c].values))

    Alldata[c] = lbl.transform(list(Alldata[c].values))



# shape

print('Shape Alldata: {}'.format(Alldata.shape))
# Adding total sqfootage feature

Alldata['TotalSF'] = Alldata['TotalBsmtSF'] + Alldata['1stFlrSF'] + Alldata['2ndFlrSF']
from scipy import stats

from scipy.stats import norm, skew #for some statistics



numeric_feats = Alldata.dtypes[Alldata.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = Alldata[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)





skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
from scipy.special import boxcox1p



skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    # all_data[feat] += 1

    Alldata[feat] = boxcox1p(Alldata[feat], lam)

Alldata = pd.get_dummies(Alldata)

print(Alldata.shape)
train_data=Alldata[:X_train]

test_data=Alldata[X_train:]
from sklearn.linear_model import LinearRegression

HousepriceNLR= LinearRegression()

HousepriceNLR.fit(train_data, y_train)



y_predNLR = HousepriceNLR.predict(test_data)



print(HousepriceNLR.score(train_data, y_train))

print(HousepriceNLR.score(test_data, y_predNLR))
sub = pd.DataFrame()

sub['Id'] = X_test_ID

sub['SalePrice'] = y_predNLR

print(sub)

sub.to_csv('submission.csv',index=False)
