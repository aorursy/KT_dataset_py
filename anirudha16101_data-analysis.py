# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 8.0)

import seaborn as sns

from scipy import stats

from scipy.stats import norm
import pandas as pd

train = pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

train.head()
print(train.shape)

print(test.shape)
train.info()
train.isnull().any().sum()
train.columns[train.isnull().any()]
missing=train.isnull().sum()/len(train)

missing=missing[missing>0]

missing.sort_values(inplace=True)

missing
#visualising missing values

missing=pd.DataFrame(missing)

missing.columns=['count']

missing.index.names=['name']

missing['name'] = missing.index

sns.barplot(x='name',y='count',data=missing)

plt.xticks(rotation=90)

#sales_plot

sns.distplot(train['SalePrice'])
print ("The skewness of SalePrice is {}".format(train['SalePrice'].skew()))
#Take log tranform to remove skewness 

target=np.log(train['SalePrice'])

print(target.skew())

sns.distplot(target)
#separate variables into new data frames

numeric_data=train.select_dtypes(include=[np.number])

cat_data=train.select_dtypes(exclude=[np.number])





numeric_data.head()

numeric_data.drop('Id',axis=1,inplace=True)

cat_data.head()
plt.figure(figsize=(20,20))

corr=numeric_data.corr()

sns.heatmap(corr,annot=True)
print(corr['SalePrice'].sort_values(ascending=False))
print(train.OverallQual.unique())

pivot = train.pivot_table(index='OverallQual', values='SalePrice',aggfunc='median')

pivot.plot(kind='bar')

sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'])

sns.jointplot(x=train['OverallQual'], y=train['SalePrice'])
train.GarageCars.unique()

pivot = train.pivot_table(index='GarageCars', values='SalePrice',aggfunc='median')

pivot.plot(kind='bar')
cat_data.describe()
cat = [f for f in train.columns if train.dtypes[f] == 'object']

def anova(frame):

    anv = pd.DataFrame()

    anv['features'] = cat

    pvals = []

    for c in cat:

           samples = []

           for cls in frame[c].unique():

                  s = frame[frame[c] == cls]['SalePrice'].values

                  samples.append(s)

           pval = stats.f_oneway(*samples)[1]

           pvals.append(pval)

    anv['pval'] = pvals

    return anv.sort_values('pval')



cat_data['SalePrice'] = train.SalePrice.values

k = anova(cat_data) 

k['disparity'] = np.log(1./k['pval'].values) 

sns.barplot(data=k, x = 'features', y='disparity') 

plt.xticks(rotation=90) 

plt 
num = [f for f in train.columns if train.dtypes[f] != 'object']

num.remove('Id')

nd = pd.melt(train, value_vars = num)

n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)

n1 = n1.map(sns.distplot, 'value')

n1
train.drop(train[train['GrLivArea']>4000].index,inplace=True)

train.shape
test.loc[666, 'GarageQual'] = "TA" #stats.mode(test['GarageQual']).mode

test.loc[666, 'GarageCond'] = "TA" #stats.mode(test['GarageCond']).mode

test.loc[666, 'GarageFinish'] = "Unf" #stats.mode(test['GarageFinish']).mode

test.loc[666, 'GarageYrBlt'] = "1980" #np.nanmedian(test['GarageYrBlt'])` 

test.loc[1116, 'GarageType'] = np.nan
#importing function

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def factorize(data, var, fill_na = None):

      if fill_na is not None:

            data[var].fillna(fill_na, inplace=True)

      le.fit(data[var])

      data[var] = le.transform(data[var])

      return data
#combine the data set

alldata = train.append(test)

alldata.shape

#impute lotfrontage by median of neighborhood

lot_frontage_by_neighborhood = train['LotFrontage'].groupby(train['Neighborhood'])



for key, group in lot_frontage_by_neighborhood:

                idx = (alldata['Neighborhood'] == key) & (alldata['LotFrontage'].isnull())

                alldata.loc[idx, 'LotFrontage'] = group.median()
#imputing missing values

alldata["MasVnrArea"].fillna(0, inplace=True)

alldata["BsmtFinSF1"].fillna(0, inplace=True)

alldata["BsmtFinSF2"].fillna(0, inplace=True)

alldata["BsmtUnfSF"].fillna(0, inplace=True)

alldata["TotalBsmtSF"].fillna(0, inplace=True)

alldata["GarageArea"].fillna(0, inplace=True)

alldata["BsmtFullBath"].fillna(0, inplace=True)

alldata["BsmtHalfBath"].fillna(0, inplace=True)

alldata["GarageCars"].fillna(0, inplace=True)

alldata["GarageYrBlt"].fillna(0.0, inplace=True)

alldata["PoolArea"].fillna(0, inplace=True)
qual_dict = {np.nan: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

name = np.array(['ExterQual','PoolQC' ,'ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu', 'GarageQual','GarageCond'])



for i in name:

     alldata[i] = alldata[i].map(qual_dict).astype(int)



alldata["BsmtExposure"] = alldata["BsmtExposure"].map({np.nan: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)



bsmt_fin_dict = {np.nan: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}

alldata["BsmtFinType1"] = alldata["BsmtFinType1"].map(bsmt_fin_dict).astype(int)

alldata["BsmtFinType2"] = alldata["BsmtFinType2"].map(bsmt_fin_dict).astype(int)

alldata["Functional"] = alldata["Functional"].map({np.nan: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)



alldata["GarageFinish"] = alldata["GarageFinish"].map({np.nan: 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)

alldata["Fence"] = alldata["Fence"].map({np.nan: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)



#encoding data

alldata["CentralAir"] = (alldata["CentralAir"] == "Y") * 1.0

varst = np.array(['MSSubClass','LotConfig','Neighborhood','Condition1','BldgType','HouseStyle','RoofStyle','Foundation','SaleCondition'])



for x in varst:

         factorize(alldata, x)



#encode variables and impute missing values

alldata = factorize(alldata, "MSZoning", "RL")

alldata = factorize(alldata, "Exterior1st", "Other")

alldata = factorize(alldata, "Exterior2nd", "Other")

alldata = factorize(alldata, "MasVnrType", "None")

alldata = factorize(alldata, "SaleType", "Oth")

#creating new variable (1 or 0) based on irregular count levels

#The level with highest count is kept as 1 and rest as 0

alldata["IsRegularLotShape"] = (alldata["LotShape"] == "Reg") * 1

alldata["IsLandLevel"] = (alldata["LandContour"] == "Lvl") * 1

alldata["IsLandSlopeGentle"] = (alldata["LandSlope"] == "Gtl") * 1

alldata["IsElectricalSBrkr"] = (alldata["Electrical"] == "SBrkr") * 1

alldata["IsGarageDetached"] = (alldata["GarageType"] == "Detchd") * 1

alldata["IsPavedDrive"] = (alldata["PavedDrive"] == "Y") * 1

alldata["HasShed"] = (alldata["MiscFeature"] == "Shed") * 1

alldata["Remodeled"] = (alldata["YearRemodAdd"] != alldata["YearBuilt"]) * 1



#Did the modeling happen during the sale year?

alldata["RecentRemodel"] = (alldata["YearRemodAdd"] == alldata["YrSold"]) * 1



# Was this house sold in the year it was built?

alldata["VeryNewHouse"] = (alldata["YearBuilt"] == alldata["YrSold"]) * 1

alldata["Has2ndFloor"] = (alldata["2ndFlrSF"] == 0) * 1

alldata["HasMasVnr"] = (alldata["MasVnrArea"] == 0) * 1

alldata["HasWoodDeck"] = (alldata["WoodDeckSF"] == 0) * 1

alldata["HasOpenPorch"] = (alldata["OpenPorchSF"] == 0) * 1

alldata["HasEnclosedPorch"] = (alldata["EnclosedPorch"] == 0) * 1

alldata["Has3SsnPorch"] = (alldata["3SsnPorch"] == 0) * 1

alldata["HasScreenPorch"] = (alldata["ScreenPorch"] == 0) * 1



#setting levels with high count as 1 and the rest as 0

#you can check for them using the value_counts function

alldata["HighSeason"] = alldata["MoSold"].replace({1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})

alldata["NewerDwelling"] = alldata["MSSubClass"].replace({20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})
alldata.shape

#create alldata2

alldata2 = train.append(test)



alldata["SaleCondition_PriceDown"] = alldata2.SaleCondition.replace({'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})



# house completed before sale or not

alldata["BoughtOffPlan"] = alldata2.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})

alldata["BadHeating"] = alldata2.HeatingQC.replace({'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1, 'Po': 1})
#calculating total area using all area columns

area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF','OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea' ]



alldata["TotalArea"] = alldata[area_cols].sum(axis=1)

alldata["TotalArea1st2nd"] = alldata["1stFlrSF"] + alldata["2ndFlrSF"]

alldata["Age"] = 2010 - alldata["YearBuilt"]

alldata["TimeSinceSold"] = 2010 - alldata["YrSold"]

alldata["SeasonSold"] = alldata["MoSold"].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3}).astype(int)

alldata["YearsSinceRemodel"] = alldata["YrSold"] - alldata["YearRemodAdd"]



# Simplifications of existing features into bad/average/good based on counts

alldata["SimplOverallQual"] = alldata.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})

alldata["SimplOverallCond"] = alldata.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})

alldata["SimplPoolQC"] = alldata.PoolQC.replace({1 : 1, 2 : 1, 3 : 2, 4 : 2})

alldata["SimplGarageCond"] = alldata.GarageCond.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

alldata["SimplGarageQual"] = alldata.GarageQual.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

alldata["SimplFireplaceQu"] = alldata.FireplaceQu.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

alldata["SimplFireplaceQu"] = alldata.FireplaceQu.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

alldata["SimplFunctional"] = alldata.Functional.replace({1 : 1, 2 : 1, 3 : 2, 4 : 2, 5 : 3, 6 : 3, 7 : 3, 8 : 4})

alldata["SimplKitchenQual"] = alldata.KitchenQual.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

alldata["SimplHeatingQC"] = alldata.HeatingQC.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

alldata["SimplBsmtFinType1"] = alldata.BsmtFinType1.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})

alldata["SimplBsmtFinType2"] = alldata.BsmtFinType2.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2})

alldata["SimplBsmtCond"] = alldata.BsmtCond.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

alldata["SimplBsmtQual"] = alldata.BsmtQual.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

alldata["SimplExterCond"] = alldata.ExterCond.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})

alldata["SimplExterQual"] = alldata.ExterQual.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2})



#grouping neighborhood variable based on this plot

train['SalePrice'].groupby(train['Neighborhood']).median().sort_values().plot(kind='bar')