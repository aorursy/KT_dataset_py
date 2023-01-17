import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling 

import math

from pandas.api.types import CategoricalDtype

%matplotlib inline

# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_x = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_data=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
print (train_data.shape)

print (test_x.shape)
train_data.head()

train_x = train_data.loc[:,train_data.columns != 'SalePrice']

train_y = train_data.loc[:,train_data.columns == 'SalePrice']

print (train_x.shape)

print (train_y.shape)
cptry = train_data.loc[:,train_data.columns == 'SalePrice']
train_y.head()
# applying log transform to train, test data.

class Utils:

    @classmethod

    def apply_log_trans(cls, indf, collist):

        temp = indf.copy()

        df = pd.DataFrame(np.log(temp[collist]))

        temp = temp.drop(columns=collist)

        outdf = pd.concat([temp,df],join="inner",axis=1)

        return outdf

    

    @classmethod

    def corr_two_feature(cls, indf, f1, f2):

        val = indf[[f1,f2]].corr()[f1][f2]

        return val

    

    @classmethod

    def corr_one_feature(cls, indf, f1):

        train_corr = pd.DataFrame(indf[indf.columns[1:]].corr()[f1][:])

        train_corr =train_corr.sort_values(by=[f1],ascending=False)

        return train_corr

    

    @classmethod

    def corr_similar_features(cls, indf, f1):

        tmp_list = [cols for cols in indf.columns if f1 in cols]

        temp_corr = pd.DataFrame(indf[tmp_list].corr())

        return temp_corr

    

    @classmethod

    def cat_myrename(cls, indf, fture, catlist):

        tmp = indf[fture].astype("category")

        gdict = { val:cnt+1 for cnt,val in enumerate(catlist) }

        tmp = pd.DataFrame(tmp.cat.rename_categories(gdict))

        return tmp

    

    @classmethod

    def missing_vals(cls, indf, id_str=None):

        if (id_str is None):

            id_str = 'Id'

        countdf = indf.count()

        missdict = {}

        for key,val in countdf.items():

            missdict[key] = countdf[id_str] - val

        missdf = pd.DataFrame(missdict.items(),columns=['name','miss_val'])

        miss_pct = pd.DataFrame((missdf['miss_val']/countdf[id_str])*100)

        miss_pct = miss_pct.rename(columns={'miss_val':'miss_pct'})

        missdf = pd.concat([missdf,miss_pct],axis=1,join='inner')

        missdf = missdf.sort_values(by='miss_pct',ascending=False)

        return missdf

    

    @classmethod

    def missing_vals2(cls, indf, id_str=None):

        all_data = indf

        all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

        all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

        missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

        return missing_data

    

    

    @classmethod

    def my_dummies(cls, indf, collist):

        dummy = pd.get_dummies(indf[collist])

        tmp = indf.copy()

        tmp = pd.concat([tmp, dummy],axis=1,join='inner')

        tmp.drop(columns=collist,axis=1,inplace=True)

        return tmp

    

    @classmethod

    def cal_skewness(cls, indf):

        all_data = indf.copy()

        numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

        # Check the skew of all numerical features

        skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

        print("\nSkew in numerical features: \n")

        skewness = pd.DataFrame({'Skew' :skewed_feats})

        return skewness
# first understand the target

sns.distplot(train_y)

plt.show()
# train_y = Utils.apply_log_trans(train_y, ['SalePrice'])

# train_y
# # understand feature vectors. First, Profile

# import warnings

# warnings.filterwarnings('ignore')

# profile = pandas_profiling.ProfileReport(train_x)

# profile
sns.catplot(x="MSSubClass",y="SalePrice",data=train_data, kind="bar")
# train_data_corr = train_data.corr()

# plt.subplots(figsize=(22,9))

# sns.heatmap(train_data_corr,cmap='coolwarm')
# correlation between target and some features.

train_corr = pd.DataFrame(train_data[train_data.columns[1:]].corr()['SalePrice'][:])

train_corr =train_corr.sort_values(by=["SalePrice"],ascending=False)

train_corr
# N largest correlated features

#saleprice correlation matrix

k = 10 #number of variables for heatmap

# cols = train_data_corr.nlargest(k, 'SalePrice')['SalePrice'].index

# cm = np.corrcoef(train_data[cols].values.T)

# sns.set(font_scale=1.25)

# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

# plt.show()
# tmp_corr = pd.DataFrame(train_data[train_data.columns[1:]].corr()['OverallQual'][:])

# tmp_corr = tmp_corr.sort_values(by=["OverallQual"],ascending=False)

# tmp_corr
sns.relplot(x="OverallQual",y="SalePrice",data=train_data)
# above correlation is a famous cone graph where relationship is non-linear. it is bad
df_y = pd.DataFrame(np.log(train_data['SalePrice']))

new_df = pd.concat([train_data["OverallQual"],df_y],axis=1, join="inner")

sns.relplot(x="OverallQual",y="SalePrice",data=new_df)
# above correlation graph is a linear graph. so, lets apply log trans on target

df = pd.DataFrame(np.log(train_data['SalePrice']))

train_data = train_data.drop(columns=["SalePrice"])

train_data = pd.concat([train_data,df],join="inner",axis=1)

train_data
sns.relplot(x="GrLivArea",y="SalePrice",data=train_data)
df = pd.DataFrame(np.log(train_data['GrLivArea']))

tmp_df = pd.concat([train_data['SalePrice'],df],join="inner",axis=1)

sns.relplot(x="GrLivArea",y="SalePrice",data=tmp_df)
collist = ['TotalBsmtSF','GarageArea','GrLivArea']

train_x = Utils.apply_log_trans(train_x,collist)

test_x = Utils.apply_log_trans(test_x,collist)

train_x = train_x.drop(columns=["GarageArea","1stFlrSF","TotRmsAbvGrd"])

test_x = test_x.drop(columns=["GarageArea","1stFlrSF","TotRmsAbvGrd"])
sns.relplot(x="GarageCars",y="GarageArea",data=train_data)


sns.relplot(x="TotalBsmtSF",y="SalePrice",data=train_data)
df = pd.DataFrame(np.log(train_data['TotalBsmtSF']))

tmp_df = pd.concat([train_data['SalePrice'],df],join="inner",axis=1)

sns.relplot(x="TotalBsmtSF",y="SalePrice",data=tmp_df)
tmp=pd.concat([train_x,train_y],join="inner",axis=1)

sns.catplot(x="FullBath",y="SalePrice",data=tmp)


tmp=pd.concat([train_x,train_y],join="inner",axis=1)

# sns.barplot(x="TotRmsAbvGrd",y="SalePrice",data=tmp)

# sns.catplot(x="TotRmsAbvGrd",y="SalePrice",data=tmp)
# sns.relplot(x="TotRmsAbvGrd",y="SalePrice",data=tmp)
# tmp=pd.concat([train_x,train_y],join="inner",axis=1)

# sns.barplot(x="YearBuilt",y="SalePrice",data=tmp)
sns.relplot(x="YearBuilt",y="SalePrice",data=tmp)
# train_corr = pd.DataFrame(train_data[train_data.columns[1:]].corr()['SalePrice'][:])

train_x[['YearBuilt','YearRemodAdd']].corr()['YearBuilt']['YearRemodAdd']
train_x.columns
# sns.relplot(x="GarageYrBlt",y="YearBuilt",data=tmp)

Utils.corr_two_feature(tmp, "GarageYrBlt","YearBuilt")
train_x = train_x.drop(columns=["YearBuilt"])

test_x = test_x.drop(columns=['YearBuilt'])
# check correlation between garagexxx features.

desired_cols = [ col for col in tmp.columns if "Garage" in col]

print (desired_cols)

corr_ = Utils.corr_similar_features(tmp, "Garage")

print (corr_)

sns.heatmap(corr_, cmap="coolwarm",annot=True)
%matplotlib inline

plt.figure(figsize=(16, 6))

sns.catplot(x="GarageType",y="SalePrice",data=tmp,kind="swarm")

plt.show()


# print (train_x['GarageType'].unique())

# garagelist = ['Attchd', 'BuiltIn','Detchd', 'Basment','CarPort' , '2Types']

# garagelist.reverse()

# # tmp_x = train_x["GarageType"].astype("category",ordered=True,categories=garagelist).cat.codes

# tmp_x = train_x["GarageType"].astype("category")

# cat_type=CategoricalDtype(categories=garagelist,ordered=True)

# tmp_x = tmp_x.astype(cat_type)

# tmp_x.cat.codes
glist = ['Attchd', 'BuiltIn','Detchd', 'Basment','CarPort' , '2Types']

glist.reverse()

tmp = Utils.cat_myrename(train_x, "GarageType", glist)

train_x = train_x.drop(columns=["GarageType"])

train_x = pd.concat([train_x, tmp],join="inner",axis=1)

tmp = Utils.cat_myrename(test_x, "GarageType", glist)

test_x = test_x.drop(columns=["GarageType"])

test_x = pd.concat([test_x, tmp],join="inner",axis=1)
test_x["GarageType"]
# %matplotlib inline

# plt.figure(figsize=(16, 6))

# sns.catplot(x="GarageFinish",y="SalePrice",data=tmp,kind="swarm")

# plt.show()

train_x["GarageFinish"]
gf_order = ['Unf','RFn','Fin']

tmp = Utils.cat_myrename(train_x, "GarageFinish", gf_order)

train_x = train_x.drop(columns=["GarageFinish"])

train_x = pd.concat([train_x, tmp],join="inner",axis=1)

tmp = Utils.cat_myrename(test_x, "GarageFinish", gf_order)

test_x = test_x.drop(columns=["GarageFinish"])

test_x = pd.concat([test_x, tmp],join="inner",axis=1)
#checking the correlation of cars and target

tmp = pd.concat([train_x,train_y],axis=1,join="inner")

val = Utils.corr_two_feature(tmp,"GarageCars","SalePrice")

val
train_x["GarageQual"]

sns.catplot(x="GarageQual",y="SalePrice",data=tmp,kind="swarm")

plt.show()
qual_order = ['Po','Fa','TA','Gd','Ex']

tmp = Utils.cat_myrename(train_x, "GarageQual", qual_order)

train_x = train_x.drop(columns=["GarageQual"])

train_x = pd.concat([train_x, tmp],join="inner",axis=1)

tmp = Utils.cat_myrename(test_x, "GarageQual", qual_order)

test_x = test_x.drop(columns=["GarageQual"])

test_x = pd.concat([test_x, tmp],join="inner",axis=1)
qual_order = ['Po','Fa','TA','Gd','Ex']

tmp = Utils.cat_myrename(train_x, "GarageCond", qual_order)

train_x = train_x.drop(columns=["GarageCond"])

train_x = pd.concat([train_x, tmp],join="inner",axis=1)

tmp = Utils.cat_myrename(test_x, "GarageCond", qual_order)

test_x = test_x.drop(columns=["GarageCond"])

test_x = pd.concat([test_x, tmp],join="inner",axis=1)
tot_tmp = pd.concat([train_x,train_y],axis=1,join="inner")
tmp = pd.concat([train_x,train_y],axis=1,join="inner")

sns.relplot(x="GarageCond",y="GarageQual",data=tmp)

# val = Utils.corr_two_feature(tmp,"GarageCond","GarageQual")

# val
train_x.drop(["GarageCond"],axis=1,inplace=True)

test_x.drop(["GarageCond"],axis=1,inplace=True)
# sns.relplot(x="OverallQual",y="OverallCond",data=tmp)

val = Utils.corr_two_feature(tmp, "OverallQual","OverallCond")

print(val)

sns.lineplot(data=tmp, x="OverallQual",y="OverallCond")
desired_cols = [ col for col in tmp.columns if "Bsmt" in col]

print(desired_cols)

dcorr = train_x[desired_cols].corr()

sns.heatmap(dcorr,cmap="coolwarm",annot=True)
train_x[['BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF']].describe()
train_x = Utils.apply_log_trans(train_x,['BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF'])

test_x = Utils.apply_log_trans(test_x,['BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF'])
train_x[['BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF']].describe()
adf = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',]

train_x[adf]
train_x['BsmtFinType1'].unique()
# edit 1: 

# we could map as follows. No need to use cat_myrename func

# qual_dict = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

# all_df["ExterQual"] = df["ExterQual"].map(qual_dict).astype(int)



qual_order = ['Po','Fa','TA','Gd','Ex']

colname = "BsmtQual"

tmp = Utils.cat_myrename(train_x, colname, qual_order)

train_x = train_x.drop(columns=[colname])

train_x = pd.concat([train_x, tmp],join="inner",axis=1)

tmp = Utils.cat_myrename(test_x, colname, qual_order)

test_x = test_x.drop(columns=[colname])

test_x = pd.concat([test_x, tmp],join="inner",axis=1)



qual_order = ['Po','Fa','TA','Gd','Ex']

colname = "BsmtCond"

tmp = Utils.cat_myrename(train_x, colname, qual_order)

train_x = train_x.drop(columns=[colname])

train_x = pd.concat([train_x, tmp],join="inner",axis=1)

tmp = Utils.cat_myrename(test_x, colname, qual_order)

test_x = test_x.drop(columns=[colname])

test_x = pd.concat([test_x, tmp],join="inner",axis=1)



qual_order = ['No','Mn','Av','Gd']

colname = "BsmtExposure"

tmp = Utils.cat_myrename(train_x, colname, qual_order)

train_x = train_x.drop(columns=[colname])

train_x = pd.concat([train_x, tmp],join="inner",axis=1)

tmp = Utils.cat_myrename(test_x, colname, qual_order)

test_x = test_x.drop(columns=[colname])

test_x = pd.concat([test_x, tmp],join="inner",axis=1)





qual_order = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf']

qual_order.reverse()

colname = "BsmtFinType1"

tmp = Utils.cat_myrename(train_x, colname, qual_order)

train_x = train_x.drop(columns=[colname])

train_x = pd.concat([train_x, tmp],join="inner",axis=1)

tmp = Utils.cat_myrename(test_x, colname, qual_order)

test_x = test_x.drop(columns=[colname])

test_x = pd.concat([test_x, tmp],join="inner",axis=1)



colname = "BsmtFinType2"

tmp = Utils.cat_myrename(train_x, colname, qual_order)

train_x = train_x.drop(columns=[colname])

train_x = pd.concat([train_x, tmp],join="inner",axis=1)

tmp = Utils.cat_myrename(test_x, colname, qual_order)

test_x = test_x.drop(columns=[colname])

test_x = pd.concat([test_x, tmp],join="inner",axis=1)
adf = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',]

train_x[adf]
# first get all the categorical features.

all_cols = train_x.columns

num_cols = train_x._get_numeric_data().columns

print (list(set(all_cols)-set(num_cols)))
ln = len(list(set(all_cols)-set(num_cols)))

ln
outdf = Utils.missing_vals(train_x,'Id')

outdf.head(8)
outdf = Utils.missing_vals(test_x,'Id')

outdf.head(8)
train_x.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1,inplace=True)

test_x.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'],axis=1,inplace=True)
# lets convert all the categorical vectors to numerical

sns.catplot(x='LotShape',y='SalePrice',data=tot_tmp,kind='box')
# get dummies for LotShape

train_x = Utils.my_dummies(train_x, ["LotShape"])

test_x = Utils.my_dummies(test_x, ["LotShape"])
sns.catplot(x="LotConfig",data=train_x,kind="count")
sns.catplot(x="LotConfig",y='SalePrice',data=tot_tmp)
# get dummies for LotShape

train_x = Utils.my_dummies(train_x, ["LotConfig"])

test_x = Utils.my_dummies(test_x, ["LotConfig"])
# and get dummies for all other categorical vectors

catvec = ['Condition1', 'CentralAir', 'ExterQual', 'Foundation', 'Electrical', 'GarageFinish', 'Street', 'BldgType', 'RoofStyle', 'LandSlope', 'BsmtExposure', 'KitchenQual', 'Utilities', 'RoofMatl', 'SaleCondition', 'MSZoning', 'GarageType', 'BsmtFinType2', 'Exterior2nd', 'BsmtCond', 'BsmtQual', 'Functional', 'SaleType', 'HouseStyle', 'Exterior1st', 'HeatingQC', 'Neighborhood', 'Condition2', 'ExterCond', 'MasVnrType', 'PavedDrive', 'BsmtFinType1', 'LandContour', 'GarageQual', 'Heating']

train_x = Utils.my_dummies(train_x,catvec)

test_x = Utils.my_dummies(test_x,catvec)
num_cols
val = Utils.corr_one_feature(tot_tmp,"MSSubClass")

val
train_x.drop(['MSSubClass'],axis=1,inplace=True)

test_x.drop(['MSSubClass'],axis=1,inplace=True)
sns.relplot(x="LotFrontage",y="SalePrice",data=tot_tmp)
# correlation between big and small values. so, lets apply log transform to lotfrontage

train_x = Utils.apply_log_trans(train_x,['LotFrontage'])

test_x = Utils.apply_log_trans(test_x,['LotFrontage'])
tot_tmp = pd.concat([train_x,train_y],axis=1,join="inner")

sns.relplot(x="LotFrontage",y="SalePrice",data=tot_tmp)
# remove the outliers i.e LotFrontage>5.5

train_x.drop(train_x[train_x['LotFrontage']>5.5].index,inplace=True)
tot_tmp = pd.concat([train_x,train_y],axis=1,join="inner")

sns.relplot(x="LotFrontage",y="SalePrice",data=tot_tmp)
train_x[['LotArea','OverallQual','OverallCond', 'YearRemodAdd', 'MasVnrArea', '2ndFlrSF', 'LowQualFinSF']]

# drop LowQualFinSF as 98% are zeros

# retaining OverallQuall, OverallCond as it is and applying log transformation on LotArea, MasVnrArea, 2ndFlrSF

train_x = Utils.apply_log_trans(train_x, ['LotArea','MasVnrArea','2ndFlrSF'])

test_x = Utils.apply_log_trans(test_x, ['LotArea','MasVnrArea','2ndFlrSF'])
tmp=pd.concat([train_x,train_y],join="inner",axis=1)

sns.catplot(x="YearRemodAdd",y="SalePrice",data=tmp,kind="box")
train_x[['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr','KitchenAbvGr', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'WoodDeckSF']]
tmp = train_x.copy()

df = pd.DataFrame(tmp['BsmtFullBath']+tmp['BsmtHalfBath']+tmp['FullBath']+tmp['HalfBath'])

trdf = df.rename(columns={0:"TotalBath"})

tmp = test_x.copy()

df = pd.DataFrame(tmp['BsmtFullBath']+tmp['BsmtHalfBath']+tmp['FullBath']+tmp['HalfBath'])

tsdf = df.rename(columns={0:"TotalBath"})

train_x = pd.concat([train_x,trdf],axis=1,join='inner')

test_x = pd.concat([test_x,tsdf],axis=1,join='inner')
train_x.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],axis=1,inplace=True)

test_x.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],axis=1,inplace=True)
sns.relplot(x="WoodDeckSF",y='SalePrice',data=tot_tmp)
train_x = Utils.apply_log_trans(train_x, ['WoodDeckSF'])

test_x = Utils.apply_log_trans(test_x, ['WoodDeckSF'])
tot_tmp = pd.concat([train_x,train_y],axis=1,join='inner')

sns.relplot(x="WoodDeckSF",y='SalePrice',data=tot_tmp)
train_x[['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea','MiscVal', 'MoSold', 'YrSold', 'TotalBsmtSF', 'GrLivArea', 'BsmtFinSF1',]]
df = train_x.copy()

totdf = pd.DataFrame(df['OpenPorchSF']+df['EnclosedPorch']+df['3SsnPorch']+df['ScreenPorch'])

totdf.rename(columns={0:"Tot_porchSF"},inplace=True)

train_x = pd.concat([df,totdf],axis=1,join='inner')

train_x.drop(['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'],axis=1,inplace=True)

df = test_x.copy()

totdf = pd.DataFrame(df['OpenPorchSF']+df['EnclosedPorch']+df['3SsnPorch']+df['ScreenPorch'])

totdf.rename(columns={0:"Tot_porchSF"},inplace=True)

test_x = pd.concat([df,totdf],axis=1,join='inner')

test_x.drop(['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'],axis=1,inplace=True)
# sns.relplot(x="PoolArea",y="SalePrice",kind="line",data=tot_tmp)

test_x[test_x['PoolArea']==0]["PoolArea"].count()
train_x.drop(['PoolArea'],axis=1,inplace=True)

test_x.drop(['PoolArea'],axis=1,inplace=True)
print(train_x[train_x['MiscVal']==0]["MiscVal"].count()/1460)

# sns.relplot(x="MiscVal",y="SalePrice",kind="line",data=tot_tmp)
train_x.drop(['MiscVal'],axis=1,inplace=True)

test_x.drop(['MiscVal'],axis=1,inplace=True)
# lets check MoSold, YrSold correlation

sns.catplot(x="MoSold",y='SalePrice',data=tot_tmp,kind='box')
sns.catplot(x="YrSold",y='SalePrice',data=tot_tmp,kind='box')
train_x['GarageYrBlt'].unique()
test_x[test_x['GarageYrBlt'].isnull()]['GarageYrBlt']
test_data[test_data['GarageYrBlt'].isnull()]['GarageYrBlt']
val = Utils.missing_vals(train_x,"Id")

val.head(10)
tmp = train_data[['Neighborhood','YearBuilt','GarageYrBlt']]

tmp2 = tmp[(tmp['YearBuilt'].notnull()) & (tmp['GarageYrBlt'].isnull())]

tmp2
for ind, row in tmp2.iterrows():

    train_x.at[ind,'GarageYrBlt'] = row['YearBuilt']
train_x[['Id','GarageYrBlt']].count()
tmp = test_data[['Neighborhood','YearBuilt','GarageYrBlt']]

tmp3 = tmp[(tmp['YearBuilt'].notnull()) & (tmp['GarageYrBlt'].isnull())]

tmp3
for ind, row in tmp3.iterrows():

    test_x.at[ind,'GarageYrBlt'] = row['YearBuilt']
test_x[test_x['GarageYrBlt'].isnull()]['GarageYrBlt']
tmp2 = pd.DataFrame(train_data.groupby('Neighborhood').agg({'LotFrontage':'mean'}))
tmp2['LotFrontage']['Blmngtn']
cptr = train_x

for ind, row in cptr.iterrows():

    if (math.isnan(cptr.at[ind,'LotFrontage'])):

        val = train_data.at[ind,'Id']

        val2 = pd.DataFrame(train_data.loc[train_data['Id']==val]['Neighborhood'])

        cptr.at[ind,'LotFrontage'] = tmp2['LotFrontage'][val2.iloc[0]['Neighborhood']]
train_x['LotFrontage'].count()
cptr = test_x

tmp2 = pd.DataFrame(test_data.groupby('Neighborhood').agg({'LotFrontage':'mean'}))

for ind, row in cptr.iterrows():

    if (math.isnan(cptr.at[ind,'LotFrontage'])):

        val = test_data.at[ind,'Id']

        val2 = pd.DataFrame(test_data.loc[test_data['Id']==val]['Neighborhood'])

        cptr.at[ind,'LotFrontage'] = tmp2['LotFrontage'][val2.iloc[0]['Neighborhood']]
print (test_x['LotFrontage'].count())

print (test_x['Id'].count())
not_inf = train_x[train_x['MasVnrArea']!=np.NINF]

val = not_inf['MasVnrArea'].mean()

train_x.loc[train_x.MasVnrArea==np.NINF,'MasVnrArea'] = val

# train_x['MasVnrArea'].unique()
not_inf = train_x[train_x['MasVnrArea']!=np.NAN]

val = not_inf['MasVnrArea'].mean()

train_x.loc[train_x.MasVnrArea==np.NAN,'MasVnrArea'] = val
# all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

train_x['MasVnrArea'] = train_x['MasVnrArea'].fillna(val)
# use mean of train to fill test data too.

test_x['MasVnrArea'] = test_x['MasVnrArea'].fillna(val)
test_x.loc[test_x.MasVnrArea==np.NINF,'MasVnrArea'] = val
val = Utils.missing_vals(test_x,"Id")

val.head(10)
test_x = test_x.replace({np.NINF:0})

train_x = train_x.replace({np.NINF:0})

test_x['BsmtUnfSF'].mean()
test_x['GarageCars'] = test_x['GarageCars'].fillna(test_x['GarageCars'].mode()[0])

test_x['BsmtUnfSF'] = test_x['BsmtUnfSF'].fillna(test_x['BsmtUnfSF'].mean())

test_x['BsmtFinSF2'] = test_x['BsmtFinSF2'].fillna(test_x['BsmtFinSF2'].mean())

test_x['BsmtFinSF1'] = test_x['BsmtFinSF1'].fillna(test_x['BsmtFinSF1'].mean())

test_x['TotalBsmtSF'] = test_x['TotalBsmtSF'].fillna(test_x['TotalBsmtSF'].mean())

test_x['TotalBath'] = test_x['TotalBath'].fillna(test_x['TotalBath'].mean())
val = Utils.missing_vals(test_x,"Id")

val.head(10)
print (train_x.shape)

print (test_x.shape)
tr_cols = train_x.columns

ts_cols = test_x.columns

diff = [col for col in tr_cols if col not in ts_cols]

diff
# we still need to figure out what should we do with month and year
tmp=pd.concat([train_x,train_y],join="inner",axis=1)

sns.catplot(x="MoSold",y="SalePrice",data=tmp,kind="box")
train_x.drop(['MoSold'],axis=1,inplace=True)

test_x.drop(['MoSold'],axis=1,inplace=True)
# lets see correlation between age of a age house sold and saleprice.

age = pd.DataFrame(train_x['YrSold'] - train_x['GarageYrBlt'])

age.rename({0:"agelog"},inplace=True,axis=1)

age = Utils.apply_log_trans(age,['agelog'])

train_x = pd.concat([train_x,age],axis=1,join='inner')

train_x.drop(['YrSold','GarageYrBlt'],axis=1,inplace=True)
age = pd.DataFrame(test_x['YrSold'] - test_x['GarageYrBlt'])

age.rename({0:"agelog"},inplace=True,axis=1)

age = Utils.apply_log_trans(age,['agelog'])

test_x = pd.concat([test_x,age],axis=1,join='inner')

test_x.drop(['YrSold','GarageYrBlt'],axis=1,inplace=True)
test_x = test_x.replace({np.NINF:0})

train_x = train_x.replace({np.NINF:0})
tmp=pd.concat([train_x,train_y],join="inner",axis=1)

sns.catplot(x="agelog",y="SalePrice",data=tmp)
train_x.describe()
# for col in train_x.columns():

#     print (train_x)

a = train_x.max()

cols = train_x.columns

for i,col in zip(a,cols) :

    print (col,i)
train_x[['Tot_porchSF','YearRemodAdd','LowQualFinSF']]
train_x[train_x['LowQualFinSF']==0]['LowQualFinSF']
train_x.drop('LowQualFinSF',axis=1,inplace=True)

test_x.drop('LowQualFinSF',axis=1,inplace=True)
train_x.drop('YearRemodAdd',axis=1,inplace=True)

test_x.drop('YearRemodAdd',axis=1,inplace=True)
train_x = Utils.apply_log_trans(train_x, ['Tot_porchSF'])

test_x = Utils.apply_log_trans(test_x, ['Tot_porchSF'])
test_x = test_x.replace({np.NINF:0})

train_x = train_x.replace({np.NINF:0})
train_x.drop('Id',axis=1,inplace=True)

test_x.drop('Id',axis=1,inplace=True)
print (train_x.shape)

print (train_y.shape)

print (test_x.shape)
# so drop those 2 rows from train_y

tx_rows = train_x.index

ty_rows = train_y.index

for row in ty_rows:

    if row not in tx_rows:

        print (row)

train_y.drop([934,1298],inplace=True)

print (train_x.shape)

print (train_y.shape)

print (test_x.shape)
types = train_x.dtypes.unique()

print (types)
num_features = train_x.dtypes[train_x.dtypes != "uint8"].index

for val in num_features:

    print(val)
from scipy import stats

from scipy.stats import norm, skew #for some statistics

skewness = Utils.cal_skewness(train_x)

skewness
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0])+" But we are only transforming following columns")

print (num_features)

# but we apply transformation to numerical features only, because they are ordinal values.(above listed features)

# cptr = train_x.copy()

# from scipy.special import boxcox1p

# lam = 0.05

# for feat in num_features:

#     train_x[feat] = boxcox1p(train_x[feat], lam)

#     test_x[feat] = boxcox1p(test_x[feat], lam)

# ylam = 0.001

# train_y = boxcox1p(train_y,ylam)
# boxcox coudn't transform saleprice to 0-3 range instead its capping at 10-15

# so we are using standard scalar to all numerical and target features.

from sklearn.preprocessing import StandardScaler

xscal = StandardScaler()

train_x[num_features] = xscal.fit_transform(train_x[num_features])

test_x[num_features] = xscal.transform(test_x[num_features])



yscal = StandardScaler()

train_y = pd.DataFrame(yscal.fit_transform(train_y))

print (train_x.shape)

print (train_y.shape)

print (test_x.shape)
tr_cols = train_x.columns

ts_cols = test_x.columns

diff = list (set(tr_cols) - set(ts_cols))

for col in diff:

    train_x.drop (col, inplace=True,axis=1)
print (train_x.shape)

print (train_y.shape)

print (test_x.shape)
# from scipy import stats

# from scipy.stats import norm, skew #for some statistics

skewness = Utils.cal_skewness(train_x)

skewness
train_x[['OverallQual', 'OverallCond', 'BedroomAbvGr', 'KitchenAbvGr','Fireplaces', 'GarageCars', 'TotalBsmtSF', 'GrLivArea', 'BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF', 'LotFrontage', 'LotArea', 'MasVnrArea','2ndFlrSF', 'TotalBath', 'WoodDeckSF', 'agelog', 'Tot_porchSF']]
train_y
n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_x.values)

    rmse= np.sqrt(-cross_val_score(model, train_x.values, train_y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
# this is a basic modelling technique.

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb







model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213,

                             random_state =7, nthread = -1)



model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# lets get test results 

model = model_lgb.fit(train_x,train_y)

y_test_pred = model.predict(test_x)

y_test_pred
# lets inverse transform y_test_prid

y_test_pred = yscal.inverse_transform(y_test_pred)
y_test_pred = pd.DataFrame(y_test_pred)
y_test_pred.rename({0:"SalePrice"},axis=1,inplace=True)
y_test_pred
yid = pd.DataFrame(test_data['Id'])

yid
result = pd.concat([yid,y_test_pred],axis=1,join='inner')

result
result.to_csv("with_scalar_trans_all.csv",index=False)
# following are existing methods to do some stuff. I did these in a traditional way.

# indexes = df1.loc[df1.Code.isin(df2.Code.values)].index

# df1.at[indexes,'Value'] = df2['Value'].values



# all_df["SimplOverallQual"] = all_df.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, 4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})



# qual_dict = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

# all_df["ExterQual"] = df["ExterQual"].map(qual_dict).astype(int)


