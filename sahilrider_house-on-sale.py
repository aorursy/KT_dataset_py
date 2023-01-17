#Loading libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0,8.0)



import seaborn as sns

from scipy import stats

from scipy.stats import norm
#Reading files



train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')
train.head()
test.head()
train.shape , test.shape
#checking missing values



train.columns[train.isnull().any()]
miss=train.isnull().sum()/len(train)

miss=miss[miss>0]

miss.sort_values(inplace=True)

miss
#visualing missing values



miss=miss.to_frame()

miss.columns=['count']

miss.index.names=['Name']

miss['Name']=miss.index

miss
#plot the missing values

sns.set(style='whitegrid',color_codes=True)

sns.barplot(x='Name', y='count',data=miss)

plt.xticks(rotation=90)

sns
#SalePrice

sns.distplot(train['SalePrice'])
#skewness of SalePrice

train['SalePrice'].skew()
sales=np.log(train['SalePrice'])

print(sales.skew())

sns.distplot(sales)
#train=train.drop(['Id'],axis=1)
numeric_data=train.select_dtypes(include=[np.number])

cat_data=train.select_dtypes(exclude=[np.number])

numeric_data=numeric_data.drop('Id',axis=1)

numeric_data.shape, cat_data.shape
#correlation bewteen the numeric data



corr=numeric_data.corr()

sns.heatmap(corr)
#numeric correlation

corr['SalePrice'].sort_values(ascending=False)
#Working with OverallQual



train['OverallQual'].unique()
pivot=train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)

pivot
pivot.plot(kind='bar', color='black')
#Working with GrLivArea



sns.jointplot(x=train['GrLivArea'],y=train['SalePrice'])
#GarageArea

sns.jointplot(x=train['GarageArea'],y=train['SalePrice'])
cat_data.describe()
sp_pivot=train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)

sp_pivot
sp_pivot.plot(kind='bar', color='green')
cat=[f for f in train.columns if train.dtypes[f]=='object']



def anova(frame):

    anv=pd.DataFrame()

    anv['features']=cat

    pvals=[]

    for c in cat:

        samples=[]

        for cls in frame[c].unique():

            s=frame[frame[c]==cls]['SalePrice'].values

            samples.append(s)

        pval=stats.f_oneway(*samples)[1]

        pvals.append(pval)

    anv['pval']=pvals

    return anv.sort_values('pval')



cat_data['SalePrice']=train.SalePrice.values

k=anova(cat_data)

k['disparity']=np.log(1./k['pval'].values)

sns.barplot(data=k, x='features',y='disparity')

plt.xticks(rotation=90)

plt
#Numeric plots

num=[f for f in train.columns if train.dtypes[f]!='object']

num.remove('Id')

nd=pd.melt(train,value_vars=num)

n1=sns.FacetGrid(nd,col='variable', col_wrap=4, sharex=False, sharey=False)

n1=n1.map(sns.distplot,'value')

n1
#Categorical data



def boxplot(x,y,**kwargs):

    sns.boxplot(x=x,y=y)

    x=plt.xticks(rotation=90)

    

cat=[f for f in train.columns if train.dtypes[f]=='object']



p=pd.melt(train, id_vars='SalePrice',value_vars=cat)

g=sns.FacetGrid(p, col='variable', col_wrap=2, sharex= False, sharey='False',size=5)

g=g.map(boxplot, 'value','SalePrice')

g
#removing outliers



train.drop(train[train['GrLivArea']>4000].index, inplace=True)

train.shape
test.info()
#imputing using mode

test.loc[666, 'GarageQual'] = "TA" #stats.mode(test['GarageQual']).mode

test.loc[666, 'GarageCond'] = "TA" #stats.mode(test['GarageCond']).mode

test.loc[666, 'GarageFinish'] = "Unf" #stats.mode(test['GarageFinish']).mode

test.loc[666, 'GarageYrBlt'] = "1980" #np.nanmedian(test['GarageYrBlt'])` 
#mark as missing

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
neighborhood_map = {"MeadowV" : 0, "IDOTRR" : 1, "BrDale" : 1, "OldTown" : 1, "Edwards" : 1, "BrkSide" : 1,"Sawyer" : 1, "Blueste" : 1, "SWISU" : 2, "NAmes" : 2, "NPkVill" : 2, "Mitchel" : 2, "SawyerW" : 2, "Gilbert" : 2, "NWAmes" : 2, "Blmngtn" : 2, "CollgCr" : 2, "ClearCr" : 3, "Crawfor" : 3, "Veenker" : 3, "Somerst" : 3, "Timber" : 3, "StoneBr" : 4, "NoRidge" : 4, "NridgHt" : 4}



alldata['NeighborhoodBin'] = alldata2['Neighborhood'].map(neighborhood_map)

alldata.loc[alldata2.Neighborhood == 'NridgHt', "Neighborhood_Good"] = 1

alldata.loc[alldata2.Neighborhood == 'Crawfor', "Neighborhood_Good"] = 1

alldata.loc[alldata2.Neighborhood == 'StoneBr', "Neighborhood_Good"] = 1

alldata.loc[alldata2.Neighborhood == 'Somerst', "Neighborhood_Good"] = 1

alldata.loc[alldata2.Neighborhood == 'NoRidge', "Neighborhood_Good"] = 1

alldata["Neighborhood_Good"].fillna(0, inplace=True)

alldata["SaleCondition_PriceDown"] = alldata2.SaleCondition.replace({'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})



# House completed before sale or not

alldata["BoughtOffPlan"] = alldata2.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1})

alldata["BadHeating"] = alldata2.HeatingQC.replace({'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1, 'Po': 1})

alldata.shape
#create new data

train_new = alldata[alldata['SalePrice'].notnull()]

test_new = alldata[alldata['SalePrice'].isnull()]



print ('Train', train_new.shape)

print ('----------------')

print ('Test', test_new.shape)
#get numeric features

numeric_features = [f for f in train_new.columns if train_new[f].dtype != object]



#transform the numeric features using log(x + 1)

from scipy.stats import skew

skewed = train_new[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))

skewed = skewed[skewed > 0.75]

skewed = skewed.index

train_new[skewed] = np.log1p(train_new[skewed])

test_new[skewed] = np.log1p(test_new[skewed])

del test_new['SalePrice']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(train_new[numeric_features])

scaled = scaler.transform(train_new[numeric_features])



for i, col in enumerate(numeric_features):

       train_new[col] = scaled[:,i]



numeric_features.remove('SalePrice')

scaled = scaler.fit_transform(test_new[numeric_features])



for i, col in enumerate(numeric_features):

      test_new[col] = scaled[:,i]
def onehot(onehot_df, df, column_name, fill_na):

       onehot_df[column_name] = df[column_name]

       if fill_na is not None:

            onehot_df[column_name].fillna(fill_na, inplace=True)

       dummies = pd.get_dummies(onehot_df[column_name], prefix="_"+column_name)

       onehot_df = onehot_df.join(dummies)

       onehot_df = onehot_df.drop([column_name], axis=1)

       return onehot_df



def munge_onehot(df):

       onehot_df = pd.DataFrame(index = df.index)

       onehot_df = onehot(onehot_df, df, "MSSubClass", None)

       onehot_df = onehot(onehot_df, df, "MSZoning", "RL")

       onehot_df = onehot(onehot_df, df, "LotConfig", None)

       onehot_df = onehot(onehot_df, df, "Neighborhood", None)

       onehot_df = onehot(onehot_df, df, "Condition1", None)

       onehot_df = onehot(onehot_df, df, "BldgType", None)

       onehot_df = onehot(onehot_df, df, "HouseStyle", None)

       onehot_df = onehot(onehot_df, df, "RoofStyle", None)

       onehot_df = onehot(onehot_df, df, "Exterior1st", "VinylSd")

       onehot_df = onehot(onehot_df, df, "Exterior2nd", "VinylSd")

       onehot_df = onehot(onehot_df, df, "Foundation", None)

       onehot_df = onehot(onehot_df, df, "SaleType", "WD")

       onehot_df = onehot(onehot_df, df, "SaleCondition", "Normal")



       #Fill in missing MasVnrType for rows that do have a MasVnrArea.

       temp_df = df[["MasVnrType", "MasVnrArea"]].copy()

       idx = (df["MasVnrArea"] != 0) & ((df["MasVnrType"] == "None") | (df["MasVnrType"].isnull()))

       temp_df.loc[idx, "MasVnrType"] = "BrkFace"

       onehot_df = onehot(onehot_df, temp_df, "MasVnrType", "None")



       onehot_df = onehot(onehot_df, df, "LotShape", None)

       onehot_df = onehot(onehot_df, df, "LandContour", None)

       onehot_df = onehot(onehot_df, df, "LandSlope", None)

       onehot_df = onehot(onehot_df, df, "Electrical", "SBrkr")

       onehot_df = onehot(onehot_df, df, "GarageType", "None")

       onehot_df = onehot(onehot_df, df, "PavedDrive", None)

       onehot_df = onehot(onehot_df, df, "MiscFeature", "None")

       onehot_df = onehot(onehot_df, df, "Street", None)

       onehot_df = onehot(onehot_df, df, "Alley", "None")

       onehot_df = onehot(onehot_df, df, "Condition2", None)

       onehot_df = onehot(onehot_df, df, "RoofMatl", None)

       onehot_df = onehot(onehot_df, df, "Heating", None)



       # we'll have these as numerical variables too

       onehot_df = onehot(onehot_df, df, "ExterQual", "None")

       onehot_df = onehot(onehot_df, df, "ExterCond", "None")

       onehot_df = onehot(onehot_df, df, "BsmtQual", "None")

       onehot_df = onehot(onehot_df, df, "BsmtCond", "None")

       onehot_df = onehot(onehot_df, df, "HeatingQC", "None")

       onehot_df = onehot(onehot_df, df, "KitchenQual", "TA")

       onehot_df = onehot(onehot_df, df, "FireplaceQu", "None")

       onehot_df = onehot(onehot_df, df, "GarageQual", "None")

       onehot_df = onehot(onehot_df, df, "GarageCond", "None")

       onehot_df = onehot(onehot_df, df, "PoolQC", "None")

       onehot_df = onehot(onehot_df, df, "BsmtExposure", "None")

       onehot_df = onehot(onehot_df, df, "BsmtFinType1", "None")

       onehot_df = onehot(onehot_df, df, "BsmtFinType2", "None")

       onehot_df = onehot(onehot_df, df, "Functional", "Typ")

       onehot_df = onehot(onehot_df, df, "GarageFinish", "None")

       onehot_df = onehot(onehot_df, df, "Fence", "None")

       onehot_df = onehot(onehot_df, df, "MoSold", None)



       # Divide  the years between 1871 and 2010 into slices of 20 years

       year_map = pd.concat(pd.Series("YearBin" + str(i+1), index=range(1871+i*20,1891+i*20))  for i in range(0, 7))

       yearbin_df = pd.DataFrame(index = df.index)

       yearbin_df["GarageYrBltBin"] = df.GarageYrBlt.map(year_map)

       yearbin_df["GarageYrBltBin"].fillna("NoGarage", inplace=True)

       yearbin_df["YearBuiltBin"] = df.YearBuilt.map(year_map)

       yearbin_df["YearRemodAddBin"] = df.YearRemodAdd.map(year_map)



       onehot_df = onehot(onehot_df, yearbin_df, "GarageYrBltBin", None)

       onehot_df = onehot(onehot_df, yearbin_df, "YearBuiltBin", None)

       onehot_df = onehot(onehot_df, yearbin_df, "YearRemodAddBin", None)

       return onehot_df



#create one-hot features

onehot_df = munge_onehot(train)



neighborhood_train = pd.DataFrame(index=train_new.shape)

neighborhood_train['NeighborhoodBin'] = train_new['NeighborhoodBin']

neighborhood_test = pd.DataFrame(index=test_new.shape)

neighborhood_test['NeighborhoodBin'] = test_new['NeighborhoodBin']



onehot_df = onehot(onehot_df, neighborhood_train, 'NeighborhoodBin', None)
train_new = train_new.join(onehot_df) 
train_new.shape
#adding one hot features to test

onehot_df_te = munge_onehot(test)

onehot_df_te = onehot(onehot_df_te, neighborhood_test, "NeighborhoodBin", None)

test_new = test_new.join(onehot_df_te)

test_new.shape
#dropping some columns from the train data as they are not found in test

drop_cols = ["_Exterior1st_ImStucc", "_Exterior1st_Stone","_Exterior2nd_Other","_HouseStyle_2.5Fin","_RoofMatl_Membran", "_RoofMatl_Metal", "_RoofMatl_Roll", "_Condition2_RRAe", "_Condition2_RRAn", "_Condition2_RRNn", "_Heating_Floor", "_Heating_OthW", "_Electrical_Mix", "_MiscFeature_TenC", "_GarageQual_Ex",  "_PoolQC_Fa"]

train_new.drop(drop_cols, axis=1, inplace=True)

train_new.shape
#removing one column missing from train data

test_new.drop(["_MSSubClass_150"], axis=1, inplace=True)



# Drop these columns

drop_cols = ["_Condition2_PosN", # only two are not zero

         "_MSZoning_C (all)",

         "_MSSubClass_160"]



train_new.drop(drop_cols, axis=1, inplace=True)

test_new.drop(drop_cols, axis=1, inplace=True)
#create a label set

label_df = pd.DataFrame(index = train_new.index, columns = ['SalePrice'])

label_df['SalePrice'] = np.log(train['SalePrice'])

print("Training set size:", train_new.shape)

print("Test set size:", test_new.shape)
ct=['Alley', 'Condition2', 'Electrical', 'GarageType', 'GarageYrBlt', 'Heating', 'LandContour', 'LandSlope',

    'LotShape', 'MiscFeature', 'PavedDrive', 'RoofMatl', 'Street', 'Utilities']

for x in ct:

    train_new[x].fillna('None',inplace=True)

    test_new[x].fillna('None',inplace=True)
lbl=LabelEncoder()
train_new['Alley']=lbl.fit_transform(train_new['Alley'])

test_new['Alley']=lbl.fit_transform(test_new['Alley'])



train_new['Condition2']=lbl.fit_transform(train_new['Condition2'])

test_new['Condition2']=lbl.fit_transform(test_new['Condition2'])



train_new['Electrical']=lbl.fit_transform(train_new['Electrical'])

test_new['Electrical']=lbl.fit_transform(test_new['Electrical'])



train_new['GarageType']=lbl.fit_transform(train_new['GarageType'])

test_new['GarageType']=lbl.fit_transform(test_new['GarageType'])



train_new['Heating']=lbl.fit_transform(train_new['Heating'])

test_new['Heating']=lbl.fit_transform(test_new['Heating'])



train_new['LandContour']=lbl.fit_transform(train_new['LandContour'])

test_new['LandContour']=lbl.fit_transform(test_new['LandContour'])



train_new['LandSlope']=lbl.fit_transform(train_new['LandSlope'])

test_new['LandSlope']=lbl.fit_transform(test_new['LandSlope'])



train_new['LotShape']=lbl.fit_transform(train_new['LotShape'])

test_new['LotShape']=lbl.fit_transform(test_new['LotShape'])



train_new['MiscFeature']=lbl.fit_transform(train_new['MiscFeature'])

test_new['MiscFeature']=lbl.fit_transform(test_new['MiscFeature'])



train_new['PavedDrive']=lbl.fit_transform(train_new['PavedDrive'])

test_new['PavedDrive']=lbl.fit_transform(test_new['PavedDrive'])



train_new['RoofMatl']=lbl.fit_transform(train_new['RoofMatl'])

test_new['RoofMatl']=lbl.fit_transform(test_new['RoofMatl'])



train_new['Street']=lbl.fit_transform(train_new['Street'])

test_new['Street']=lbl.fit_transform(test_new['Street'])



train_new['Utilities']=lbl.fit_transform(train_new['Utilities'])

test_new['Utilities']=lbl.fit_transform(test_new['Utilities'])



train_new['GarageYrBlt']=lbl.fit_transform(train_new['GarageYrBlt'])

#test_new['Utilities']=lbl.fit_transform(test_new['Utilities'])
test_new.loc[666, 'GarageYrBlt'] = 1980

test_new['GarageYrBlt']=lbl.fit_transform(test_new['GarageYrBlt'])
train_new=train_new.drop('_NeighborhoodBin_-0.0476080847178',axis=1)

test_new=test_new.drop('_NeighborhoodBin_-1.08757678794',axis=1)



train_new.shape,test_new.shape
train_new=train_new.drop('SalePrice',axis=1)
import xgboost as xgb
regr=xgb.XGBRegressor(colsample_bytree=0.2,

                     gamma=0.0,

                     learning_rate=0.05,

                     max_depth=6,

                     min_child_weight=1.5,

                     n_estimators=7200,

                     reg_alpha=0.9,reg_lambda=0.6,

                     subsample=0.2, seed=42,

                     silent=1)

regr.fit(train_new,label_df)
from sklearn.metrics import mean_squared_error

def rmse(y_test,y_pred):

      return np.sqrt(mean_squared_error(y_test,y_pred))



# run prediction on training set to get an idea of how well it does

y_pred = regr.predict(train_new)

y_test = label_df

print("XGBoost score on training set: ", rmse(y_test, y_pred))
y_pred_xgb = regr.predict(test_new)
pred1 = pd.DataFrame({'Id': test['Id'], 'SalePrice': np.exp(y_pred_xgb)})

pred1.to_csv('xgbnono.csv', header=True, index=False)
from sklearn.linear_model import Lasso
alpha=0.00099

regr=Lasso(alpha=alpha,max_iter=50000)

regr.fit(train_new,label_df)
y_pred=regr.predict(train_new)
rmse(label_df,y_pred)
y_pred_lasso=regr.predict(test_new)

lasso_ex=np.exp(y_pred_lasso)

pred1 = pd.DataFrame({'Id': test['Id'], 'SalePrice': lasso_ex})

pred1.to_csv('lasso_model.csv', header=True, index=False)
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import StandardScaler



np.random.seed(10)



def base_model():

    model=Sequential()

    model.add(Dense(20,input_dim=412,init='normal',activation='relu'))

    model.add(Dense(10,init='normal',activation='relu'))

    model.add(Dense(1,init='normal'))

    model.compile(loss='mean_squared_error',optimizer='adam')

    return model



seed=7

np.random.seed(seed)



scale=StandardScaler()

X_train=scale.fit_transform(train_new)

X_test=scale.fit_transform(test_new)



keras_label=label_df.as_matrix()

clf=KerasRegressor(build_fn=base_model,nb_epoch=1000,batch_size=8,verbose=0)

clf.fit(X_train,keras_label)
kpred = clf.predict(X_test) 

kpred = np.exp(kpred)

pred_df = pd.DataFrame(kpred, index=test["Id"], columns=["SalePrice"]) 

pred_df.to_csv('keras1.csv', header=True, index_label='Id') 
#simple average

y_pred = (y_pred_xgb + y_pred_lasso) / 2

y_pred = np.exp(y_pred)

pred_df = pd.DataFrame(y_pred, index=test["Id"], columns=["SalePrice"])

pred_df.to_csv('ensemble1.csv', header=True, index_label='Id')