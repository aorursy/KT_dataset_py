import pandas as pd

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns                         # Importing Important Libraries
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")     # Getting test and train data
df_train.head(6)
df_test.head(6)
df_train.columns    #list of columns in train
df_test.columns            #list of columns in test
df_train.info() # to see datatypes and check null values
df_train.describe()  #to check how varies with respect to each columns and any outliers 
# Custom function for initial checks

def DF_initial_observations(df):

    '''Gives basic details of columns in a dataframe : Data types, distinct values, NAs and sample'''

    if isinstance(df, pd.DataFrame):

        total_na=0

        for i in range(len(df.columns)):        

            total_na+= df.isna().sum()[i]

        print('Dimensions : %d rows, %d columns' % (df.shape[0],df.shape[1]))

        print("Total NA values : %d" % (total_na))

        print('%38s %10s     %10s %10s %15s' % ('Column name', ' Data Type', '# Distinct', ' NA values', ' Sample value'))

        for i in range(len(df.columns)):

            col_name = df.columns[i]

            sampl = df[col_name].sample(1)

            sampl.apply(pd.Categorical)

            sampl_p = str(sampl.iloc[0,])

            print('%38s %10s :   %10d  %10d %15s' % (df.columns[i],df.dtypes[i],df.nunique()[i],df.isna().sum()[i], sampl_p))

    else:

        print('Expected a DataFrame but got a %15s ' % (type(data)))
DF_initial_observations(df_train)
# Correlation matrix - linear relation among independent attributes and with the Target attribute



sns.set(style="white")



# Compute the correlation matrix

correln = df_train.corr()



# Generate a mask for the upper triangle

#mask = np.zeros_like(correln, dtype=np.bool)

#mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(correln,  cmap=cmap, vmax=.3, #mask=mask,

            linewidths=.8, cbar_kws={"shrink": .9})
np.corrcoef(df_train[["GarageCars", "GarageArea"]])
df_train.corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
df_train.corr()[["GarageCars", "1stFlrSF", "LotFrontage", "YrSold"]] #df_train[df_train.columns[1:]].corr()["GarageCars"][:]
plt.boxplot(df_train["LotArea"])
df_train["LotFrontage"].unique()
df_train["Neighborhood"].unique()
df_train_NAmes = df_train.loc[df_train["Neighborhood"]== "NAmes"]
df_train_NAmes
df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='NAmes') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='NAmes']['LotFrontage']), df_train['LotFrontage'])
df_train["LotFrontage"].isna().sum()
df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='Sawyer') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='Sawyer']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='NWAmes') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='NWAmes']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='SawyerW') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='SawyerW']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='CollgCr') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='CollgCr']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='Gilbert') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='Gilbert']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='IDOTRR') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='IDOTRR']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='Crawfor') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='Crawfor']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='ClearCr') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='ClearCr']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='NPkVill') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='NPkVill']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='Timber') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='Timber']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='BrkSide') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='BrkSide']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='Veenker') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='Veenker']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='OldTown') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='OldTown']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='Somerst') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='Somerst']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='Mitchel') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='Mitchel']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='Edwards') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='Edwards']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='Blmngtn') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='Blmngtn']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='StoneBr') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='StoneBr']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='MeadowV') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='MeadowV']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='NoRidge') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='NoRidge']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='SWISU') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='SWISU']['LotFrontage']), df_train['LotFrontage'])

df_train['LotFrontage'] = np.where(((df_train['Neighborhood']=='NridgHt') & (df_train['LotFrontage'].isna())), np.nanmedian(df_train[df_train['Neighborhood']=='NridgHt']['LotFrontage']), df_train['LotFrontage'])
a = df_train[["Neighborhood","LotFrontage"]]

nulls = a[a.isnull().any(axis=1)]
nulls
df_train["BsmtQual"].value_counts()
maxi_BsmtQual = df_train["BsmtQual"].value_counts().index.tolist()

maxi_BsmtQual[0]
df_train["BsmtQual"].unique()
df_train["BsmtQual"] = df_train["BsmtQual"].replace(np.nan, maxi_BsmtQual[0])
df_train["BsmtQual"].unique()
df_train["BsmtCond"].value_counts()
maxi_BsmtCond = df_train["BsmtCond"].value_counts().index.tolist()

maxi_BsmtCond[0]
df_train["BsmtCond"].unique()
df_train["BsmtCond"] = df_train["BsmtCond"].replace(np.nan, maxi_BsmtCond[0])
df_train["BsmtCond"].unique()
df_train["BsmtExposure"].value_counts()

df_train["BsmtExposure"].unique()
maxi_BsmtExposure = df_train["BsmtExposure"].value_counts().index.tolist()

maxi_BsmtExposure[0]
df_train["BsmtExposure"] = df_train["BsmtExposure"].replace(np.nan,maxi_BsmtExposure[0] )
df_train["BsmtExposure"].unique()
df_train["BsmtExposure"].isna().sum()
df_train["BsmtFinType1"].unique()
df_train["BsmtFinType1"].value_counts()
maxi_BsmtFinType1 = df_train["BsmtFinType1"].value_counts().index.tolist()

maxi_BsmtFinType1[0]
df_train["BsmtFinType1"] = df_train["BsmtFinType1"].replace(np.nan, maxi_BsmtFinType1[0])
df_train["BsmtFinType1"].unique()
df_train["BsmtFinType2"].unique()
df_train["BsmtFinType2"].value_counts()
maxi_BsmtFinType2 = df_train["BsmtFinType2"].value_counts().index.tolist()

maxi_BsmtFinType2[0]
df_train["BsmtFinType2"] = df_train["BsmtFinType2"].replace(np.nan, maxi_BsmtFinType2[0])
df_train["BsmtFinType2"].unique()
df_train["Electrical"].unique()
df_train["Electrical"].value_counts()
maxi_Electrical = df_train["Electrical"].value_counts().index.tolist()

maxi_Electrical[0]
df_train["Electrical"] = df_train["Electrical"].replace(np.nan, maxi_Electrical[0])
df_train["Electrical"].unique()
df_train["FireplaceQu"].unique()
df_train["FireplaceQu"].value_counts()
df_train[df_train["FireplaceQu"].isnull()]
b = df_train[["FireplaceQu","Neighborhood", "2ndFlrSF", "Fireplaces"]]

nulls_FireplaceQu = b[b.isnull().any(axis=1)]
nulls_FireplaceQu
nulls_FireplaceQu["Fireplaces"].unique()
df_train["Fireplaces"].value_counts()
df_train["FireplaceQu"] = df_train["FireplaceQu"].replace(np.nan, "unknown")
df_train["FireplaceQu"].value_counts()
df_train["GarageType"].unique()
df_train["GarageType"].value_counts()
c = df_train[["GarageType","Neighborhood", "GarageArea", "GarageCars", "GarageYrBlt"]]

nulls_GarageType = c[c.isnull().any(axis=1)]
nulls_GarageType
nulls_GarageType["GarageCars"].value_counts()
df_train["GarageCars"].value_counts()
nulls_GarageType["GarageYrBlt"].unique()
df_train["GarageYrBlt"].unique()
d = df_train[["GarageYrBlt","Neighborhood", "GarageType", "GarageFinish", "GarageQual", "GarageCond"]]

nulls_GarageYrBlt = d[d.isnull().any(axis=1)]
nulls_GarageYrBlt
df_train["GarageYrBlt"] = df_train["GarageYrBlt"].replace(np.nan, "unknown")

df_train["GarageType"] = df_train["GarageType"].replace(np.nan, "unknown")

df_train["GarageFinish"] = df_train["GarageFinish"].replace(np.nan, "unknown")

df_train["GarageQual"] = df_train["GarageQual"].replace(np.nan, "unknown")

df_train["GarageCond"] = df_train["GarageCond"].replace(np.nan, "unknown")
DF_initial_observations(df_train)
df_train1 = df_train.drop(["PoolQC", "Fence", "MiscFeature", "Alley"], axis = 1)
e = df_train1[["Id","MasVnrType","MasVnrArea", "Foundation", "Exterior1st", "Exterior2nd"]]

nulls_MasVnr = e[e.isnull().any(axis=1)]
nulls_MasVnr
masvar  =df_train1[(df_train1["Foundation"] == 'PConc') & (df_train1['Exterior1st'] == 'VinylSd') & (df_train1['Exterior2nd'] == 'VinylSd' ) ]
masvar
df_masvar = masvar[['MasVnrType','MasVnrArea','Foundation','Exterior1st','Exterior2nd']]
df_masvar["MasVnrType"].value_counts()
maxi_MasVnrType = df_masvar["MasVnrType"].value_counts().index.tolist()

maxi_MasVnrType[0]
df_train1['MasVnrType'] = np.where((df_train1['MasVnrType'].isna()) & (df_train1["Foundation"] == 'PConc') & (df_train1['Exterior1st'] == 'VinylSd') & (df_train1['Exterior2nd'] == 'VinylSd' ),

maxi_MasVnrType[0], df_train1['MasVnrType'])
maxi_MasVnrArea = df_masvar["MasVnrArea"].value_counts().index.tolist()

maxi_MasVnrArea[0]
df_train1['MasVnrArea'] = np.where((df_train1['MasVnrArea'].isna()) & (df_train1["Foundation"] == 'PConc') & (df_train1['Exterior1st'] == 'VinylSd') & (df_train1['Exterior2nd'] == 'VinylSd' ),

maxi_MasVnrArea[0], df_train1['MasVnrArea'])
maxi_MasVnrType_final = df_train1["MasVnrType"].value_counts().index.tolist()

maxi_MasVnrType_final[0]
df_train1["MasVnrType"] = df_train1["MasVnrType"].replace(np.nan, maxi_MasVnrType_final[0])
maxi_MasVnrArea_final = df_train1["MasVnrArea"].value_counts().index.tolist()

maxi_MasVnrArea_final[0]
df_train1["MasVnrArea"] = df_train1["MasVnrArea"].replace(np.nan, maxi_MasVnrArea_final[0])
DF_initial_observations(df_train1)
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train1[cols], size = 2.5)

plt.show();
df_train1["MiscVal"].describe()
plt.boxplot(df_train1["MiscVal"])

plt.show()
plt.boxplot(df_train1["GrLivArea"])
df_train1["WoodDeckSF"].describe()
plt.boxplot(df_train1["WoodDeckSF"])
plt.boxplot(df_train1["3SsnPorch"])
var = '3SsnPorch'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
df_train1[df_train1["3SsnPorch"]==508.0000000]

df_train_out =df_train1[df_train1["3SsnPorch"] != 508.0000000]
plt.boxplot(df_train1["LotFrontage"])

plt.show()
var1 = 'LotFrontage'

data = pd.concat([df_train['SalePrice'], df_train[var1]], axis=1)

data.plot.scatter(x=var1, y='SalePrice', ylim=(0,800000));
var2 = 'MasVnrArea'

data = pd.concat([df_train['SalePrice'], df_train[var2]], axis=1)

data.plot.scatter(x=var2, y='SalePrice', ylim=(0,800000));
var3 = 'BsmtFinSF1'

data = pd.concat([df_train['SalePrice'], df_train[var3]], axis=1)

data.plot.scatter(x=var3, y='SalePrice', ylim=(0,800000));
var4 = 'BsmtFinSF2'

data = pd.concat([df_train['SalePrice'], df_train[var4]], axis=1)

data.plot.scatter(x=var4, y='SalePrice', ylim=(0,800000));
var5 = 'ScreenPorch'

data = pd.concat([df_train['SalePrice'], df_train[var5]], axis=1)

data.plot.scatter(x=var5, y='SalePrice', ylim=(0,800000));
df_train1[(df_train1['MiscVal'] == 15500.000000) | (df_train1['GrLivArea'] == 5642.000000) | (df_train1['LotFrontage'] == 313.000000) |(df_train1['LotArea'] == 215245.000000) | (df_train1['MasVnrArea'] == 1600.000000) |(df_train1['BsmtFinSF1'] == 5644.000000)

| (df_train1['BsmtFinSF2'] == 1474.000000) | (df_train1['TotalBsmtSF'] == 6110.000000) | (df_train1['1stFlrSF'] == 4692.000000)

    | (df_train1['LowQualFinSF'] == 572.000000) | (df_train1['WoodDeckSF'] == 857.000000)

    | (df_train1['EnclosedPorch'] == 552.000000) | (df_train1['PoolArea'] == 738.000000) | (df_train1['3SsnPorch'] == 508.000000)]
fina_train = df_train1[(df_train1["MiscVal"] != 15500.000000) & (df_train1["GrLivArea"] != 5642.000000 ) & (df_train1["LotFrontage"] != 313.000000) & (df_train1["LotArea"] != 215245.000000) & (df_train1["MasVnrArea"] != 1600.000000) & (df_train1["BsmtFinSF1"] != 5644.000000) & (df_train["BsmtFinSF2"] != 1474.000000) & (df_train1["TotalBsmtSF"] != 6110.000000) & (df_train1["1stFlrSF"] != 4692.000000) & (df_train1["LowQualFinSF"] != 572.000000) & (df_train1["WoodDeckSF"] != 857.000000) & (df_train1["EnclosedPorch"] != 552.000000) & (df_train1["PoolArea"] != 738.000000) & (df_train1["3SsnPorch"] != 508.000000)]
fina_train.shape
df_train1.shape
df_train.shape
fina_train["Street"].unique()
fina_train['HouseStyle'].unique()
sns.barplot(fina_train['HouseStyle'],fina_train['SalePrice'])
sns.barplot(fina_train["MSSubClass"], fina_train["SalePrice"])
sns.barplot(fina_train["MSZoning"], fina_train["SalePrice"])
sns.barplot(fina_train["OverallQual"], fina_train["SalePrice"])
sns.barplot(fina_train["BldgType"], fina_train["SalePrice"])
sns.barplot(fina_train["RoofStyle"], fina_train["SalePrice"])
sns.barplot(fina_train["Foundation"], fina_train["SalePrice"])
sns.barplot(fina_train["FullBath"], fina_train["SalePrice"])
sns.barplot(fina_train["HalfBath"], fina_train["SalePrice"])
col_name="Neighborhood"

col_value=np.sort(fina_train[col_name].unique()).tolist()

plt.figure(figsize=(16,8))

sns.stripplot(x=col_name,y="SalePrice", data=fina_train,order=col_value,linewidth=.6)

plt.xticks(rotation=45)

plt.xlabel("Neighborhood")

plt.ylabel("SalePrice")

plt.title("neighborhood with salesprice")

plt.show()
col_name="Neighborhood"

col_value=np.sort(fina_train[col_name].unique()).tolist()

plt.figure(figsize=(16,8))

sns.violinplot(x=col_name,y="SalePrice", data=fina_train,order=col_value,linewidth=.6)

plt.xticks(rotation=45)

plt.xlabel("Neighborhood")

plt.ylabel("SalePrice")

plt.title("neighborhood with salesprice")

plt.show()
col_name="Neighborhood"

col_value=np.sort(fina_train[col_name].unique()).tolist()

plt.figure(figsize=(16,8))

sns.boxplot(x=col_name,y="SalePrice", data=fina_train,order=col_value,linewidth=.6)

plt.xticks(rotation=45)

plt.xlabel("Neighborhood")

plt.ylabel("SalePrice")

plt.title("neighborhood with salesprice")

plt.show()
fina_train.dtypes
fina_train["MSZoning"].value_counts()
fina_train["MSZoning"] = fina_train["MSZoning"].replace("RL", 5)

fina_train["MSZoning"] = fina_train["MSZoning"].replace("RM", 4)

fina_train["MSZoning"] = fina_train["MSZoning"].replace("FV", 3)

fina_train["MSZoning"] = fina_train["MSZoning"].replace("RH", 2)

fina_train["MSZoning"] = fina_train["MSZoning"].replace("C (all)", 1)
fina_train["MSZoning"].unique()
fina_train["Street"].value_counts()
fina_train["Street"] = fina_train["Street"].map({"Pave" :1 , "Grvl" : 0}).astype(int)
fina_train["Street"].unique()
fina_train["LotShape"].value_counts()
fina_train['LotShape'] = fina_train["LotShape"].map({'Reg':3, 'IR1':2,'IR2':1,'IR3':0 }).astype(int)
fina_train["LotShape"].unique()
fina_train['LandContour'].value_counts()
fina_train["LandContour"] = fina_train["LandContour"].map({"Lvl" : 3, "Bnk" : 2, "HLS" : 1, "Low" : 0}).astype(int)
fina_train["LandContour"].unique()
fina_train['Utilities'].value_counts()
fina_train['Utilities'] = fina_train["Utilities"].map({'AllPub':1, 'NoSeWa':0 }).astype(int)
fina_train['LandSlope'] = fina_train["LandSlope"].map({'Gtl':0, 'Mod':1, 'Sev':2 }).astype(int)
fina_train['BldgType'].unique()
fina_train['BldgType'] = fina_train["BldgType"].map({'1Fam':0, '2fmCon':1, 'Duplex':2, 'TwnhsE':3,

'Twnhs':4}).astype(int)
fina_train['HouseStyle'].unique()
fina_train['HouseStyle'] = fina_train["HouseStyle"].map({'2Story':3, '1Story':0, '1.5Fin':2, '1.5Unf':1,

'SFoyer':6, 'SLvl':7, '2.5Unf':4, '2.5Fin':5  }).astype(int)
fina_train["ExterQual"].unique()
fina_train["ExterQual"] = fina_train["ExterQual"].map({"Ex" : 3, "Gd" : 2, "TA" : 1, "Fa" : 0 }).astype(int)
fina_train["ExterQual"].unique()
fina_train["ExterCond"].unique()
fina_train["ExterCond"] = fina_train["ExterCond"].map({"Ex" : 4, "Gd" : 3, "TA" : 2, "Fa" : 1, "Po" : 0}).astype(int)
fina_train["BsmtQual"].unique()
fina_train["BsmtQual"] = fina_train["BsmtQual"].map({"Ex" : 3, "Gd" : 2, "TA" : 1, "Fa" : 0 }).astype(int)
fina_train["KitchenQual"].unique()
fina_train["KitchenQual"] = fina_train["KitchenQual"].map({"Ex" : 3, "Gd" : 2, "TA" : 1, "Fa" : 0 }).astype(int)
fina_train["Functional"].value_counts()
fina_train["Functional"] = fina_train["Functional"].map({"Typ" : 6, "Min1" : 5, "Maj1" : 2, "Min2" : 4, "Mod" : 3, "Maj2" : 1, "Sev" : 0}).astype(int)
fina_train["FireplaceQu"].unique()
fina_train["FireplaceQu"] = fina_train["FireplaceQu"].map({"unknown" : 0, "Ex" : 5, "Gd" : 4, "TA" : 3, "Fa" : 2, "Po" : 1}).astype(int)
fina_train["GarageYrBlt"].unique()
fina_train["GarageYrBlt"] = fina_train["GarageYrBlt"].replace("unknown", 0).astype(float)
fina_train["GarageFinish"].unique()
fina_train["GarageFinish"] = fina_train["GarageFinish"].map({"RFn" : 2, "Unf" : 1, "Fin" : 3, "unknown" : 0}).astype(int)
fina_train["GarageQual"].unique()
fina_train["GarageQual"] = fina_train["GarageQual"].map({"unknown" : 0, "Ex" : 5, "Gd" : 4, "TA" : 3, "Fa" : 2, "Po" : 1}).astype(int)
fina_train["GarageCond"].unique()
fina_train["GarageCond"] = fina_train["GarageCond"].map({"unknown" : 0, "Ex" : 5, "Gd" : 4, "TA" : 3, "Fa" : 2, "Po" : 1}).astype(int)
fina_train["PavedDrive"].unique()
fina_train["PavedDrive"] = fina_train["PavedDrive"].map({"Y" : 2, "N" : 0, "P" : 1}).astype(int)
fina_train["BsmtCond"].unique()
fina_train["BsmtCond"] = fina_train["BsmtCond"].map({ "Gd" : 3, "TA" : 2, "Fa" : 1, "Po" : 0 }).astype(int)
fina_train["BsmtExposure"].unique()
fina_train["BsmtExposure"] = fina_train["BsmtExposure"].map({ "Gd" : 3, "No" : 0, "Mn" : 1, "Av" : 2 }).astype(int)
fina_train["BsmtFinType1"].unique()
fina_train["BsmtFinType1"] = fina_train["BsmtFinType1"].map({"GLQ" : 5, "ALQ" : 4, "Unf" : 0, "Rec" : 2, "BLQ" : 3, "LwQ" : 1}).astype(int)
fina_train["BsmtFinType2"].unique()
fina_train["BsmtFinType2"] = fina_train["BsmtFinType2"].map({"GLQ" : 5, "ALQ" : 4, "Unf" : 0, "Rec" : 2, "BLQ" : 3, "LwQ" : 1}).astype(int)
fina_train["BsmtFinType2"].unique()
fina_train["HeatingQC"].unique()
fina_train["HeatingQC"] = fina_train["HeatingQC"].map({"Ex" : 4, "Gd" : 3, "TA" : 2, "Fa" : 1, "Po" : 0}).astype(int)
fina_train["CentralAir"].unique()
fina_train["CentralAir"] = fina_train["CentralAir"].map({"Y" : 1, "N" : 0}).astype(int)
final_train = fina_train.drop(['SalePrice'], axis=1)

y = fina_train['SalePrice']
dummy_final_train = pd.get_dummies(final_train, columns= ["LotConfig", "Neighborhood", "Condition1" , "Condition2", "RoofMatl", "Exterior1st", "Exterior2nd", "Heating", "Electrical","RoofStyle" ,  "MasVnrType", "Foundation", "GarageType", "SaleType" , "SaleCondition"])

final_train = pd.concat([final_train, dummy_final_train], axis = 1)
final_train = final_train.drop(["LotConfig", "Neighborhood", "Condition1", "Condition2", "RoofMatl", "Exterior1st", "Exterior2nd", "Heating", "Electrical", "RoofStyle",  "MasVnrType", "Foundation", "GarageType", "SaleType", "SaleCondition"], axis = 1)
final_train.shape
final_train.dtypes
df_test.describe()
# Custom function for initial checks

def DF_initial_observations(df):

    '''Gives basic details of columns in a dataframe : Data types, distinct values, NAs and sample'''

    if isinstance(df, pd.DataFrame):

        total_na=0

        for i in range(len(df.columns)):        

            total_na+= df.isna().sum()[i]

        print('Dimensions : %d rows, %d columns' % (df.shape[0],df.shape[1]))

        print("Total NA values : %d" % (total_na))

        print('%38s %10s     %10s %10s %15s' % ('Column name', ' Data Type', '# Distinct', ' NA values', ' Sample value'))

        for i in range(len(df.columns)):

            col_name = df.columns[i]

            sampl = df[col_name].sample(1)

            sampl.apply(pd.Categorical)

            sampl_p = str(sampl.iloc[0,])

            print('%38s %10s :   %10d  %10d %15s' % (df.columns[i],df.dtypes[i],df.nunique()[i],df.isna().sum()[i], sampl_p))

    else:

        print('Expected a DataFrame but got a %15s ' % (type(data)))
DF_initial_observations(df_test)
df_test["MSZoning"].unique()
f = df_test[["Neighborhood", "MSZoning"]]

f_nulls = f[f.isnull().any(axis = 1)]
f_nulls
f_mode_IDOTRR = f[f["Neighborhood"]== "IDOTRR"]["MSZoning"].value_counts().index.tolist()
f_mode_IDOTRR[0]
f_mode_Mitchel = f[f["Neighborhood"]== "Mitchel"]["MSZoning"].value_counts().index.tolist()
f_mode_Mitchel[0]
df_test["MSZoning"] = np.where((df_test["MSZoning"].isna()) & (df_test["Neighborhood"] == "IDOTRR"), f_mode_IDOTRR[0], df_test["MSZoning"])

df_test["MSZoning"] = np.where((df_test["MSZoning"].isna()) & (df_test["Neighborhood"] == "Mitchel"), f_mode_Mitchel[0], df_test["MSZoning"])
g = df_test[["Neighborhood", "LotFrontage"]]

nulls_Lot = g[g.isna().any(axis=1)]
nulls_Lot["Neighborhood"].unique()
df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='Sawyer') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='Sawyer']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='Gilbert') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='Gilbert']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='Somerst') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='Somerst']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='NWAmes') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='NWAmes']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='OldTown') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='OldTown']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='BrkSide') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='BrkSide']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='SWISU') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='SWISU']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='ClearCr') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='ClearCr']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='Edwards') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='Edwards']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='CollgCr') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='CollgCr']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='Mitchel') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='Mitchel']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='NoRidge') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='NoRidge']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='SawyerW') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='SawyerW']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='Veenker') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='Veenker']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='Crawfor') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='Crawfor']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='Timber') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='Timber']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='Blmngtn') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='Blmngtn']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='IDOTRR') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='IDOTRR']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='MeadowV') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='MeadowV']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='NridgHt') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='NridgHt']['LotFrontage']), df_test['LotFrontage'])

df_test['LotFrontage'] = np.where(((df_test['Neighborhood']=='NAmes') & (df_test['LotFrontage'].isna())), np.nanmedian(df_test[df_test['Neighborhood']=='NAmes']['LotFrontage']), df_test['LotFrontage'])
df_test["Exterior1st"].unique()
maxi_Ext1st = df_test["Exterior1st"].value_counts().index.tolist()

maxi_Ext1st[0]
df_test["Exterior1st"] = df_test["Exterior1st"].replace(np.nan, maxi_Ext1st[0])
maxi_Ext2nd = df_test["Exterior2nd"].value_counts().index.tolist()

maxi_Ext2nd[0]
df_test["Exterior2nd"] = df_test["Exterior2nd"].replace(np.nan, maxi_Ext2nd[0])
test_mas = df_test[['MasVnrType','MasVnrArea','Foundation','Exterior1st','Exterior2nd']]
nulls_mas = test_mas[test_mas.isnull().any(axis =1)]
nulls_mas
df_test_Pconc = df_test[(df_test["Foundation"] == "PConc") & (df_test["Exterior1st"] == "VinylSd") & (df_test["Exterior2nd"] == "VinylSd") ]
df_test_Pconc[["MasVnrType", "MasVnrArea", "Foundation", "Exterior1st", "Exterior2nd"]]
df_test_mastype = df_test_Pconc["MasVnrType"].value_counts().index.tolist()
df_test_mastype[0]
df_test['MasVnrType'] = np.where((df_test['MasVnrType'].isna()) & (df_test["Foundation"] == 'PConc') & (df_test['Exterior1st'] == 'VinylSd') & (df_test['Exterior2nd'] == 'VinylSd' ),

df_test_mastype[0], df_test['MasVnrType'])
df_test_masarea = df_test_Pconc["MasVnrArea"].value_counts().index.tolist()
df_test_masarea[0]
df_test['MasVnrArea'] = np.where((df_test['MasVnrArea'].isna()) & (df_test["Foundation"] == 'PConc') & (df_test['Exterior1st'] == 'VinylSd') & (df_test['Exterior2nd'] == 'VinylSd' ),

df_test_masarea[0], df_test['MasVnrArea'])
nulls_mas1 = test_mas[test_mas.isnull().any(axis =1)]
nulls_mas1
df_test_mast = df_test["MasVnrType"].value_counts().index.tolist()

df_test_mast[0]
df_test["MasVnrType"] = df_test["MasVnrType"].replace(np.nan, df_test_mast[0])
df_test_masa = df_test["MasVnrArea"].value_counts().index.tolist()

df_test_masa[0]
df_test["MasVnrArea"] = df_test["MasVnrArea"].replace(np.nan, df_test_masa[0])
df_test["MasVnrArea"].isnull().sum()
df_test_bsmt = df_test[["Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]]
nulls_bsmt = df_test_bsmt[df_test_bsmt.isnull().any(axis = 1)]
nulls_bsmt
df_test_bsmt_f = df_test[(df_test["Foundation"] == "Slab")]
bsmt_slab = df_test_bsmt_f[["Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]]
bsmt_slab
df_test['BsmtQual'] = np.where((df_test['BsmtQual'].isna()) & (df_test["Foundation"] == 'Slab'), "unknown", df_test['BsmtQual'])

df_test['BsmtCond'] = np.where((df_test['BsmtCond'].isna()) & (df_test["Foundation"] == 'Slab'), "unknown", df_test['BsmtCond'])

df_test['BsmtExposure'] = np.where((df_test['BsmtExposure'].isna()) & (df_test["Foundation"] == 'Slab'), "unknown", df_test['BsmtExposure'])

df_test['BsmtFinType1'] = np.where((df_test['BsmtFinType1'].isna()) & (df_test["Foundation"] == 'Slab'), "unknown", df_test['BsmtFinType1'])

df_test['BsmtFinType2'] = np.where((df_test['BsmtFinType2'].isna()) & (df_test["Foundation"] == 'Slab'), "unknown", df_test['BsmtFinType2'])
df_test["BsmtQual"].isnull().sum()
df_test_bsmt_cb = df_test[(df_test["Foundation"] == "CBlock")]
bsmt_cblock = df_test_bsmt_cb[["Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]]
bsmt_cblock
bsmt_qual_mode = bsmt_cblock["BsmtQual"].value_counts().index.tolist()

bsmt_qual_mode[0]
df_test['BsmtQual'] = np.where((df_test['BsmtQual'].isna()) & (df_test["Foundation"] == 'CBlock'), bsmt_qual_mode[0], df_test['BsmtQual'])
bsmt_cond_mode = bsmt_cblock["BsmtCond"].value_counts().index.tolist()

bsmt_cond_mode[0]
df_test['BsmtCond'] = np.where((df_test['BsmtCond'].isna()) & (df_test["Foundation"] == 'CBlock'), bsmt_cond_mode[0], df_test['BsmtCond'])
bsmt_Exp_mode = bsmt_cblock["BsmtExposure"].value_counts().index.tolist()

bsmt_Exp_mode[0]
df_test['BsmtExposure'] = np.where((df_test['BsmtExposure'].isna()) & (df_test["Foundation"] == 'CBlock'), bsmt_Exp_mode[0], df_test['BsmtExposure'])
bsmt_fin1_mode = bsmt_cblock["BsmtFinType1"].value_counts().index.tolist()

bsmt_fin1_mode[0]
bsmt_fin2_mode = bsmt_cblock["BsmtFinType2"].value_counts().index.tolist()

bsmt_fin2_mode[0]
df_test['BsmtFinType1'] = np.where((df_test['BsmtFinType1'].isna()) & (df_test["Foundation"] == 'CBlock'), bsmt_fin1_mode[0], df_test['BsmtFinType1'])

df_test['BsmtFinType2'] = np.where((df_test['BsmtFinType2'].isna()) & (df_test["Foundation"] == 'CBlock'), bsmt_fin2_mode[0], df_test['BsmtFinType2'])
nulls_bsmt
df_test_bsmt_Pc = df_test[(df_test["Foundation"] == "PConc")]
bsmt_PConc = df_test_bsmt_Pc[["Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]]
bsmt_PConc
bsmt_pc_qual_mode = bsmt_PConc["BsmtQual"].value_counts().index.tolist()

bsmt_pc_qual_mode[0]
bsmt_pc_Cond_mode = bsmt_PConc["BsmtCond"].value_counts().index.tolist()

bsmt_pc_Cond_mode[0]
bsmt_pc_Ex_mode = bsmt_PConc["BsmtExposure"].value_counts().index.tolist()

bsmt_pc_Ex_mode[0]
bsmt_pc_fin1_mode = bsmt_PConc["BsmtFinType1"].value_counts().index.tolist()

bsmt_pc_fin1_mode[0]
bsmt_pc_fin2_mode = bsmt_PConc["BsmtFinType2"].value_counts().index.tolist()

bsmt_pc_fin2_mode[0]
df_test['BsmtQual'] = np.where((df_test['BsmtQual'].isna()) & (df_test["Foundation"] == 'PConc'), bsmt_pc_qual_mode[0], df_test['BsmtQual'])

df_test['BsmtCond'] = np.where((df_test['BsmtCond'].isna()) & (df_test["Foundation"] == 'PConc'), bsmt_pc_Cond_mode[0], df_test['BsmtCond'])

df_test['BsmtExposure'] = np.where((df_test['BsmtExposure'].isna()) & (df_test["Foundation"] == 'PConc'), bsmt_pc_Ex_mode[0], df_test['BsmtExposure'])

df_test['BsmtFinType1'] = np.where((df_test['BsmtFinType1'].isna()) & (df_test["Foundation"] == 'PConc'), bsmt_pc_fin1_mode[0], df_test['BsmtFinType1'])

df_test['BsmtFinType2'] = np.where((df_test['BsmtFinType2'].isna()) & (df_test["Foundation"] == 'PConc'), bsmt_pc_fin2_mode[0], df_test['BsmtFinType2'])
df_test_bsmt_BT = df_test[(df_test["Foundation"] == "BrkTil")]
bsmt_BT = df_test_bsmt_BT[["Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]]
bsmt_BT
bsmt_BT_qual_mode = bsmt_BT["BsmtQual"].value_counts().index.tolist()

bsmt_BT_qual_mode[0]
bsmt_BT_Cond_mode = bsmt_BT["BsmtCond"].value_counts().index.tolist()

bsmt_BT_Cond_mode[0]
bsmt_BT_Ex_mode = bsmt_BT["BsmtExposure"].value_counts().index.tolist()

bsmt_BT_Ex_mode[0]
bsmt_BT_fin1_mode = bsmt_BT["BsmtFinType1"].value_counts().index.tolist()

bsmt_BT_fin1_mode[0]
bsmt_BT_fin2_mode = bsmt_BT["BsmtFinType2"].value_counts().index.tolist()

bsmt_BT_fin2_mode[0]
df_test['BsmtQual'] = np.where((df_test['BsmtQual'].isna()) & (df_test["Foundation"] == 'BrkTil'), bsmt_BT_qual_mode[0], df_test['BsmtQual'])

df_test['BsmtCond'] = np.where((df_test['BsmtCond'].isna()) & (df_test["Foundation"] == 'BrkTil'), bsmt_BT_Cond_mode[0], df_test['BsmtCond'])

df_test['BsmtExposure'] = np.where((df_test['BsmtExposure'].isna()) & (df_test["Foundation"] == 'BrkTil'), bsmt_BT_Ex_mode[0], df_test['BsmtExposure'])

df_test['BsmtFinType1'] = np.where((df_test['BsmtFinType1'].isna()) & (df_test["Foundation"] == 'BrkTil'), bsmt_BT_fin1_mode[0], df_test['BsmtFinType1'])

df_test['BsmtFinType2'] = np.where((df_test['BsmtFinType2'].isna()) & (df_test["Foundation"] == 'BrkTil'), bsmt_BT_fin2_mode[0], df_test['BsmtFinType2'])
df_test_bsmt_st = df_test[(df_test["Foundation"] == "Stone")]
bsmt_st = df_test_bsmt_st[["Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]]
bsmt_st
bsmt_st_qual_mode = bsmt_st["BsmtQual"].value_counts().index.tolist()

bsmt_st_qual_mode[0]
df_test['BsmtQual'] = np.where((df_test['BsmtQual'].isna()) & (df_test["Foundation"] == 'Stone'), bsmt_st_qual_mode[0], df_test['BsmtQual'])
DF_initial_observations(df_test)
garage = df_test[["GarageYrBlt","Neighborhood", "GarageType", "GarageFinish", "GarageQual", "GarageCond"]]

nulls_Garage = garage[garage.isnull().any(axis=1)]
nulls_Garage
df_test["GarageYrBlt"] = df_test["GarageYrBlt"].replace(np.nan, "unknown")

df_test["GarageType"] = df_test["GarageType"].replace(np.nan, "unknown")

df_test["GarageFinish"] = df_test["GarageFinish"].replace(np.nan, "unknown")

df_test["GarageQual"] = df_test["GarageQual"].replace(np.nan, "unknown")

df_test["GarageCond"] = df_test["GarageCond"].replace(np.nan, "unknown")
df_test["FireplaceQu"] = df_test["FireplaceQu"].replace(np.nan, "unknown")
df_test["SaleType"].value_counts()

sal_type_mode = df_test["SaleType"].value_counts().index.tolist()

sal_type_mode[0]
df_test["SaleType"] = df_test["SaleType"].replace(np.nan, sal_type_mode[0])

df_test['Utilities'].value_counts()
utilities_mode = df_test["Utilities"].value_counts().index.tolist()

utilities_mode[0]



df_test["Utilities"] = df_test["Utilities"].replace(np.nan, utilities_mode[0])
bsmt_fin1 = df_test["BsmtFinType1"].value_counts().index.tolist()

bsmt_fin1[0]
bsmt_fin2 = df_test["BsmtFinType2"].value_counts().index.tolist()

bsmt_fin2[0]
bsmt_unf = df_test["BsmtUnfSF"].value_counts().index.tolist()

bsmt_unf[0]
bsmt_totalsf = df_test["TotalBsmtSF"].value_counts().index.tolist()

bsmt_totalsf[0]
bsmt_fullbath = df_test["BsmtFullBath"].value_counts().index.tolist()

bsmt_fullbath[0]
bsmt_halfbath = df_test["BsmtHalfBath"].value_counts().index.tolist()

bsmt_halfbath[0]
kitchenqual = df_test["KitchenQual"].value_counts().index.tolist()

kitchenqual[0]
functional = df_test["Functional"].value_counts().index.tolist()

functional[0]
garage_cars = df_test["GarageCars"].value_counts().index.tolist()

garage_cars[0]
garage_area = df_test["GarageArea"].value_counts().index.tolist()

garage_area[0]
bsmt_finsf = df_test["BsmtFinSF1"].value_counts().index.tolist()

bsmt_finsf[0]
bsmt_finsf2 = df_test["BsmtFinSF2"].value_counts().index.tolist()

bsmt_finsf2[0]
df_test["BsmtFinType1"] = df_test["BsmtFinType1"].replace(np.nan, bsmt_fin1[0])

df_test["BsmtFinType2"] = df_test["BsmtFinType2"].replace(np.nan,bsmt_fin2[0])

df_test["BsmtFinSF1"] = df_test["BsmtFinSF1"].replace(np.nan, bsmt_finsf[0])

df_test["BsmtFinSF2"] = df_test["BsmtFinSF2"].replace(np.nan, bsmt_finsf2[0])
df_test["BsmtUnfSF"] = df_test["BsmtUnfSF"].replace(np.nan, bsmt_unf[0])

df_test["TotalBsmtSF"] = df_test["TotalBsmtSF"].replace(np.nan, bsmt_totalsf[0])

df_test["BsmtFullBath"] = df_test["BsmtFullBath"].replace(np.nan, bsmt_fullbath[0])

df_test["BsmtHalfBath"] = df_test["BsmtHalfBath"].replace(np.nan, bsmt_halfbath[0])

df_test["KitchenQual"] = df_test["KitchenQual"].replace(np.nan, kitchenqual[0])

df_test["Functional"] = df_test["Functional"].replace(np.nan, functional[0])

df_test["GarageCars"] = df_test["GarageCars"].replace(np.nan, garage_cars[0])

df_test["GarageArea"]= df_test["GarageArea"].replace(np.nan, garage_area[0])
df_test["BsmtFinType1"].isnull().sum()
# Custom function for initial checks

def DF_initial_observations(df):

    '''Gives basic details of columns in a dataframe : Data types, distinct values, NAs and sample'''

    if isinstance(df, pd.DataFrame):

        total_na=0

        for i in range(len(df.columns)):        

            total_na+= df.isna().sum()[i]

        print('Dimensions : %d rows, %d columns' % (df.shape[0],df.shape[1]))

        print("Total NA values : %d" % (total_na))

        print('%38s %10s     %10s %10s %15s' % ('Column name', ' Data Type', '# Distinct', ' NA values', ' Sample value'))

        for i in range(len(df.columns)):

            col_name = df.columns[i]

            sampl = df[col_name].sample(1)

            sampl.apply(pd.Categorical)

            sampl_p = str(sampl.iloc[0,])

            print('%38s %10s :   %10d  %10d %15s' % (df.columns[i],df.dtypes[i],df.nunique()[i],df.isna().sum()[i], sampl_p))

    else:

        print('Expected a DataFrame but got a %15s ' % (type(data)))
DF_initial_observations(df_test)
final_test = df_test.drop(['Alley','PoolQC','Fence','MiscFeature'], axis=1)
final_test.shape
final_test.describe()
final_test['LotShape'].value_counts()
final_test['LotShape'] = final_test["LotShape"].map({'Reg':3, 'IR1':2,'IR2':1,'IR3':0 }).astype(int)
final_test["MSZoning"].value_counts()
final_test["MSZoning"] = final_test["MSZoning"].replace("RL", 5)

final_test["MSZoning"] = final_test["MSZoning"].replace("RM", 4)

final_test["MSZoning"] = final_test["MSZoning"].replace("FV", 3)

final_test["MSZoning"] = final_test["MSZoning"].replace("RH", 2)

final_test["MSZoning"] = final_test["MSZoning"].replace("C (all)", 1)
final_test["Street"].value_counts()
final_test["Street"] = final_test["Street"].map({"Pave" :1 , "Grvl" : 0}).astype(int)
final_test['LandContour'].value_counts()
final_test["LandContour"] = final_test["LandContour"].map({"Lvl" : 3, "Bnk" : 2, "HLS" : 1, "Low" : 0}).astype(int)
final_test['Utilities'].value_counts()
final_test['Utilities'] = final_test["Utilities"].map({'AllPub':1 }).astype(int)
final_test['LandSlope'] = final_test["LandSlope"].map({'Gtl':0, 'Mod':1, 'Sev':2 }).astype(int)
final_test['BldgType'] = final_test["BldgType"].map({'1Fam':0, '2fmCon':1, 'Duplex':2, 'TwnhsE':3,

'Twnhs':4}).astype(int)
final_test['HouseStyle'].unique()
final_test['HouseStyle'] = final_test["HouseStyle"].map({'2Story':3, '1Story':0, '1.5Fin':2, '1.5Unf':1,

'SFoyer':6, 'SLvl':7, '2.5Unf':4, '2.5Fin':5  }).astype(int)
final_test["BsmtExposure"].value_counts()
final_test["ExterQual"] = final_test["ExterQual"].map({"Ex" : 3, "Gd" : 2, "TA" : 1, "Fa" : 0 }).astype(int)

final_test["ExterCond"] = final_test["ExterCond"].map({"Ex" : 4, "Gd" : 3, "TA" : 2, "Fa" : 1, "Po" : 0}).astype(int)

final_test["BsmtQual"] = final_test["BsmtQual"].map({"Ex" : 4, "Gd" : 3, "TA" : 2, "Fa" : 1, "unknown" : 0}).astype(int)

final_test["KitchenQual"] = final_test["KitchenQual"].map({"Ex" : 3, "Gd" : 2, "TA" : 1, "Fa" : 0 }).astype(int)
final_test["Functional"] = final_test["Functional"].map({"Typ" : 6, "Min1" : 5, "Maj1" : 2, "Min2" : 4, "Mod" : 3, "Maj2" : 1, "Sev" : 0}).astype(int)
final_test["FireplaceQu"] = final_test["FireplaceQu"].map({"unknown" : 0, "Ex" : 5, "Gd" : 4, "TA" : 3, "Fa" : 2, "Po" : 1}).astype(int)
final_test["GarageYrBlt"] = final_test["GarageYrBlt"].replace("unknown", 0).astype(float)
final_test["GarageFinish"] = final_test["GarageFinish"].map({"RFn" : 2, "Unf" : 1, "Fin" : 3, "unknown" : 0}).astype(int)
final_test["GarageQual"] = final_test["GarageQual"].map({"unknown" : 0, "Ex" : 5, "Gd" : 4, "TA" : 3, "Fa" : 2, "Po" : 1}).astype(int)
final_test["GarageCond"] = final_test["GarageCond"].map({"unknown" : 0, "Ex" : 5, "Gd" : 4, "TA" : 3, "Fa" : 2, "Po" : 1}).astype(int)
final_test["PavedDrive"] = final_test["PavedDrive"].map({"Y" : 2, "N" : 0, "P" : 1}).astype(int)
final_test["BsmtCond"] = final_test["BsmtCond"].map({ "Gd" : 4, "TA" : 3, "Fa" : 2, "Po" : 1, "unknown" : 0 }).astype(int)
final_test["BsmtExposure"] = final_test["BsmtExposure"].map({ "Gd" : 4, "No" : 1, "Mn" : 2, "Av" : 3 , "unknown" : 0}).astype(int)
final_test["BsmtFinType1"] = final_test["BsmtFinType1"].map({"GLQ" : 6, "ALQ" : 5, "Unf" : 1, "Rec" : 3, "BLQ" : 4, "LwQ" : 2, "unknown" : 0}).astype(int)
final_test["CentralAir"].value_counts()
final_test["BsmtFinType2"] = final_test["BsmtFinType2"].map({"GLQ" : 6, "ALQ" : 5, "Unf" : 1, "Rec" : 3, "BLQ" : 4, "LwQ" : 2, "unknown" : 0}).astype(int)
final_test["HeatingQC"] = final_test["HeatingQC"].map({"Ex" : 4, "Gd" : 3, "TA" : 2, "Fa" : 1, "Po" : 0}).astype(int)
final_test["CentralAir"] = final_test["CentralAir"].map({"Y" : 1, "N" : 0}).astype(int)
DF_initial_observations(final_test)
DF_initial_observations(final_train)
dummy_final_test = pd.get_dummies(final_test, columns= ["LotConfig", "Neighborhood", "Condition1", "Condition2", "RoofStyle" , "RoofMatl", "MasVnrType", "Foundation", "Exterior1st", "Exterior2nd", "Heating", "Electrical" ,"GarageType", "SaleType", "SaleCondition"])
final_test = pd.concat([final_test, dummy_final_test], axis = 1)
final_test = final_test.drop(["LotConfig", "Neighborhood", "Condition1", "Condition2",  "RoofStyle", "RoofMatl",  "MasVnrType", "Exterior1st", "Exterior2nd", "Heating", "Electrical", "Foundation", "GarageType", "SaleType", "SaleCondition"], axis = 1)
missing_cols = set(final_train)- set(final_test)

for col in missing_cols:

    final_test[col] = 0
final_test.shape
final_train.shape
DF_initial_observations(final_test)
DF_initial_observations(final_train)
x = final_train
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x,y)


x_train_scaled = x_train.copy()

x_val_scaled = x_val.copy()

x_test_scaled = final_test.copy()



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(x_train_scaled)

x_train_scaled_1 = scaler.transform(x_train_scaled)

x_val_scaled_1 = scaler.transform(x_val_scaled)

x_test_scaled_1 = scaler.transform(x_test_scaled)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



rf_parm = dict(n_estimators = [20,30,50, 70, 100], max_features = [10, 15, 20, 30])
rc = RandomForestRegressor()

rf_grid = GridSearchCV(estimator = rc, param_grid = rf_parm)
rf_grid.fit(x_train_scaled_1,y_train)
rf_grid.best_score_
rf_grid.best_params_
rc_best = RandomForestRegressor(n_estimators = 100,  max_features = 20)
rc_best.fit(x_train_scaled_1, y_train)

rc_tr_pred = rc_best.predict(x_train_scaled_1)

rc_val_pred = rc_best.predict(x_val_scaled_1)

rc_test_pred = rc_best.predict(x_test_scaled_1)
from sklearn.metrics import r2_score



print(r2_score(y_train, rc_tr_pred))

print(r2_score(y_val, rc_val_pred))    # Accuracy Scores for Train and Validation Data
print(rc_test_pred) #Final Sale Price Predictions for Test Data
from sklearn.ensemble import GradientBoostingRegressor



gbrt = GradientBoostingRegressor(random_state=42)

gbrt_grid = GridSearchCV(estimator=gbrt, param_grid=dict(n_estimators= [2,5,7,8,9,10,11,15,20], max_depth= [1,2,3,4,5,6,7,8,9,10], learning_rate= [0.01,0.1,1,10,100]))
gbrt_grid.fit(x_train_scaled_1,y_train)
gbrt_grid.best_score_
gbrt_grid.best_params_
gbrt_best = GradientBoostingRegressor(n_estimators=20, max_depth = 5, learning_rate=0.1, random_state=42)

gbrt_best.fit(x_train_scaled_1, y_train)
y_test_predict = gbrt_best.predict(x_test_scaled_1)
y_test_predict