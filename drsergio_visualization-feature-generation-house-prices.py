import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline  



project_location='../input/' 

path_to_train_data = ''.join([project_location, '/train.csv']) 
df = pd.read_csv(path_to_train_data)



print ('Number of houses: {0}'.format(df.shape[0]))

print ('Number of features: {0}'.format(df.shape[1]-2)) # 1 is Id (not a feature), another - is salary (is target variable)



df.head(20)
df.describe()
# How many columns with different datatypes are there?

df.get_dtype_counts()
df.info()
null_cols = df.columns[df.isnull().any()] # check whether there are any columns with missing values

df[null_cols].isnull().sum()
fig, ax = plt.subplots(1, 2, figsize = (11, 4))



b=df['LotFrontage'].groupby(df['Neighborhood']).agg({'N Houses': np.size, 'LotFrontage': np.median}).sort_values(['N Houses'], ascending=False).plot(kind='bar', ax=ax[0])

ax[0].set_title("Number of houses / LotFrontage VS Neighborhood",size=14)

plt.setp(ax[0].get_xticklabels(),size=13)

plt.setp(ax[0].get_yticklabels(),size=13)

plt.tight_layout()



m=df[df['LotFrontage'].isnull()].groupby('Neighborhood').size().sort_values(ascending=False).plot(kind='bar', ax=ax[1])

ax[1].set_title("Number of houses with na LotFrontage",size=14)

plt.setp(ax[1].get_xticklabels(),size=13)

plt.setp(ax[1].get_yticklabels(),size=13)

plt.tight_layout()



del b, m
# LotFrontage : NA most likely means no lot frontage

df.LotFrontage.fillna(0.0, inplace=True) # replace na with 0.0
df.Alley.fillna("None",inplace=True) # replace na with None
df.MasVnrType.fillna("None",inplace=True) # replace na with None

df.MasVnrArea.fillna(0,inplace=True) # replace na with 0.0
df.BsmtQual.fillna("No",inplace=True) # replace na with No

df.BsmtCond.fillna("No",inplace=True) # replace na with No

df.BsmtExposure.fillna("NA",inplace=True) # replace na with NA -> no basement

df.BsmtFinType1.fillna("No",inplace=True) # replace na with No

df.BsmtFinType2.fillna("No",inplace=True) # replace na with No
print (df.groupby('Electrical').size().sort_values(ascending=False))

df.Electrical.value_counts().plot(kind='bar')

plt.tick_params(labelsize=12)

plt.ylabel("Electrical system",size=12)

plt.tight_layout()



df.Electrical.fillna("SBrkr",inplace=True) # replace na with SBrkr
print (df.groupby('FireplaceQu').size().sort_values(ascending=False))

df.FireplaceQu.fillna('No',inplace=True) # replace na with No
df.GarageType.fillna("No",inplace=True) # replace na with No

df.GarageYrBlt.fillna("No",inplace=True) # replace na with No

df.GarageFinish.fillna("No",inplace=True) # replace na with No

df.GarageQual.fillna("No",inplace=True) # replace na with No

df.GarageCond.fillna("No",inplace=True) # replace na with No
df.PoolQC.fillna("No",inplace=True) # replace na with No
df.Fence.fillna("No",inplace=True) # replace na with No
df.MiscFeature.fillna("No",inplace=True) # replace na with No
null_cols = df.columns[df.isnull().any()] # check whether there are any columns with missing values

df[null_cols].isnull().sum()
df_original = df.copy() # saving preprocessed data set
print ('ExterQual:    {0}'.format(df['ExterQual'].unique()))

print ('ExterCond:    {0}'.format(df['ExterCond'].unique()))

print ('BsmtQual:     {0}'.format(df['BsmtQual'].unique()))

print ('BsmtCond:     {0}'.format(df['BsmtCond'].unique()))

print ('HeatingQC:    {0}'.format(df['HeatingQC'].unique()))

print ('KitchenQual:  {0}'.format(df['KitchenQual'].unique()))

print ('FireplaceQu:  {0}'.format(df['FireplaceQu'].unique()))

print ('GarageQual:   {0}'.format(df['GarageQual'].unique()))

print ('GarageCond:   {0}'.format(df['GarageCond'].unique()))

print ('PoolQC:       {0}'.format(df['PoolQC'].unique()))

print ('\n')

print ('BsmtExposure: {0}'.format(df['BsmtExposure'].unique()))

print ('BsmtFinType1: {0}'.format(df['BsmtFinType1'].unique()))

print ('BsmtFinType2: {0}'.format(df['BsmtFinType2'].unique()))

print ('Functional:   {0}'.format(df['Functional'].unique()))

print ('GarageFinish: {0}'.format(df['GarageFinish'].unique()))

print ('Fence:        {0}'.format(df['Fence'].unique()))
qual_dict = {'No': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

df['ExterQual']   = df['ExterQual'].map(qual_dict).astype(int)

df['ExterCond']   = df['ExterCond'].map(qual_dict).astype(int)

df['BsmtQual']    = df['BsmtQual'].map(qual_dict).astype(int)

df['BsmtCond']    = df['BsmtCond'].map(qual_dict).astype(int)

df['HeatingQC']   = df['HeatingQC'].map(qual_dict).astype(int)

df['KitchenQual'] = df['KitchenQual'].map(qual_dict).astype(int)

df['FireplaceQu'] = df['FireplaceQu'].map(qual_dict).astype(int)

df['GarageQual']  = df['GarageQual'].map(qual_dict).astype(int)

df['GarageCond']  = df['GarageCond'].map(qual_dict).astype(int)

df['PoolQC']      = df['PoolQC'].map(qual_dict).astype(int)



del qual_dict
df['BsmtExposure'] = df['BsmtExposure'].map({'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}).astype(int)



bsmt_fin_dict = {'No': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}

df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmt_fin_dict).astype(int)

df['BsmtFinType2'] = df['BsmtFinType2'].map(bsmt_fin_dict).astype(int)



df['Functional'] = df['Functional'].map({'No': 0, 'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 

                                         'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}).astype(int)



df['GarageFinish'] = df['GarageFinish'].map({'No': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}).astype(int)

df['Fence'] = df['Fence'].map({'No': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}).astype(int)



del bsmt_fin_dict
# Set up the matplotlib figure

corr = df.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

f, ax = plt.subplots(figsize=(11, 11))

sns.heatmap(corr, vmin=-0.8, vmax=0.8, square=True)

f.tight_layout()



del corr
corr_sale=df.corr()["SalePrice"]



fig, ax = plt.subplots(figsize = (6, 10))

corr_sale[np.argsort(corr_sale, axis=0)[::-1]].plot(kind='barh')

plt.tick_params(labelsize=12)

plt.ylabel("Pearson correlation",size=12)

plt.title('Correlated features with Sale Price', size=13)

plt.tight_layout()



del corr_sale
fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('MSSubClass').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("MSSubClass - type of dwelling involved in the sale",size=13)

plt.tight_layout()



sns.boxplot('MSSubClass', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 500000))

ax[1].set_title("SalePrice VS MSSubClass",size=13)

plt.tight_layout()
qual_dict = {20: 1, 30: 0, 40: 0, 45: 0, 50: 0, 60: 1, 70: 0, 75: 0, 

             80: 0, 85: 0, 90: 0, 120: 1, 150: 0, 160: 1, 180: 0, 190: 0}



df['MSSubClass'] = df['MSSubClass'].map(qual_dict).astype(object)



del qual_dict
print (df.groupby('MSSubClass').size().sort_values(ascending=False))



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('MSSubClass').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("MSSubClass - type of dwelling involved in the sale",size=13)

plt.tight_layout()



sns.boxplot('MSSubClass', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 500000))

ax[1].set_title("SalePrice VS MSSubClass",size=13)

plt.tight_layout()
print (df.groupby('MSZoning').size().sort_values(ascending=False))



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('MSZoning').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("MSZoning - general zoning classification of the sale",size=13)

plt.tight_layout()



sns.boxplot('MSZoning', 'SalePrice', data = df, order=['RL', 'RM', 'FV', 'RH', 'C (all)'], ax = ax[1]).set(ylim = (0, 500000))

ax[1].set_title("SalePrice VS MSZoning",size=13)

plt.tight_layout()
print (df["LotFrontage"].describe()) 

print ('-----------------------------')

print (df["LotFrontage"].skew())



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df['LotFrontage'].groupby(df['Neighborhood']).agg({'N Houses': np.size, 'LotFrontage': np.median}).sort_values(['N Houses'], ascending=False).plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=90,size=10)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_title("Number of houses / LotFrontage VS Neighborhood",size=13)

plt.tight_layout()



ax[1].scatter(range(df.shape[0]), df["LotFrontage"].values,color='orange')

ax[1].set_title("Distribution of LotFrontage", size=13)

ax[1].set_xlabel("Number of Occurences", size=12)

ax[1].set_ylabel("LotFrontage, Feet", size=12)

plt.setp(ax[1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[1].get_yticklabels(),size=12)

plt.tight_layout()
print (df["LotArea"].describe()) 

print ('-----------------------------')

print (df["LotArea"].skew())



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



ax[0].scatter(range(df.shape[0]), df["LotArea"].values,color='orange')

ax[0].set_title("Distribution of LotArea", size=13)

ax[0].set_xlabel("Number of Occurences", size=12)

ax[0].set_ylabel("LotArea, Square Feet", size=12)

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

plt.tight_layout()
print (df.groupby('Street').size().sort_values(ascending=False))



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('Street').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("Type of road access to property",size=13)

plt.tight_layout()



sns.boxplot('Street', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 500000))

ax[1].set_title("SalePrice VS Street",size=13)

plt.tight_layout()
print (df.groupby('Alley').size().sort_values(ascending=False))



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('Alley').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("Type of alley access to property",size=13)

plt.tight_layout()



sns.boxplot('Alley', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 500000))

ax[1].set_title("SalePrice VS Alley",size=13)

plt.tight_layout()
print (df.groupby('LotShape').size().sort_values(ascending=False))



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('LotShape').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("General shape of the property",size=13)

plt.tight_layout()



sns.boxplot('LotShape', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 500000))

ax[1].set_title("SalePrice VS LotShape",size=13)

plt.tight_layout()
print (df.groupby('LandContour').size().sort_values(ascending=False))



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('LandContour').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("Flatness of the property",size=13)

plt.tight_layout()



sns.boxplot('LandContour', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 500000))

ax[1].set_title("SalePrice VS LandContour",size=13)

plt.tight_layout()
print (df.groupby('Utilities').size().sort_values(ascending=False))



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('Utilities').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("Type of utilities available",size=13)

plt.tight_layout()



sns.boxplot('Utilities', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 500000))

ax[1].set_title("SalePrice VS Utilities",size=13)

plt.tight_layout()
print (df.groupby('LotConfig').size().sort_values(ascending=False))



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('LotConfig').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("Lot Configuration",size=13)

plt.tight_layout()



sns.boxplot('LotConfig', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 500000))

ax[1].set_title("SalePrice VS LotConfig",size=13)

plt.tight_layout()
print (df.groupby('LandSlope').size().sort_values(ascending=False))



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('LandSlope').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("Slope of property",size=13)

plt.tight_layout()



sns.boxplot('LandSlope', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 400000))

ax[1].set_title("SalePrice VS LandSlope",size=13)

plt.tight_layout()
sns.boxplot(x = 'Neighborhood', y = 'SalePrice',  data = df)

xt = plt.xticks(rotation=90)

plt.tight_layout()
df_saleneighb = df['SalePrice'].groupby(df['Neighborhood']).agg({'Neighborhood': np.size, 'SalePrice': np.median}).sort_values(['SalePrice'], ascending=False).copy()



print (df['SalePrice'].groupby(df['Neighborhood']).agg({'Neighborhood': np.size, 'SalePrice': np.median}).sort_values(['SalePrice'], ascending=False))



del df_saleneighb
print (df.groupby('Condition1').size().sort_values(ascending=False))

print ('--------------')

print (df.groupby('Condition2').size().sort_values(ascending=False))



fig, ax = plt.subplots(2, 2, figsize = (11, 8))



df.groupby('Condition1').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][0])

plt.setp(ax[0][0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0][0].get_yticklabels(),size=12)

ax[0][0].set_ylabel('Number of houses',size=12)

ax[0][0].set_title("Condition 1 - proximity to various conditions",size=13)

plt.tight_layout()



df.groupby('Condition2').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][1]) 

plt.setp(ax[0][1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0][1].get_yticklabels(),size=12)

ax[0][1].set_ylabel('Number of houses',size=12)

ax[0][1].set_title("Condition 2 - proximity to various conditions",size=13)

plt.tight_layout()



sns.boxplot('Condition1', 'SalePrice', data = df, ax = ax[1][0]).set(ylim = (0, 400000))

ax[1][0].set_title("SalePrice VS Condition1",size=13)

plt.setp(ax[1][0].get_xticklabels(),rotation=0,size=12)

plt.tight_layout()



sns.boxplot('Condition2', 'SalePrice', data = df, ax = ax[1][1]).set(ylim = (0, 400000))

ax[1][1].set_title("SalePrice VS Condition2",size=13)

plt.setp(ax[1][1].get_xticklabels(),rotation=0,size=12)

plt.tight_layout()
print (df.groupby('BldgType').size().sort_values(ascending=False))

print ('--------------')

print (df.groupby('HouseStyle').size().sort_values(ascending=False))



fig, ax = plt.subplots(2, 2, figsize = (11, 8))



df.groupby('BldgType').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][0])

plt.setp(ax[0][0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0][0].get_yticklabels(),size=12)

ax[0][0].set_ylabel('Number of houses',size=12)

ax[0][0].set_title("Type of dwelling",size=13)

plt.tight_layout()



df.groupby('HouseStyle').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][1])

plt.setp(ax[0][1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0][1].get_yticklabels(),size=12)

ax[0][1].set_ylabel('Number of houses',size=12)

ax[0][1].set_title("Style of dwelling",size=13)

plt.tight_layout()



sns.boxplot('BldgType', 'SalePrice', data = df, ax = ax[1][0]).set(ylim = (0, 400000))

ax[1][0].set_title("SalePrice VS BldgType",size=13)

plt.setp(ax[1][0].get_xticklabels(),rotation=0,size=12)

plt.tight_layout()



sns.boxplot('HouseStyle', 'SalePrice', data = df, ax = ax[1][1]).set(ylim = (0, 400000))

ax[1][1].set_title("SalePrice VS HouseStyle",size=13)

plt.setp(ax[1][1].get_xticklabels(),rotation=0,size=12)

plt.tight_layout()
print (df.groupby('OverallQual').size())

print ('--------------')

print (df.groupby('OverallCond').size())



overallqual_x=df.groupby('OverallQual').size().index.get_values()

overallqual_y=df.groupby('OverallQual').size().get_values()



overallcond_x=df.groupby('OverallCond').size().index.get_values()

overallcond_y=df.groupby('OverallCond').size().get_values()



plt.bar(overallqual_x-0.1, overallqual_y,width=0.2,color='orange',align='center',label="OverallQual")

plt.bar(overallcond_x+0.1, overallcond_y,width=0.2,color='g',align='center',label="OverallCond")

plt.legend(prop={'size':12})

plt.title("Overall quality / Overall condition",size=13)

plt.xticks([1,2,3,4,5,6,7,8,9,10], size=12)

plt.ylabel('Number of houses',size=12)

plt.yticks(size=12)

plt.tight_layout()



del overallqual_x, overallqual_y, overallcond_x, overallcond_y
fig, ax = plt.subplots(2, 1, figsize = (11, 6))

sns.countplot(x = 'YearBuilt', data = df, ax=ax[0])

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("Year of construction",size=13)

plt.setp(ax[0].get_xticklabels(),rotation=90,size=9)

plt.setp(ax[0].get_yticklabels(),size=12)

plt.tight_layout()



sns.countplot(x = 'YearRemodAdd', data = df, ax=ax[1])

ax[1].set_ylabel('Number of houses',size=12)

ax[1].set_title("Year of remodel",size=13)

plt.setp(ax[1].get_xticklabels(),rotation=90,size=9)

plt.setp(ax[1].get_yticklabels(),size=12)

plt.tight_layout()
df_built = (df['YearBuilt'].groupby(pd.cut(df['YearBuilt'], [1870,1890,1910,1930,1950,1970,1990,2010], right=False))

                     .count())

df_remod = (df['YearRemodAdd'].groupby(pd.cut(df['YearRemodAdd'], [1870,1890,1910,1930,1950,1970,1990,2010], right=False))

                     .count())



fig, ax = plt.subplots(2, 1, figsize = (11, 6))



df_built.plot(kind='bar',ax=ax[0])

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("Year of construction",size=13)

plt.setp(ax[0].get_xticklabels(),rotation=45,size=11)

plt.setp(ax[0].get_yticklabels(),size=12)

plt.tight_layout()



df_remod.plot(kind='bar',ax=ax[1])

ax[1].set_ylabel('Number of houses',size=12)

ax[1].set_title("Year of remodel",size=13)

plt.setp(ax[1].get_xticklabels(),rotation=45,size=11)

plt.setp(ax[1].get_yticklabels(),size=12)

plt.tight_layout()



del df_built, df_remod
df_4 = df[["YearBuilt", "YearRemodAdd"]][(df["YearBuilt"] == df["YearRemodAdd"])]

df_5 = df[["YearBuilt", "YearRemodAdd"]][(df["YearBuilt"] != df["YearRemodAdd"])]



print ('N houses were not altered: {0}'.format(df_4.shape[0]))

print ('N houses were remodeled: {0}'.format(df_5.shape[0]))



df_6 = pd.DataFrame({'Indx': df.index, 'ConstrRemod': np.abs(df.YearBuilt - df.YearRemodAdd)})

df_6 = df_6[(df_6.ConstrRemod>0) & (df_6.ConstrRemod<=80)] # selecting houses that were remodeled and not later than 80 years after construction



fig, ax = plt.subplots(figsize = (11, 4))

b = df_6.ConstrRemod.value_counts().sort_index().plot(kind='bar')

print (df_6.ConstrRemod.value_counts().sort_index().head(10))



plt.title('YearRemodAdd - YearBuilt', fontsize=16, color='black') 

plt.xlabel('Number of houses',fontsize=14, color='black') 

plt.ylabel('YearRemodAdd - YearBuilt', fontsize=14, color='black') 

plt.tight_layout()



del df_4, df_5, df_6, b
print (df.groupby('RoofStyle').size().sort_values(ascending=False))

print ('--------------')

print (df.groupby('RoofMatl').size().sort_values(ascending=False))



fig, ax = plt.subplots(2, 2, figsize = (11, 8))



df.groupby('RoofStyle').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][0])

plt.setp(ax[0][0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0][0].get_yticklabels(),size=12)

ax[0][0].set_ylabel('Number of houses',size=12)

ax[0][0].set_title("Type of roof",size=13)

plt.tight_layout()



df.groupby('RoofMatl').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][1])

plt.setp(ax[0][1].get_xticklabels(),rotation=90,size=12)

plt.setp(ax[0][1].get_yticklabels(),size=12)

ax[0][1].set_ylabel('Number of houses',size=12)

ax[0][1].set_title("Roof material",size=13)

plt.tight_layout()



sns.boxplot('RoofStyle', 'SalePrice', data = df, ax = ax[1][0]).set(ylim = (0, 500000))

ax[1][0].set_title("SalePrice VS RoofStyle",size=13)

plt.setp(ax[1][0].get_xticklabels(),rotation=0,size=12)

plt.tight_layout()



sns.boxplot('RoofMatl', 'SalePrice', data = df, ax = ax[1][1]).set(ylim = (0, 500000))

ax[1][1].set_title("SalePrice VS RoofMatl",size=13)

plt.setp(ax[1][1].get_xticklabels(),rotation=90,size=12)

plt.tight_layout()
print (df.groupby('Exterior1st').size().sort_values(ascending=False))

print ('--------------')

print (df.groupby('Exterior2nd').size().sort_values(ascending=False))



fig, ax = plt.subplots(2, 2, figsize = (11, 8))



df.groupby('Exterior1st').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][0])

plt.setp(ax[0][0].get_xticklabels(),rotation=45,size=12)

plt.setp(ax[0][0].get_yticklabels(),size=12)

ax[0][0].set_ylabel('Number of houses',size=12)

ax[0][0].set_title("Exterior1st - exterior covering on house",size=13)

plt.tight_layout()



df.groupby('Exterior2nd').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][1]) 

plt.setp(ax[0][1].get_xticklabels(),rotation=45,size=12)

plt.setp(ax[0][1].get_yticklabels(),size=12)

ax[0][1].set_ylabel('Number of houses',size=12)

ax[0][1].set_title("Exterior2nd - exterior covering on house (if more than 1 material)",size=13)

plt.tight_layout()



sns.boxplot('Exterior1st', 'SalePrice', data = df, ax = ax[1][0]).set(ylim = (0, 400000))

ax[1][0].set_title("SalePrice VS Exterior1st",size=13)

plt.setp(ax[1][0].get_xticklabels(),rotation=45,size=12)

plt.tight_layout()



sns.boxplot('Exterior2nd', 'SalePrice', data = df, ax = ax[1][1]).set(ylim = (0, 400000))

ax[1][1].set_title("SalePrice VS Exterior2nd",size=13)

plt.setp(ax[1][1].get_xticklabels(),rotation=45,size=12)

plt.tight_layout()
print (df.groupby('MasVnrType').size().sort_values(ascending=False))

print ('-----------------------------')

print (df["MasVnrArea"].skew())



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('MasVnrType').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("Masonry veneer type",size=13)

plt.tight_layout()



ax[1].scatter(range(df.shape[0]), df["MasVnrArea"].values,color='orange')

ax[1].set_title("Distribution of MasVnrArea", size=13)

ax[1].set_xlabel("Number of Occurences", size=12)

ax[1].set_ylabel("MasVnrArea, Square Feet", size=12)

plt.setp(ax[1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[1].get_yticklabels(),size=12)

plt.tight_layout()
print (df.groupby('ExterQual').size())

print ('--------------')

print (df.groupby('ExterCond').size())



exterqual_x=df.groupby('ExterQual').size().index.get_values()

exterqual_y=df.groupby('ExterQual').size().get_values()



extercond_x=df.groupby('ExterCond').size().index.get_values()

extercond_y=df.groupby('ExterCond').size().get_values()



plt.bar(exterqual_x-0.1, exterqual_y,width=0.2,color='orange',align='center',label="ExterQual")

plt.bar(extercond_x+0.1, extercond_y,width=0.2,color='g',align='center',label="ExterCond")

plt.legend(prop={'size':12})

plt.title("ExterQual / ExterCond",size=13)

plt.xticks([1,2,3,4,5], size=12)

plt.ylabel('Number of houses',size=12)

plt.yticks(size=12)

plt.tight_layout()



del exterqual_x, exterqual_y, extercond_x, extercond_y
print (df.groupby('Foundation').size().sort_values(ascending=False))



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('Foundation').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("Type of foundation",size=13)

plt.tight_layout()



sns.boxplot('Foundation', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 800000))

ax[1].set_title("SalePrice VS Foundation",size=13)

plt.tight_layout()
print (df.groupby('BsmtQual').size())

print ('--------------')

print (df.groupby('BsmtCond').size())



bsmtqual_x=df.groupby('BsmtQual').size().index.get_values()

bsmtqual_y=df.groupby('BsmtQual').size().get_values()



bsmtcond_x=df.groupby('BsmtCond').size().index.get_values()

bsmtcond_y=df.groupby('BsmtCond').size().get_values()



fig, ax = plt.subplots(2, 2, figsize = (11, 8))



ax[0][0].bar(bsmtqual_x-0.1, bsmtqual_y,width=0.2,color='orange',align='center',label="BsmtQual")

ax[0][0].bar(bsmtcond_x+0.1, bsmtcond_y,width=0.2,color='g',align='center',label="BsmtCond")

ax[0][0].legend(prop={'size':12})

ax[0][0].set_title("BsmtQual / BsmtCond condition",size=13)

ax[0][0].set_xticks([0,1,2,3,4,5])

ax[0][0].set_ylabel('Number of houses',size=12)

plt.setp(ax[0][0].get_xticklabels(), rotation=0,size=12)

plt.setp(ax[0][0].get_yticklabels(), rotation=0,size=12)

plt.tight_layout()



sns.boxplot('BsmtQual', 'SalePrice', data = df, ax = ax[0][1]).set(ylim = (0, 500000))

ax[0][1].set_title("SalePrice VS BsmtQual",size=13)

plt.tight_layout()



sns.boxplot('BsmtCond', 'SalePrice', data = df, ax = ax[1][0]).set(ylim = (0, 500000))

ax[1][0].set_title("SalePrice VS BsmtCond",size=13)

plt.tight_layout()



del bsmtqual_x, bsmtqual_y, bsmtcond_x, bsmtcond_y
print (df.groupby('BsmtExposure').size())



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('BsmtExposure').size().plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("BsmtExposure - refers to walkout or garden level walls",size=13)

plt.tight_layout()



sns.boxplot('BsmtExposure', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 800000))

ax[1].set_title("SalePrice VS BsmtExposure",size=13)

plt.tight_layout()
print (df.groupby('BsmtFinType1').size())

print ('--------------')

print (df.groupby('BsmtFinType2').size())



BsmtFinType1_x=df.groupby('BsmtFinType1').size().index.get_values()

BsmtFinType1_y=df.groupby('BsmtFinType1').size().get_values()



BsmtFinType2_x=df.groupby('BsmtFinType2').size().index.get_values()

BsmtFinType2_y=df.groupby('BsmtFinType2').size().get_values()



plt.bar(BsmtFinType1_x-0.1, BsmtFinType1_y,width=0.2,color='orange',align='center',label="BsmtFinType1")

plt.bar(BsmtFinType2_x+0.1, BsmtFinType2_y,width=0.2,color='g',align='center',label="BsmtFinType2")

plt.legend(prop={'size':12})

plt.title("BsmtFinType1 / BsmtFinType2",size=13)

plt.xticks([0,1,2,3,4,5,6], size=12)

plt.ylabel('Number of houses',size=12)

plt.yticks(size=12)

plt.tight_layout()



del BsmtFinType1_x, BsmtFinType1_y, BsmtFinType2_x, BsmtFinType2_y
fig, ax = plt.subplots(1, 2, figsize = (11, 4))



sns.boxplot('BsmtFinType1', 'SalePrice', data = df, ax = ax[0]).set(ylim = (0, 800000))

ax[0].set_title("SalePrice VS BsmtFinType1",size=13)

plt.tight_layout()



sns.boxplot('BsmtFinType2', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 800000))

ax[1].set_title("SalePrice VS BsmtFinType2",size=13)

plt.tight_layout()
print ('Skew BsmtFinSF1: {0}'.format(df["BsmtFinSF1"].skew())) 

print ('-----------------------------------------------------------')

print ('Skew BsmtFinSF2: {0}'.format(df["BsmtFinSF2"].skew())) 



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



ax[0].scatter(range(df.shape[0]), df["BsmtFinSF1"].values,color='orange')

ax[0].set_title("BsmtFinSF1 - type 1 finished basement area", size=13)

ax[0].set_xlabel("Number of Occurences", size=12)

ax[0].set_ylabel("BsmtFinSF1, Square Feet", size=12)

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

plt.tight_layout()



ax[1].scatter(range(df.shape[0]), df["BsmtFinSF2"].values,color='orange')

ax[1].set_title("BsmtFinSF2 - type 2 finished basement area", size=13)

ax[1].set_xlabel("Number of Occurences", size=12)

ax[1].set_ylabel("BsmtFinSF2, Square Feet", size=12)

plt.setp(ax[1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[1].get_yticklabels(),size=12)

plt.tight_layout()
fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df[['BsmtFinSF1','BsmtFinType1']].groupby('BsmtFinType1').agg({'BsmtFinType1': np.size, 'BsmtFinSF1': np.median}).plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses / Area, Square Feet',size=12)

ax[0].set_title("BsmtFinType1 / BsmtFinSF1.median()",size=13)

plt.tight_layout()



df[['BsmtFinSF2','BsmtFinType2']].groupby('BsmtFinType2').agg({'BsmtFinType2': np.size, 'BsmtFinSF2': np.median}).plot(kind='bar', ax=ax[1])

plt.setp(ax[1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[1].get_yticklabels(),size=12)

ax[1].set_ylabel('Number of houses / Area, Square Feet',size=12)

ax[1].set_title("BsmtFinType2 / BsmtFinSF2.median()",size=13)

plt.tight_layout()
print ('Skew BsmtUnfSF: {0}'.format(df["BsmtUnfSF"].skew())) 

print ('-----------------------------------------------------------')

print ('Skew TotalBsmtSF: {0}'.format(df["TotalBsmtSF"].skew())) 



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



ax[0].scatter(range(df.shape[0]), df["BsmtUnfSF"].values,color='orange')

ax[0].set_title("BsmtUnfSF - unfinished square feet of basement area", size=13)

ax[0].set_xlabel("Number of Occurences", size=12)

ax[0].set_ylabel("BsmtUnfSF, Square Feet", size=12)

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

plt.tight_layout()



ax[1].scatter(range(df.shape[0]), df["TotalBsmtSF"].values,color='orange')

ax[1].set_title("TotalBsmtSF - total square feet of basement area", size=13)

ax[1].set_xlabel("Number of Occurences", size=12)

ax[1].set_ylabel("TotalBsmtSF, Square Feet", size=12)

plt.setp(ax[1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[1].get_yticklabels(),size=12)

plt.tight_layout()



print ('\n')

print (df[['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']].head(10)) 
print (df.groupby('Heating').size().sort_values(ascending=False))

print ('--------------')

print (df.groupby('HeatingQC').size())



fig, ax = plt.subplots(3, 2, figsize = (11, 12))



df.groupby('Heating').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][0])

plt.setp(ax[0][0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0][0].get_yticklabels(),size=12)

ax[0][0].set_ylabel('Number of houses',size=12)

ax[0][0].set_title("Heating - type of heating",size=13)

plt.tight_layout()



df.groupby('HeatingQC').size().plot(kind='bar', ax=ax[0][1]) 

plt.setp(ax[0][1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0][1].get_yticklabels(),size=12)

ax[0][1].set_ylabel('Number of houses',size=12)

ax[0][1].set_title("HeatingQC - heating quality",size=13)

plt.tight_layout()



sns.boxplot('Heating', 'SalePrice', data = df, ax = ax[1][0]).set(ylim = (0, 500000))

ax[1][0].set_title("SalePrice VS type of heating",size=13)

plt.tight_layout()



sns.violinplot('HeatingQC', 'SalePrice', data = df, ax = ax[1][1]).set(ylim = (0, 500000))

ax[1][1].set_title("SalePrice VS heating quality",size=13)

plt.tight_layout()



sns.stripplot("HeatingQC", "SalePrice",data = df,hue = 'CentralAir', jitter=True, split=True, ax = ax[2][0]).set(ylim = (0, 600000))

ax[2][0].set_title("Sale Price vs Heating Quality / Air Conditioning",size=13)

plt.tight_layout()
print (df.groupby('CentralAir').size().sort_values(ascending=False))



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('CentralAir').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("CentralAir - central air conditioning",size=13)

plt.tight_layout()



sns.boxplot('CentralAir', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 400000))

ax[1].set_title("SalePrice VS CentralAir",size=13)

plt.tight_layout()
print (df.groupby('Electrical').size().sort_values(ascending=False))



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('Electrical').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("Electrical - type of electrical system",size=13)

plt.tight_layout()



sns.boxplot('Electrical', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 400000))

ax[1].set_title("SalePrice VS Electrical",size=13)

plt.tight_layout()
print ('Skew 1stFlrSF: {0}'.format(df["1stFlrSF"].skew())) 

print ('Skew 2ndFlrSF: {0}'.format(df["2ndFlrSF"].skew())) 

print ('-----------------------------------------------------------')



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



ax[0].scatter(range(df.shape[0]), df["1stFlrSF"].values,color='orange')

ax[0].set_title("1stFlrSF - first floor area", size=13)

ax[0].set_xlabel("Number of Occurences", size=12)

ax[0].set_ylabel("1stFlrSF, Square Feet", size=12)

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

plt.tight_layout()



ax[1].scatter(range(df.shape[0]), df["2ndFlrSF"].values,color='orange')

ax[1].set_title("2ndFlrSF - second floor area", size=13)

ax[1].set_xlabel("Number of Occurences", size=12)

ax[1].set_ylabel("2ndFlrSF, Square Feet", size=12)

plt.setp(ax[1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[1].get_yticklabels(),size=12)

plt.tight_layout()
print ('Number of houses having LowQualFinSF=0: {0}'.format((df['LowQualFinSF'][df['LowQualFinSF'] == 0]).count()))

print ('--------------------------------------------------------------')

print ('Skew LowQualFinSF: {0}'.format(df["LowQualFinSF"].skew())) 

print ('Skew GrLivArea:{0}'.format(df["GrLivArea"].skew())) 



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



ax[0].scatter(range(df.shape[0]), df["LowQualFinSF"].values,color='orange')

ax[0].set_title("LowQualFinSF - low quality finished living area", size=13)

ax[0].set_xlabel("Number of Occurences", size=12)

ax[0].set_ylabel("LowQualFinSF, Square Feet", size=12)

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

plt.tight_layout()



ax[1].scatter(range(df.shape[0]), df["GrLivArea"].values,color='orange')

ax[1].set_title("GrLivArea - above grade (ground) living area", size=13)

ax[1].set_xlabel("Number of Occurences", size=12)

ax[1].set_ylabel("GrLivArea, Square Feet", size=12)

plt.setp(ax[1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[1].get_yticklabels(),size=12)

plt.tight_layout()



print ('\n')

print (df[['1stFlrSF','2ndFlrSF', 'LowQualFinSF','GrLivArea']].head(10))
print (df.groupby('BsmtFullBath').size())

print ('------------')

print (df.groupby('BsmtHalfBath').size())

print ('------------')

print (df.groupby('FullBath').size())

print ('------------')

print (df.groupby('HalfBath').size())



fig, ax = plt.subplots(2, 2, figsize = (11, 8))



df.groupby('BsmtFullBath').size().plot(kind='bar', ax=ax[0][0])

plt.setp(ax[0][0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0][0].get_yticklabels(),size=12)

ax[0][0].set_ylabel('Number of houses',size=12)

ax[0][0].set_title("BsmtFullBath - basement full bathrooms",size=13)

plt.tight_layout()



df.groupby('BsmtHalfBath').size().plot(kind='bar', ax=ax[0][1]) 

plt.setp(ax[0][1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0][1].get_yticklabels(),size=12)

ax[0][1].set_ylabel('Number of houses',size=12)

ax[0][1].set_title("BsmtHalfBath - basement half bathrooms",size=13)

plt.tight_layout()



df.groupby('FullBath').size().plot(kind='bar', ax=ax[1][0])

plt.setp(ax[1][0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[1][0].get_yticklabels(),size=12)

ax[1][0].set_ylabel('Number of houses',size=12)

ax[1][0].set_title("FullBath - full bathrooms above grade",size=13)

plt.tight_layout()



df.groupby('HalfBath').size().plot(kind='bar', ax=ax[1][1]) 

plt.setp(ax[1][1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[1][1].get_yticklabels(),size=12)

ax[1][1].set_ylabel('Number of houses',size=12)

ax[1][1].set_title("HalfBath - half bathrooms above grade",size=13)

plt.tight_layout()
print (df.groupby('BedroomAbvGr').size())



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('BedroomAbvGr').size().plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("Bedroom - number of bedrooms above grade",size=13)

plt.tight_layout()



sns.boxplot('BedroomAbvGr', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 500000))

ax[1].set_title("SalePrice VS BedroomAbvGr",size=13)

plt.tight_layout()
print (df.groupby('KitchenAbvGr').size())

print ('--------------')

print (df.groupby('KitchenQual').size())



fig, ax = plt.subplots(2, 2, figsize = (11, 8))



df.groupby('KitchenAbvGr').size().plot(kind='bar', ax=ax[0][0])

plt.setp(ax[0][0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0][0].get_yticklabels(),size=12)

ax[0][0].set_ylabel('Number of houses',size=12)

ax[0][0].set_title("KitchenAbvGr - number of kitchen",size=13)

plt.tight_layout()



df.groupby('KitchenQual').size().plot(kind='bar', ax=ax[0][1]) 

plt.setp(ax[0][1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0][1].get_yticklabels(),size=12)

ax[0][1].set_ylabel('Number of houses',size=12)

ax[0][1].set_title("KitchenQual - kitchen quality",size=13)

plt.tight_layout()



sns.boxplot('KitchenAbvGr', 'SalePrice', data = df, ax = ax[1][0]).set(ylim = (0, 500000))

ax[1][0].set_title("SalePrice VS KitchenAbvGr",size=13)

plt.tight_layout()



sns.boxplot('KitchenQual', 'SalePrice', data = df, ax = ax[1][1]).set(ylim = (0, 500000))

ax[1][1].set_title("SalePrice VS KitchenQual",size=13)

plt.tight_layout()
fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('TotRmsAbvGrd').size().plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("TotRmsAbvGrd - number of rooms above grade",size=13)

plt.tight_layout()



sns.boxplot('TotRmsAbvGrd', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 500000))

ax[1].set_title("SalePrice VS TotRmsAbvGrd",size=13)

plt.tight_layout()
print (df.groupby('Functional').size())

print ('--------------')



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('Functional').size().plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("Functional - home functionality",size=13)

plt.tight_layout()



sns.boxplot('Functional', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 400000))

ax[1].set_title("SalePrice VS Functional",size=13)

plt.tight_layout()
all_df = df.copy()

all_df['Functional'] = all_df.Functional.replace(

      {1 : 1, 2 : 1, 3 : 1, 4 : 1, 5 : 2, 6 : 2, 7 : 2, 8 : 3})

    

fig, ax = plt.subplots(1, 2, figsize = (11, 4))



all_df.groupby('Functional').size().plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("Functional - home functionality",size=13)

plt.tight_layout()



sns.boxplot('Functional', 'SalePrice', data = all_df, ax = ax[1]).set(ylim = (0, 400000))

ax[1].set_title("SalePrice VS Functional",size=13)

plt.tight_layout()



del all_df
print (df.groupby('Fireplaces').size())

print ('--------------')

print (df.groupby('FireplaceQu').size())



fig, ax = plt.subplots(2, 2, figsize = (11, 8))



df.groupby('Fireplaces').size().plot(kind='bar', ax=ax[0][0])

plt.setp(ax[0][0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0][0].get_yticklabels(),size=12)

ax[0][0].set_ylabel('Number of houses',size=12)

ax[0][0].set_title("Fireplaces - number of fireplaces",size=13)

plt.tight_layout()



df.groupby('FireplaceQu').size().plot(kind='bar', ax=ax[0][1]) 

plt.setp(ax[0][1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0][1].get_yticklabels(),size=12)

ax[0][1].set_ylabel('Number of houses',size=12)

ax[0][1].set_title("FireplaceQu - fireplace quality",size=13)

plt.tight_layout()



sns.boxplot('Fireplaces', 'SalePrice', data = df, ax = ax[1][0]).set(ylim = (0, 500000))

ax[1][0].set_title("SalePrice VS Fireplaces",size=13)

plt.tight_layout()



sns.boxplot('FireplaceQu', 'SalePrice', data = df, ax = ax[1][1]).set(ylim = (0, 500000))

ax[1][1].set_title("SalePrice VS FireplaceQu",size=13)

plt.tight_layout()
print (df.groupby('GarageType').size())

print ('--------------')



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('GarageType').size().plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("GarageType - garage location",size=13)

plt.tight_layout()



sns.boxplot('GarageType', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 400000))

ax[1].set_title("SalePrice VS GarageType",size=13)

plt.tight_layout()
fig, ax = plt.subplots(4, 1, figsize = (11, 16))



df_built = (df.loc[df['GarageYrBlt'] != 'No','GarageYrBlt'].groupby(df['GarageYrBlt']).count())

df_built.index = df_built.index.map(int)



df_built.plot(kind='bar',ax=ax[0])

ax[0].set_ylabel('Number of garages',size=12)

ax[0].set_title("Year of garage construction",size=13)

plt.setp(ax[0].get_xticklabels(),rotation=90,size=11)

plt.setp(ax[0].get_yticklabels(),size=12)

plt.tight_layout()



# Making intervals

df_built = df.loc[df['GarageYrBlt'] != 'No','GarageYrBlt'].copy()

df_built.index = df_built.index.map(int)



df_built = (df_built.groupby(pd.cut(df_built.values, [1870,1890,1910,1930,1950,1970,1990,2010], right=False))

                     .count()) # 20 years each bin

print (df_built)



df_built.plot(kind='bar',ax=ax[1])

ax[1].set_ylabel('Number of garages',size=12)

ax[1].set_title("Year of garage construction",size=13)

plt.setp(ax[1].get_xticklabels(),rotation=45,size=11)

plt.setp(ax[1].get_yticklabels(),size=12)

plt.tight_layout()



df.groupby('GarageFinish').size().plot(kind='bar', ax=ax[2])

plt.setp(ax[2].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[2].get_yticklabels(),size=12)

ax[2].set_ylabel('Number of garages',size=12)

ax[2].set_title("GarageFinish - interior finish of the garage",size=13)

plt.tight_layout()



sns.boxplot('GarageFinish', 'SalePrice', data = df, ax = ax[3]).set(ylim = (0, 400000))

ax[3].set_title("SalePrice VS GarageFinish",size=13)

plt.tight_layout()



del df_built
print (df.groupby('GarageCars').size().sort_values(ascending=False))

print ('-----------------------------')

print (df["GarageArea"].skew())



fig, ax = plt.subplots(2, 2, figsize = (11, 8))



df.groupby('GarageCars').size().sort_values(ascending=False).plot(kind='bar', ax=ax[0][0])

plt.setp(ax[0][0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0][0].get_yticklabels(),size=12)

ax[0][0].set_ylabel('Number of houses',size=12)

ax[0][0].set_title("Garage capacity in cars",size=13)

plt.tight_layout()



ax[0][1].scatter(range(df.shape[0]), df["GarageArea"].values,color='orange')

ax[0][1].set_title("Distribution of GarageArea", size=13)

ax[0][1].set_xlabel("Number of Occurences", size=12)

ax[0][1].set_ylabel("GarageArea, Square Feet", size=12)

plt.setp(ax[0][1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0][1].get_yticklabels(),size=12)

plt.tight_layout()



sns.boxplot('GarageCars', 'SalePrice', data = df, order = [2,1,3,0,4],ax = ax[1][0]).set(ylim = (0, 500000))

ax[1][0].set_title("SalePrice VS GarageCars",size=13)

plt.tight_layout()



ax[1][1].scatter(df["SalePrice"], df["GarageArea"].values,color='orange')

ax[1][1].set_title("SalePrice vs GarageArea", size=13)

ax[1][1].set_xlabel("SalePrice", size=12)

ax[1][1].set_ylabel("GarageArea, Square Feet", size=12)

plt.setp(ax[1][1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[1][1].get_yticklabels(),size=12)

plt.tight_layout()
print (df.groupby('GarageQual').size())

print ('--------------')

print (df.groupby('GarageCond').size())



garagequal_x=df.groupby('GarageQual').size().index.get_values()

garagequal_y=df.groupby('GarageQual').size().get_values()



garagecond_x=df.groupby('GarageCond').size().index.get_values()

garagecond_y=df.groupby('GarageCond').size().get_values()



fig, ax = plt.subplots(2, 2, figsize = (11, 8))



ax[0][0].bar(garagequal_x-0.1, garagequal_y,width=0.2,color='orange',align='center',label="OverallQual")

ax[0][0].bar(garagecond_x+0.1, garagecond_y,width=0.2,color='g',align='center',label="OverallCond")

ax[0][0].legend(prop={'size':12})

ax[0][0].set_title("GarageQual / GarageCond condition",size=13)

ax[0][0].set_xticks([0,1,2,3,4,5])

ax[0][0].set_ylabel('Number of houses',size=12)

plt.setp(ax[0][0].get_xticklabels(), rotation=0,size=12)

plt.setp(ax[0][0].get_yticklabels(), rotation=0,size=12)

plt.tight_layout()



sns.boxplot('GarageQual', 'SalePrice', data = df, ax = ax[0][1]).set(ylim = (0, 500000))

ax[0][1].set_title("SalePrice VS GarageQual",size=13)

plt.tight_layout()



sns.boxplot('GarageCond', 'SalePrice', data = df, ax = ax[1][0]).set(ylim = (0, 500000))

ax[1][0].set_title("SalePrice VS GarageCond",size=13)

plt.tight_layout()



del garagequal_x, garagequal_y, garagecond_x, garagecond_y
print (df.groupby('PavedDrive').size())

print ('--------------')



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('PavedDrive').size().plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("PavedDrive - driveway",size=13)

plt.tight_layout()



sns.boxplot('PavedDrive', 'SalePrice', data = df, order = ['N','P','Y'],ax = ax[1]).set(ylim = (0, 400000))

ax[1].set_title("SalePrice VS PavedDrive",size=13)

plt.tight_layout()
print ('WoodDeckSF  == 0:  {0}'.format(df.loc[df['WoodDeckSF'] == 0,'WoodDeckSF'].count()))

print ('WoodDeckSF  != 0:  {0}'.format(df.loc[df['WoodDeckSF'] != 0,'WoodDeckSF'].count()))



fig, ax = plt.subplots(figsize = (7, 4))



ax.scatter(range(df.shape[0]), df["WoodDeckSF"].values,color='orange')

ax.set_title("WoodDeckSF - wood deck area", size=13)

ax.set_xlabel("Number of Occurences", size=12)

ax.set_ylabel("WoodDeckSF, Square Feet", size=12)

plt.setp(ax.get_xticklabels(),rotation=0,size=12)

plt.setp(ax.get_yticklabels(),size=12)

plt.tight_layout()
print ('OpenPorchSF  == 0:  {0}'.format(df.loc[df['OpenPorchSF'] == 0,'OpenPorchSF'].count()))

print ('OpenPorchSF  != 0:  {0}'.format(df.loc[df['OpenPorchSF'] != 0,'OpenPorchSF'].count()))



fig, ax = plt.subplots(figsize = (7, 4))



ax.scatter(range(df.shape[0]), df["OpenPorchSF"].values,color='orange')

ax.set_title("OpenPorchSF - open porch area", size=13)

ax.set_xlabel("Number of Occurences", size=12)

ax.set_ylabel("OpenPorchSF, Square Feet", size=12)

plt.setp(ax.get_xticklabels(),rotation=0,size=12)

plt.setp(ax.get_yticklabels(),size=12)

plt.tight_layout()
print ('EnclosedPorch  == 0:  {0}'.format(df.loc[df['EnclosedPorch'] == 0,'EnclosedPorch'].count()))

print ('EnclosedPorch  != 0:  {0}'.format(df.loc[df['EnclosedPorch'] != 0,'EnclosedPorch'].count()))



fig, ax = plt.subplots(figsize = (7, 4))



ax.scatter(range(df.shape[0]), df["EnclosedPorch"].values,color='orange')

ax.set_title("EnclosedPorch - enclosed porch area", size=13)

ax.set_xlabel("Number of Occurences", size=12)

ax.set_ylabel("EnclosedPorch, Square Feet", size=12)

plt.setp(ax.get_xticklabels(),rotation=0,size=12)

plt.setp(ax.get_yticklabels(),size=12)

plt.tight_layout()
print ('3SsnPorch  == 0:  {0}'.format(df.loc[df['3SsnPorch'] == 0,'3SsnPorch'].count()))

print ('3SsnPorch  != 0:  {0}'.format(df.loc[df['3SsnPorch'] != 0,'3SsnPorch'].count()))



fig, ax = plt.subplots(figsize = (7, 4))



ax.scatter(range(df.shape[0]), df["3SsnPorch"].values,color='orange')

ax.set_title("3SsnPorch - three season porch area", size=13)

ax.set_xlabel("Number of Occurences", size=12)

ax.set_ylabel("3SsnPorch, Square Feet", size=12)

plt.setp(ax.get_xticklabels(),rotation=0,size=12)

plt.setp(ax.get_yticklabels(),size=12)

plt.tight_layout()
print ('ScreenPorch  == 0:  {0}'.format(df.loc[df['ScreenPorch'] == 0,'ScreenPorch'].count()))

print ('ScreenPorch  != 0:  {0}'.format(df.loc[df['ScreenPorch'] != 0,'ScreenPorch'].count()))



fig, ax = plt.subplots(figsize = (7, 4))



ax.scatter(range(df.shape[0]), df["ScreenPorch"].values,color='orange')

ax.set_title("ScreenPorch - screen porch area", size=13)

ax.set_xlabel("Number of Occurences", size=12)

ax.set_ylabel("ScreenPorch, Square Feet", size=12)

plt.setp(ax.get_xticklabels(),rotation=0,size=12)

plt.setp(ax.get_yticklabels(),size=12)

plt.tight_layout()
print ('PoolArea  == 0:  {0}'.format(df.loc[df['PoolArea'] == 0,'PoolArea'].count()))

print ('PoolArea  != 0:  {0}'.format(df.loc[df['PoolArea'] != 0,'PoolArea'].count()))



fig, ax = plt.subplots(figsize = (7, 4))



ax.scatter(range(df.shape[0]), df["PoolArea"].values,color='orange')

ax.set_title("PoolArea - pool area", size=13)

ax.set_xlabel("Number of Occurences", size=12)

ax.set_ylabel("PoolArea, Square Feet", size=12)

plt.setp(ax.get_xticklabels(),rotation=0,size=12)

plt.setp(ax.get_yticklabels(),size=12)

plt.tight_layout()
print (df.groupby('PoolQC').size())

print ('--------------')



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('PoolQC').size().plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("PoolQC - pool quality",size=13)

plt.tight_layout()



sns.boxplot('PoolQC', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 800000))

ax[1].set_title("SalePrice VS PoolQC",size=13)

plt.tight_layout()
print (df.groupby('Fence').size())

print ('--------------')



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('Fence').size().plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("Fence - fence quality",size=13)

plt.tight_layout()



sns.boxplot('Fence', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 800000))

ax[1].set_title("SalePrice VS Fence",size=13)

plt.tight_layout()
print (df.groupby('MiscFeature').size())

print ('--------------')



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('MiscFeature').size().plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("MiscFeature - miscellaneous features",size=13)

plt.tight_layout()



sns.boxplot('MiscFeature', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 800000))

ax[1].set_title("SalePrice VS MiscFeature",size=13)

plt.tight_layout()
print ('MiscVal  == 0:  {0}'.format(df.loc[df['MiscVal'] == 0,'MiscVal'].count()))

print ('MiscVal  != 0:  {0}'.format(df.loc[df['MiscVal'] != 0,'MiscVal'].count()))



fig, ax = plt.subplots(figsize = (7, 4))



ax.scatter(range(df.shape[0]), df["MiscVal"].values,color='orange')

ax.set_title("MiscVal - value of miscellaneous features", size=13)

ax.set_xlabel("Number of Occurences", size=12)

ax.set_ylabel("MiscVal, $", size=12)

plt.setp(ax.get_xticklabels(),rotation=0,size=12)

plt.setp(ax.get_yticklabels(),size=12)

plt.tight_layout()
fig, ax = plt.subplots(5, 2, figsize = (11, 16))



sns.countplot(x = 'YrSold', data = df, ax=ax[0][0])

plt.tight_layout()



sns.countplot(x = 'MoSold', data = df, ax=ax[0][1])

plt.tight_layout()



sns.boxplot('YrSold', 'SalePrice', data = df, ax = ax[1][0]).set(ylim = (0, 400000))

plt.tight_layout()



sns.boxplot('MoSold', 'SalePrice', data = df, ax = ax[1][1]).set(ylim = (0, 400000))

plt.tight_layout()



df[df['YrSold']==2006].groupby('MoSold').size().plot(kind='bar', ax=ax[2][0])

plt.setp(ax[2][0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[2][0].get_yticklabels(),size=12)

ax[2][0].set_ylabel('Number of houses',size=12)

ax[2][0].set_title("Sold in 2006",size=13)

ax[2][0].set_ylim(0, 80)

plt.tight_layout()



df[df['YrSold']==2007].groupby('MoSold').size().plot(kind='bar', ax=ax[2][1])

plt.setp(ax[2][1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[2][1].get_yticklabels(),size=12)

ax[2][1].set_ylabel('Number of houses',size=12)

ax[2][1].set_title("Sold in 2007",size=13)

ax[2][1].set_ylim(0, 80)

plt.tight_layout()



df[df['YrSold']==2008].groupby('MoSold').size().plot(kind='bar', ax=ax[3][0])

plt.setp(ax[3][0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[3][0].get_yticklabels(),size=12)

ax[3][0].set_ylabel('Number of houses',size=12)

ax[3][0].set_title("Sold in 2008",size=13)

ax[3][0].set_ylim(0, 80)

plt.tight_layout()



df[df['YrSold']==2009].groupby('MoSold').size().plot(kind='bar', ax=ax[3][1])

plt.setp(ax[3][1].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[3][1].get_yticklabels(),size=12)

ax[3][1].set_ylabel('Number of houses',size=12)

ax[3][1].set_title("Sold in 2009",size=13)

ax[3][1].set_ylim(0, 80)

plt.tight_layout()



df[df['YrSold']==2010].groupby('MoSold').size().plot(kind='bar', ax=ax[4][0])

plt.setp(ax[4][0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[4][0].get_yticklabels(),size=12)

ax[4][0].set_ylabel('Number of houses',size=12)

ax[4][0].set_title("Sold in 2010",size=13)

ax[4][0].set_ylim(0, 80)

ax[4][0].set_xticks([1,2,3,4,5,6,7,8,9,10,11,12])

plt.tight_layout()
print (df.groupby('SaleType').size())

print ('--------------')



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('SaleType').size().plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("Type of sale",size=13)

plt.tight_layout()



sns.boxplot('SaleType', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 800000))

ax[1].set_title("SalePrice VS SaleType",size=13)

plt.tight_layout()
print (df.groupby('SaleCondition').size())

print ('--------------')



fig, ax = plt.subplots(1, 2, figsize = (11, 4))



df.groupby('SaleCondition').size().plot(kind='bar', ax=ax[0])

plt.setp(ax[0].get_xticklabels(),rotation=0,size=12)

plt.setp(ax[0].get_yticklabels(),size=12)

ax[0].set_ylabel('Number of houses',size=12)

ax[0].set_title("Sale condition",size=13)

plt.tight_layout()



sns.boxplot('SaleCondition', 'SalePrice', data = df, ax = ax[1]).set(ylim = (0, 800000))

ax[1].set_title("SalePrice VS SaleCondition",size=13)

plt.tight_layout()
df['TotalBathBsmt']  = df["BsmtFullBath"] + df["BsmtHalfBath"]

df['TotalBathAbvGr'] = df["FullBath"] + df["HalfBath"] 

df['TotalBath'] = df['TotalBathBsmt'] + df['TotalBathAbvGr']

df['TotalRooms'] = df['TotalBathBsmt'] + df['TotalBathAbvGr'] + df['TotRmsAbvGrd']

df[['TotalBath','TotalRooms','TotRmsAbvGrd']].head(10)
# Let's create feature identifying if the house has 2nd floor

df['Has2ndFloor'] = 0 

df.loc[df['2ndFlrSF']>0,'Has2ndFloor'] = 1

df['Has2ndFloor'] = df['Has2ndFloor'].astype(object)



# Let's create explicit feature identifying if the house has basement

df['HasBsmt'] = 0 

df.loc[df['TotalBsmtSF']>0,'HasBsmt'] = 1

df['HasBsmt'] = df['HasBsmt'].astype(object)



# Let's create explicit feature identifying if the house has Masonry Veneer Area

df['HasMasVnr'] = 0 

df.loc[df['MasVnrArea']>0,'HasMasVnr'] = 1

df['HasMasVnr'] = df['HasMasVnr'].astype(object)



# Let's create explicit feature identifying if the house has LowQualFinSF

df['HasLowQualFinSF'] = 0 

df.loc[df['LowQualFinSF']>0,'HasLowQualFinSF'] = 1

df['HasLowQualFinSF'] = df['HasLowQualFinSF'].astype(object)



# Let's create explicit feature identifying if the house has 1stFlrSF > TotalBsmtSF 

df['HasAddArea1stFlr'] = 0

df['AddArea1stFlr'] = df['1stFlrSF'] - df['TotalBsmtSF']

df.loc[df['AddArea1stFlr']!=0,'HasAddArea1stFlr'] = 1

df['HasAddArea1stFlr'] = df['HasAddArea1stFlr'].astype(object)



# HasWoodDeck

df['HasWoodDeck'] = 0

df.loc[df['WoodDeckSF']!=0,'HasWoodDeck'] = 1

df['HasWoodDeck'] = df['HasWoodDeck'].astype(object)



# HasOpenPorch

df['HasOpenPorch'] = 0

df.loc[df['OpenPorchSF']!=0,'HasOpenPorch'] = 1

df['HasOpenPorch'] = df['HasOpenPorch'].astype(object)



# HasEnclosedPorch

df['HasEnclosedPorch'] = 0

df.loc[df['EnclosedPorch']!=0,'HasEnclosedPorch'] = 1

df['HasEnclosedPorch'] = df['HasEnclosedPorch'].astype(object)



# Has3SsnPorch

df['Has3SsnPorch'] = 0

df.loc[df['3SsnPorch']!=0,'Has3SsnPorch'] = 1

df['Has3SsnPorch'] = df['Has3SsnPorch'].astype(object)



# HasScreenPorch

df['HasScreenPorch'] = 0

df.loc[df['ScreenPorch']!=0,'HasScreenPorch'] = 1

df['HasScreenPorch'] = df['HasScreenPorch'].astype(object)



# HasPool

df['HasPool'] = 0

df.loc[df['PoolArea']!=0,'HasPool'] = 1

df['HasPool'] = df['HasPool'].astype(object)



# HasFence

df['HasFence'] = 0

df.loc[df['Fence']!=0,'HasFence'] = 1

df['HasFence'] = df['HasFence'].astype(object)



# HasShed

df['HasShed'] = 0

df.loc[df['MiscFeature']=='Shed','HasShed'] = 1

df['HasShed'] = df['HasShed'].astype(object)



# HasGarage

df['HasGarage'] = 0

df.loc[df['GarageArea']!=0,'HasGarage'] = 1

df['HasGarage'] = df['HasGarage'].astype(object)



# HasFireplace

df['HasFireplace'] = 0

df.loc[df['Fireplaces']!=0,'HasFireplace'] = 1

df['HasFireplace'] = df['HasFireplace'].astype(object)
f, ax = plt.subplots(6, 3, figsize=(11, 20))

sns.boxplot(x='HasBsmt', y='SalePrice', data=df, ax=ax[0][0])

sns.boxplot(x='Has2ndFloor', y='SalePrice', data=df, ax=ax[0][1])

sns.boxplot(x='HasMasVnr', y='SalePrice', data=df, ax=ax[0][2])



sns.boxplot(x='HasLowQualFinSF', y='SalePrice', data=df, ax=ax[1][0])

sns.boxplot(x='TotalBathBsmt', y='SalePrice', data=df, ax=ax[1][1])

sns.boxplot(x='TotalBathAbvGr', y='SalePrice', data=df, ax=ax[1][2])



sns.boxplot(x='TotalBath', y='SalePrice', data=df, ax=ax[2][0])

sns.boxplot(x='HasFireplace', y='SalePrice', data=df, ax=ax[2][1])

sns.boxplot(x='HasAddArea1stFlr', y='SalePrice', data=df, ax=ax[2][2])



sns.boxplot(x='HasWoodDeck', y='SalePrice', data=df, ax=ax[3][0])

sns.boxplot(x='HasOpenPorch', y='SalePrice', data=df, ax=ax[3][1])

sns.boxplot(x='HasEnclosedPorch', y='SalePrice', data=df, ax=ax[3][2])



sns.boxplot(x='Has3SsnPorch', y='SalePrice', data=df, ax=ax[4][0])

sns.boxplot(x='HasScreenPorch', y='SalePrice', data=df, ax=ax[4][1])

sns.boxplot(x='HasPool', y='SalePrice', data=df, ax=ax[4][2])



sns.boxplot(x='HasFence', y='SalePrice', data=df, ax=ax[5][0])

sns.boxplot(x='HasShed', y='SalePrice', data=df, ax=ax[5][1])

sns.boxplot(x='HasGarage', y='SalePrice', data=df, ax=ax[5][2])
df['TotalLivSF'] = df['1stFlrSF'] + df['2ndFlrSF']



area_cols = ['MasVnrArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 

             'ScreenPorch', 'PoolArea']

df['TotalArea'] = df[area_cols].sum(axis=1)



area_cols = ['OverallQual', 'ExterQual', 'BsmtQual', 'HeatingQC',

             'KitchenQual', 'FireplaceQu', 'GarageQual']

df['AggQual'] = df[area_cols].sum(axis=1)



area_cols = ['OverallCond', 'ExterCond', 'BsmtCond', 'HeatingQC',

             'KitchenQual', 'FireplaceQu', 'GarageCond']

df['AggCondPositive'] = df[area_cols].sum(axis=1)



area_cols = ['PoolQC', 'Fence']

df['AggCondNegative'] = df[area_cols].sum(axis=1)



df['TimeSinceSold'] = (2010 - df['YrSold'])*12 + df['MoSold']



df['AgeSinceBuilt'] = 2010 - df['YearBuilt']



df['SeasonSold'] = df['MoSold'].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 

                                    6:2, 7:2, 8:2, 9:3, 10:3, 11:3}).astype(int)



df['BadHeating'] = df.HeatingQC.replace({5: 0, 4: 1, 3: 2, 2: 3, 1: 4})



df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']



df['HighSeason'] = df['MoSold'].replace( 

        {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})

    

df['SeasonSold'] = df['MoSold'].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 

                                                  6:2, 7:2, 8:2, 9:3, 10:3, 11:3}).astype(int)

    

df['BoughtOffPlan'] = df.SaleCondition.replace(

        {'Abnorml' : 0, 'Alloca' : 0, 'AdjLand' : 0, 'Family' : 0, 'Normal' : 0, 'Partial' : 1})



df['SaleCondition_PriceDown'] = df.SaleCondition.replace(

        {'Abnorml': 1, 'Alloca': 2, 'AdjLand': 1, 'Family': 2, 'Normal': 2, 'Partial': 0})





print (df[['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'AddArea1stFlr', 'GrLivArea', 'Has2ndFloor']].head(10))



f, ax = plt.subplots(2, 4, figsize=(11, 6))



ax[0][0].scatter(df["LotArea"].values, df["SalePrice"].values,color='orange')

ax[0][1].scatter(df["MasVnrArea"].values, df["SalePrice"].values,color='orange')

ax[0][2].scatter(df["TotalBsmtSF"].values, df["SalePrice"].values,color='orange')

ax[0][3].scatter(df["GrLivArea"].values, df["SalePrice"].values,color='orange')

ax[1][0].scatter(df["AddArea1stFlr"].values, df["SalePrice"].values,color='orange')

ax[1][1].scatter(df["TotalArea"].values, df["SalePrice"].values,color='orange')

ax[1][2].scatter(df["TotalLivSF"].values, df["SalePrice"].values,color='orange')

# ax[1][3].scatter(df["AggQual"].values, df["SalePrice"].values,color='orange')



plt.tight_layout()
print("Find most important features relative to target")

corr_overallqual=df.corr()["SalePrice"]

print (corr_overallqual[np.argsort(corr_overallqual, axis=0)[::-1]])



fig, ax = plt.subplots(figsize = (6, 10))

corr_overallqual[np.argsort(corr_overallqual, axis=0)[::-1]].plot(kind='barh')

plt.tick_params(labelsize=12)

plt.ylabel("Pearson correlation",size=12)

plt.title('Correlated features with SalePrice', size=13)

plt.tight_layout()
df_mod = df.copy()



print (df.shape, df_mod.shape)



df_mod.drop(df_mod[df_mod.GrLivArea > 4000.0].index, inplace=True)

df_mod.drop(df_mod[df_mod.BsmtFinSF1  > 2000.0].index, inplace=True)

df_mod.drop(df_mod[df_mod.BsmtFinSF2  > 1200.0].index, inplace=True)

df_mod.drop(df_mod[df_mod.TotalBsmtSF > 3000.0].index, inplace=True)

df_mod.drop(df_mod[df_mod.MasVnrArea > 1200.0].index, inplace=True)

df_mod.drop(df_mod[df_mod.LotArea > 100000.0].index, inplace=True)

df_mod.drop(df_mod[df_mod['1stFlrSF'] > 3000.0].index, inplace=True)

df_mod.drop((df_mod[df_mod.TotalRooms < 5].index) | (df_mod[df_mod.TotalRooms > 15].index), inplace=True)



print (df.shape, df_mod.shape)



print ('------------------------------')

print ('Skew:       Original', ' Modified')

print ('OverallQual: {0:2f}   {1:2f}'.format(df["OverallQual"].skew(), df_mod["OverallQual"].skew()))

print ('OverallCond: {0:2f}   {1:2f}'.format(df["OverallCond"].skew(), df_mod["OverallCond"].skew())) 

print ('AggQual:     {0:2f}   {1:2f}'.format(df["AggQual"].skew(), df_mod["AggQual"].skew()))

print ('BsmtQual:    {0:2f}   {1:2f}'.format(df["BsmtQual"].skew(), df_mod["BsmtQual"].skew())) 

print ('BsmtQual:    {0:2f}   {1:2f}'.format(df["BsmtQual"].skew(), df_mod["BsmtQual"].skew())) 

print ('BsmtCond:    {0:2f}   {1:2f}'.format(df["BsmtCond"].skew(), df_mod["BsmtCond"].skew()))

print ('HeatingQC:   {0:2f}   {1:2f}'.format(df["HeatingQC"].skew(), df_mod["HeatingQC"].skew()))

print ('GarageQual:  {0:2f}   {1:2f}'.format(df["GarageQual"].skew(), df_mod["GarageQual"].skew()))

print ('GarageCond:  {0:2f}   {1:2f}'.format(df["GarageCond"].skew(), df_mod["GarageCond"].skew()))

print ('Functional:  {0:2f}   {1:2f}'.format(df["Functional"].skew(), df_mod["Functional"].skew()))

print ('YearBuilt:   {0:2f}   {1:2f}'.format(df["YearBuilt"].skew(), df_mod["YearBuilt"].skew()))

print ('BsmtFinSF1:  {0:2f}   {1:2f}'.format(df["BsmtFinSF1"].skew(), df_mod["BsmtFinSF1"].skew())) 

print ('BsmtFinSF2:  {0:2f}   {1:2f}'.format(df["BsmtFinSF2"].skew(), df_mod["BsmtFinSF2"].skew()))

print ('BsmtUnfSF:   {0:2f}   {1:2f}'.format(df["BsmtUnfSF"].skew(), df_mod["BsmtUnfSF"].skew()))

print ('TotalBsmtSF: {0:2f}   {1:2f}'.format(df["TotalBsmtSF"].skew(), df_mod["TotalBsmtSF"].skew()))

print ('MasVnrArea:  {0:2f}   {1:2f}'.format(df["MasVnrArea"].skew(), df_mod["MasVnrArea"].skew()))

print ('LotArea:     {0:2f}   {1:2f}'.format(df["LotArea"].skew(), df_mod["LotArea"].skew()))

print ('LotFrontage: {0:2f}   {1:2f}'.format(df["LotFrontage"].skew(), df_mod["LotFrontage"].skew()))

print ('1stFlrSF:    {0:2f}   {1:2f}'.format(df["1stFlrSF"].skew(), df_mod["1stFlrSF"].skew()))

print ('GrLivArea:   {0:2f}   {1:2f}'.format(df["GrLivArea"].skew(), df_mod["GrLivArea"].skew()))

print ('TotalArea:   {0:2f}   {1:2f}'.format(df["TotalArea"].skew(), df_mod["TotalArea"].skew()))

print ('TotalLivSF:  {0:2f}   {1:2f}'.format(df["TotalLivSF"].skew(), df_mod["TotalLivSF"].skew()))

print ('\n')

print ('SalePrice:   {0:2f}   {1:2f}'.format(df["SalePrice"].skew(), df_mod["SalePrice"].skew())) 

print ('------------------------------')
f, ax = plt.subplots(3, 3, figsize=(11, 12))

sns.boxplot(x='HasBsmt', y='SalePrice', data=df_mod, ax=ax[0][0])

sns.boxplot(x='Has2ndFloor', y='SalePrice', data=df_mod, ax=ax[0][1])

sns.boxplot(x='TotalBathBsmt', y='SalePrice', data=df_mod, ax=ax[0][2])

sns.boxplot(x='TotalBathAbvGr', y='SalePrice', data=df_mod, ax=ax[1][0])

sns.boxplot(x='TotalBath', y='SalePrice', data=df_mod, ax=ax[1][1])

sns.boxplot(x='TotalRooms', y='SalePrice', data=df_mod, ax=ax[1][2])

sns.boxplot(x='KitchenAbvGr', y='SalePrice', data=df_mod, ax=ax[2][0])
# House prices distribution

sns.set(style="whitegrid", palette="muted", color_codes=True)



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))

plt.subplots_adjust(wspace=0.5, hspace=0.5)



sns.distplot(df['SalePrice'], color="b", kde=False)

ax.set_ylim(0, 180)

ax.set_xlabel('Sale Price $')

ax.set_ylabel('Number of houses')

sns.plt.title("Housing prices distribution")

plt.tight_layout()



print ('Number of null values in [SalePrice]: {0}'.format(df['SalePrice'].isnull().sum()))
print ("Some Statistics of the Housing Price:\n")

print (df['SalePrice'].describe())
df_1 = df[(df["YearBuilt"] == df["YrSold"]) & (df["YearRemodAdd"] == df["YrSold"])].groupby('YrSold').size()

df_2 = df[(df["YearBuilt"] != df["YrSold"]) & (df["YearRemodAdd"] == df["YrSold"])].groupby('YrSold').size()

df_3 = df[(df["YearBuilt"] != df["YrSold"]) & (df["YearRemodAdd"] != df["YrSold"])].groupby('YrSold').size()



result = pd.concat([df_1, df_2, df_3], axis=1)



result.plot.bar(stacked=True)

plt.title('Features of sold houses', fontsize=16, color='black') 

plt.xlabel('YrSold',fontsize=14, color='black') 

plt.ylabel('Number of sold houses', fontsize=14, color='black') 

plt.xticks(rotation=0, size=13) 

plt.yticks(size=13) 

plt.legend(["Built, remod. & sold in same year", 

            "Built in past, remod. & sold in same year", 

            "Built and remod. in past"], bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.1, prop={'size':14}) 



del df_1, df_2, df_3, result