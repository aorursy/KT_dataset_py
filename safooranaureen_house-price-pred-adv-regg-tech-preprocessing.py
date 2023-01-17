import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import os

print(os.listdir("../input"))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

print(train.shape)

print(test.shape)
train.head()
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
df = pd.concat((train,test))

temp_df = df

print(df.shape)
df.tail()
df = df.set_index('Id')

df.shape
df.info()
plt.figure(figsize=(16,9))

sns.heatmap(df.isnull())
missing_value_percent = df.isnull().sum()/df.shape[0] * 100

missing_value_percent
null_value_clm_greater_20 = missing_value_percent[missing_value_percent > 20].keys()

null_value_clm_greater_20
df = df.drop(null_value_clm_greater_20, axis =1)

df.shape

#80-6=74
num_var = df.select_dtypes(include=['int64','float64']).columns

num_var
cat_var = df.select_dtypes(include=['object']).columns

cat_var
missing_col = df.columns[df.isnull().any()]

missing_col
isnull_per = df.select_dtypes(include=['object']).isnull().mean()*100

miss_vars = isnull_per[isnull_per >0].keys()

miss_vars
df.select_dtypes(include=['object']).isnull().sum()
for var in miss_vars:

    df[var].fillna(df[var].mode()[0],inplace=True)

    print(var,"=",df[var].mode()[0])
df.select_dtypes(include=['object']).isnull().sum().sum()
df.select_dtypes(include=['int64','float64']).isnull().sum()
less_null_col = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath'

                ,'GarageCars','GarageArea']
plt.figure(figsize=(15,15))

sns.set()

for i,var in enumerate(less_null_col):

    plt.subplot(4,4,i+1)

    sns.distplot(df[var], bins=20, kde_kws={'linewidth':8, 'color':'red'}, label="Original")
df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(df['BsmtFinSF1'].median())

df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(df['BsmtUnfSF'].median())

df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].median())

df['GarageArea'] = df['GarageArea'].fillna(df['GarageArea'].median())

for var in ['BsmtFinSF2','BsmtFullBath','BsmtHalfBath','GarageCars']:

    df[var].fillna(df[var].mode()[0],inplace=True)
df.select_dtypes(include=['int64','float64']).isnull().sum().sum()

#486+23+159
num_vars_miss = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']

cat_vars = ['LotConfig','MasVnrType','GarageType']

for cat_var, num_var_miss in zip(cat_vars,num_vars_miss):

    for var_class in df[cat_var].unique():

        df.update(df[df.loc[:,cat_var] == var_class][num_var_miss].replace(np.nan,df[df.loc[:,cat_var] == var_class][num_var_miss].mean()))
df.select_dtypes(include=['int64','float64']).isnull().sum().sum()
df.shape
df.isnull().sum().sum()
df.to_csv('hpp_missing_values_done_dataframe.csv')
df_categ = df.select_dtypes(include=['object'])

df_categ.shape
df_dummies = df.loc[:,('MSZoning', 'Street', 'LotConfig','Neighborhood', 'Condition1', 'Condition2',

                          'RoofStyle', 'RoofMatl', 'Exterior1st','Exterior2nd', 'MasVnrType', 'Foundation',

                          'Heating', 'CentralAir', 'Electrical', 'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition','BldgType')]

df_dummies.shape
dummy_df = pd.get_dummies(df_dummies, drop_first=True)

dummy_df.shape
dummy_df.head(2)
df_categ_dummies = pd.merge(df, dummy_df, on = 'Id')

df_categ_dummies.shape
df_ohe = df_categ_dummies.drop(['MSZoning', 'Street', 'LotConfig','Neighborhood', 'Condition1', 'Condition2',

                          'RoofStyle', 'RoofMatl', 'Exterior1st','Exterior2nd', 'MasVnrType', 'Foundation',

                          'Heating', 'CentralAir', 'Electrical', 'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition', 'BldgType'],

                          axis = 1)

df_ohe.shape

#205-20
df_ohe.isnull().sum().sum()
enc_df = df.loc[:,['LotShape', 'LandContour', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond','BsmtQual',

                   'BsmtCond', 'BsmtExposure','BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual',

                   'Functional', 'GarageFinish', 'GarageQual', 'GarageCond','HouseStyle']]

enc_df.shape
enc_df.isnull().sum().sum()
order_Label1 = {'Reg':4, 'IR1':3, 'IR2':2,'IR3':1}

order_Label2 = {'Lvl':4, 'Bnk' :3,'HLS':2,'Low':1}

order_Label3 = {'AllPub':4,'NoSewr':3,'NoSeWa':2,'ELO':1}

order_Label4 = {'Gtl' :3,'Mod':2,'Sev':1}

order_Label56 = {"Ex":4,"Gd":3,"TA":2,"Fa":1, "Po":0.5}

order_Label78 = {"Ex":4,"Gd":3,"TA":2,"Fa":1,"Po":0.5, "NA":0}

order_Label9 = {"Gd":4,"Av":3,"Mn":2,"No":1, "NA":0.5}

order_Label1011 = {"GLQ":4,"ALQ":3,"BLQ":2,"Rec":1,"LwQ":0.5,"Unf":0.3,"NA":0}

order_Label1213 = {"Ex":4,"Gd":3,"TA":2,"Fa":1, "Po":0.5}

order_Label14 = {"Typ":5,"Min1":4,"Min2":3,"Mod":2,"Maj1":1,"Maj2":0.5,"Sev":0.3,"Sal":0}

order_Label15 = {'Fin':4, 'RFn' :3,'Unf':2,'NA':1}

order_Label1617 = {"Ex":4,"Gd":3,"TA":2,"Fa":1,"Po":0.5, "NA":0}

order_Label19 = {"SLvl":5,"SFoyer":4,"2.5Fin":3,"2.5Unf":2,"2Story":1,"1.5Fin":0.5,"1.5Unf":0.3,"1Story":0}



enc_df["LotShape_org_enc"] = enc_df['LotShape'].map(order_Label1)
enc_df["LandContour_org_enc"] = enc_df['LandContour'].map(order_Label2)

enc_df["Utilities_org_enc"] = enc_df['Utilities'].map(order_Label3)

enc_df["LandSlope_org_enc"] = enc_df['LandSlope'].map(order_Label4)

enc_df["ExterQual_org_enc"] = enc_df['ExterQual'].map(order_Label56)

enc_df["ExterCond_org_enc"] = enc_df['ExterCond'].map(order_Label56)

enc_df["BsmtQual_org_enc"] = enc_df['BsmtQual'].map(order_Label78)

enc_df["BsmtCond_org_enc"] = enc_df['BsmtCond'].map(order_Label78)

enc_df["BsmtExposure_org_enc"] = enc_df['BsmtExposure'].map(order_Label9)

enc_df["BsmtFinType1_org_enc"] = enc_df['BsmtFinType1'].map(order_Label1011)

enc_df["BsmtFinType2_org_enc"] = enc_df['BsmtFinType2'].map(order_Label1011)

enc_df["HeatingQC_org_enc"] = enc_df['HeatingQC'].map(order_Label1213)

enc_df["KitchenQual_org_enc"] = enc_df['KitchenQual'].map(order_Label1213)

enc_df["Functional_org_enc"] = enc_df['Functional'].map(order_Label14)

enc_df["GarageFinish_org_enc"] = enc_df['GarageFinish'].map(order_Label15)

enc_df["GarageQual_org_enc"] = enc_df['GarageQual'].map(order_Label1617)

enc_df["GarageCond_org_enc"] = enc_df['GarageCond'].map(order_Label1617)

enc_df["HouseStyle_org_enc"] = enc_df['HouseStyle'].map(order_Label19)

enc_df.shape
enc_df.head(2)
enc_df = enc_df.drop(['LotShape', 'LandContour', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond','BsmtQual',

                   'BsmtCond', 'BsmtExposure','BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual',

                   'Functional', 'GarageFinish', 'GarageQual', 'GarageCond','HouseStyle'], axis = 1)

enc_df.shape
preproccd_df = pd.merge(df_ohe, enc_df, on = "Id")

preproccd_df.shape

#185+18
preproccd_df = preproccd_df.drop(['LotShape', 'LandContour', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond','BsmtQual',

                   'BsmtCond', 'BsmtExposure','BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual',

                   'Functional', 'GarageFinish', 'GarageQual', 'GarageCond','HouseStyle'], axis = 1)

preproccd_df.shape
preproccd_df.head(3)
preproccd_df.isnull().sum().sum()
preproccd_df.to_csv('preprocessed_houseprice_dataframe.csv')