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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
housetrain=pd.read_csv("../input/train.csv")
housetest=pd.read_csv("../input/test.csv")
house_train_num=housetrain.select_dtypes(include=[np.number])
house_train_cat=housetrain.select_dtypes(include=['object'])
house_train_num.columns
plt.boxplot(house_train_num["SalePrice"])
plt.hist(house_train_num["SalePrice"])
h_train = housetrain[housetrain["SalePrice"]<300000]
plt.boxplot(h_train["SalePrice"])
h_train.shape
plt.boxplot(np.log(housetrain["SalePrice"]))
house_train_num_corr=house_train_num.corr()
house_train_num_corr=house_train_num.corr()
house_train_num_corr["SalePrice"]
house_train_num_cols = []

house_train_num_cols.extend(house_train_num_corr[(house_train_num_corr["SalePrice"]>0.3) ].index.values)

house_train_num_cols.extend(house_train_num_corr[(house_train_num_corr["SalePrice"]<-0.3) ].index.values)


house_train_num_cols
h_train_num_col_filtered=house_train_num[house_train_num_cols]
(house_train_num.isnull().sum().sort_values(ascending=False))


for hc in ["LotFrontage","GarageYrBlt","MasVnrArea"]:

    print (hc)

    print(house_train_num[hc].mean())

    print(house_train_num[hc].median())
for col in ["LotFrontage","GarageYrBlt","MasVnrArea"]:

    h_train_num_col_filtered[col].fillna(h_train_num_col_filtered[col].median(),inplace=True)
(house_train_cat.isnull().sum().sort_values(ascending=False))
for col in ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu"]:

    house_train_cat[col].fillna('No Value',inplace=True)
for col in ["GarageCond","GarageQual","GarageFinish","GarageType","BsmtFinType2","BsmtExposure","BsmtFinType1","BsmtQual","BsmtCond","MasVnrType","Electrical"]:

    house_train_cat[col].fillna(house_train_cat[col].value_counts().idxmax(),inplace=True)
house_train_cat["LotConfig"].unique()
for col in ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu","GarageCond","GarageQual","GarageFinish","GarageType","BsmtFinType2","BsmtExposure","BsmtFinType1","BsmtQual","BsmtCond","MasVnrType","Electrical"]:

     if len(house_train_cat[col].value_counts())< 10:

            print (col)

house_train_cat1 = house_train_cat

for col in ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu","GarageCond","GarageQual","GarageFinish","GarageType","BsmtFinType2","BsmtExposure","BsmtFinType1","BsmtQual","BsmtCond","MasVnrType","Electrical"]:

    n_df = pd.get_dummies(house_train_cat1[col])

    house_train_cat1 = pd.concat([house_train_cat1,n_df],axis=1)

    house_train_cat1.drop([col],axis=1,inplace = True)

    print(col)
house_train_cat1.columns
house_train_cat1.shape
house_train_df=pd.concat([h_train_num_col_filtered,house_train_cat1],axis=1)
house_train_df.shape