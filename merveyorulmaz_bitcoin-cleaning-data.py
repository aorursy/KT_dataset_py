# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")

data.head()
data.info()
f,ax = plt.subplots(figsize=(12,12))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt=".1f", ax=ax)

plt.show()
for index,value in data[ ['Weighted_Price']][5:10].iterrows():

    print(index,": ",value)
data.tail()
data.columns
data.shape



# row = 2099760, column = 8
data.info()
# value_counts

# dropna=False -> Nan değerleri de say



d = data["Low"].value_counts(dropna=False)

d
data.describe()
# Visual Exploratory Data Analysis



data.boxplot(column="Open")

plt.show()
data.boxplot(column="Weighted_Price")
# Tidy 

# Melt



data_new = data.head()

data_new



melted = pd.melt(frame=data_new, id_vars="Timestamp", value_vars=["High","Low"])

melted
# pivoting data

melted.pivot(index="Timestamp", columns="variable", values="value")
# Concatenating Data

# 1 . Dikey birleştirme : axis=0



data1 = data.head()

data2 = data.tail()



concat = pd.concat([data1,data2], axis=0, ignore_index=True)

concat
# 2. Yatay birleştirme



data1 = data["Timestamp"][10:15]

data2 = data["High"][1200:1205]



concat2 = pd.concat([data1,data2], axis=1)

concat2
# Data types

data.dtypes
#data convert

int(1.2)
data["Timestamp"] = data["Timestamp"].astype('float')
data.dtypes
data["Low"].value_counts(dropna=False)
# 109069 Nan data var

# bu değerleri listeden atma



data["Low"].dropna(inplace=True)
data["Low"].value_counts(dropna=False)
# Nan değerler gitti

#yukarıda yapılan işlemin işe yarayıp yaramadığını anlamak için;



assert data["Low"].notnull().all()



#birşey döndürmüyorsa doğru demektir
# Nan value'lara empty yaz



data["Low"].fillna("empty",inplace=True)
data.head(1)
# 1. column adı Open old ıcın bişi döndürmez

assert data.columns[1]=="Open"