# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/oasis_cross-sectional.csv')
data.head()
data.shape
print(data["M/F"].value_counts(dropna=False))
data.describe()
data_new=data.head()

data_new
melted=pd.melt(frame=data_new,id_vars="ID",value_vars=["Age","nWBV"])

melted
melted.pivot(index="ID",columns="variable",values="value")
data1=data.head()

data2=data.tail()

conc_dataRow=pd.concat([data1,data2],axis=0,ignore_index=True)

conc_dataRow
data3=data["Age"].head()

data4=data["nWBV"].head()

conc_dataCol=pd.concat([data3,data4],axis=1)

conc_dataCol
data.dtypes
data.info()
print(data["SES"].value_counts(dropna=False))#drpna=false none değer varsa drop etme onu da göster
data["SES"].dropna(inplace=True)#inplace=True çıkar değerleri data1 e kaydet demek
assert data["SES"].notnull().all()#doğru olduğunda hiçbir şey return etmez
print(data["SES"].value_counts(dropna=False))
data["SES"].fillna("empty",inplace=True)

print(data["SES"].value_counts(dropna=False))
data.corr()
f,ax = plt.subplots(figsize=(13,13))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
plt.scatter(data.Age,data.nWBV,alpha=0.5,color='red')
data.ASF.plot(kind = 'line', color = 'g',label = 'Age',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.nWBV.plot(color = 'r',label = 'nWBV',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.show()
#data.Age.plot(kind = 'hist',bins = 12,figsize = (12,12))

data.nWBV.plot(kind = 'hist',bins = 12,figsize = (12,12))

plt.show()
series = data['Age']        # data['Defense'] = series

print(type(series))

data_frame = data[['Age']]  # data[['Defense']] = data frame

print(type(data_frame))