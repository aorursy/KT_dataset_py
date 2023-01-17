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
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
print(data.info())
data.columns
data.head()


data.shape
print(data.country_txt.value_counts(dropna=False))

print(data.alternative.value_counts(dropna=False)) #approxdate,resolution,location,alternative,alternative_txt,attacktype2,attacktype2_txt,attacktype3,attacktype3_txt

#weapsubtype4, weapsubtype4_txt,nkillus,nkillter,propextent

#summary is an explanation of that event..
data1 = data['approxdate']



data1.dropna(inplace=True)

data1.head()

len(data1)

assert data1.notnull().all()
print(data.attacktype1_txt.value_counts(dropna=False))
data.info()
data1 = data.head()

data2 = data.tail()



conc_data_row = pd.concat([data1,data2],axis=0,ignore_index=True)

print(data1.shape,data2.shape,conc_data_row.shape)
data3 = data['country_txt'].head()

data4 = data['region_txt'].head()



conc_data_col = pd.concat([data3,data4],axis=1,ignore_index=True)

print(data3,data4,conc_data_col)
data.corr()
import seaborn as sns

import matplotlib.pyplot as plt
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()

data.describe()
print(data.country_txt.unique())

print(data.attacktype1_txt.unique())
turkey = data[data.country_txt=="Turkey"]

turkeyAssassination = turkey[turkey.attacktype1_txt=="Assassination"]

argentina = data[data.country_txt=="Argentina"]

mexico = data[data.country_txt=="Mexico"]

usa = data[data.country_txt=="United States"]

russia = data[data.country_txt=="Russia"]



datatargtype1 =turkey.loc[:,["targtype1","specificity","latitude","longitude"]]

datatargtype1.plot()
datatargtype1.plot(subplots = True)

plt.show()
#scatter plot

datatargtype1.plot(kind="scatter",x="latitude",y="longitude")

plt.show()
istanbul = turkey[turkey.city=="Istanbul"]

ankara = turkey[turkey.city=="Ankara"]

izmir = turkey[turkey.city=="Izmir"]
istanbul.plot(kind="scatter",x="latitude",y="longitude",color="red")

izmir.plot(kind="scatter",x="latitude",y="longitude",color="blue")

ankara.plot(kind="scatter",x="latitude",y="longitude",color="green")

plt.show()
fig , axes = plt.subplots(nrows=2,ncols=1)

istanbul.plot(kind="hist",y="longitude",bins=50,range=(0,250),normed=True,ax=axes[0])

istanbul.plot(kind="hist",y="longitude",bins=50,range=(0,250),normed=True,ax=axes[1],cumulative=True)

plt.savefig('graph1.png')

plt
print(turkey.city.unique())
plt.plot(data.eventid,data.country_txt,color="red",label="Country based event")

plt.show()



turkey.boxplot(column='iyear',by = 'attacktype1')

istanbul.boxplot(column='iyear',by = 'attacktype1_txt')

plt.show()
turkey.plot("iyear","targsubtype1")

russia.plot("iyear","targsubtype1")

plt.show()
two_cloumns = data.loc[:,["iyear","attacktype1_txt"]]

two_cloumns.columns
plt.plot(two_cloumns.iyear,two_cloumns.attacktype1_txt,color="blue",label="yearly event")

plt.legend()

plt.show()
melt_fonksiyonu = data.loc[:,["country_txt","attacktype1_txt","attacktype1"]]

melt_fonksiyonu.shape
print(data.attacktype1_txt.unique())
melted =pd.melt(frame = melt_fonksiyonu,id_vars='country_txt',value_vars=['Assassination','Bombing/Explosion'])

melted