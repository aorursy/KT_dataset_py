import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import numpy as np

%matplotlib inline
bands=pd.read_csv("../input/metal_bands_2017.csv",

                  low_memory=False,encoding='latin-1')
# remove any row which has NAN values and drop unnecessory column with drop function, axis=0 will drop rows.

bands=bands.dropna(axis=0,how='any')
#create a new data frame with band name and origin

g1=bands[['band_name','origin']]



# count the number of band names of country

g1=g1.groupby(['origin']).count()



g1=g1.rename(columns={'band_name':'Number_of_bands'})



g1=g1.sort_values('Number_of_bands')



# select top 20 countries with highest number of bands

g2=g1.tail(20)
plt.style.use('dark_background')
# top 20 countries in number of bands formed

g2.plot.bar(figsize=(8,8),color='green',)

plt.xlabel("Number of Bands")

plt.ylabel("Country of Origin")

plt.title("Top 20 countries with most number of bands from 1964 to 2016")
# pick up the desired columns and create a new data frame

g3=bands[['band_name','split']]



# remove all the lines which are missing details of split

g3=g3[g3.split!='-']



# groip all the splits based of the year

g3=g3.groupby(by=['split'],axis=0).count()



#rename the column

g3=g3.rename(columns={'band_name':'number_of_bands'})



# select the columns with more than 100 splits 

g3=g3[g3['number_of_bands']>100]
x=g3.index.values

y=g3.number_of_bands.values
g3.plot(kind='line',linewidth=2.0,linestyle='--',color='c',marker='.',markeredgewidth='2.5',markeredgecolor='b',figsize=(8,5),

        fontsize=12,)

plt.xlabel("Year Of split")

plt.title("Top 20 countries with most number of bands from 1964 to 2016")
# pick up the desired columns and create a new data frame

g4=bands[['band_name','split','formed']]



# remove all the lines which are missing details of split

g4=g4[g4.split!='-']



g4=g4.sort_values(axis=0,ascending=True,by='formed')



#reset the index

g4.index=pd.RangeIndex(len(g4.index))
# count the number of bands formed and split.

split=g4.groupby(by=['split'],axis=0,as_index=False)['band_name'].count()

created=g4.groupby(by=['formed'],axis=0,as_index=False)['band_name'].count()
#for ease of understanding we can rename the columns

split=split.rename(columns={'split':'year','band_name':"number_of_bands_Split"})

created=created.rename(columns={'formed':'year','band_name':"number_of_bands_formed"})
# join two dataframes

g4=pd.merge(split,created,on=['year','year'])
g4.plot.scatter('number_of_bands_Split','number_of_bands_formed',marker='8',c='r',figsize=(10,8),grid=True)
plt.figure(1,figsize=(15,20),edgecolor='m')

plt.subplot(121)

plt.barh(width=split['number_of_bands_Split'],bottom=split.year.astype(str).astype(int))

plt.ylabel("Year")

plt.xlabel("Bands Split")

plt.yticks(created.year.astype(str).astype(int))

plt.subplot(122)

plt.barh(width=created['number_of_bands_formed'],bottom=created.year.astype(str).astype(int))

plt.suptitle('Categorical Plotting of number of bands formed and split')

plt.ylabel("Year")

plt.xlabel("Bands formed")

plt.yticks(created.year.astype(str).astype(int))