#Exploring the dataset





import pandas as pd

import numpy as np

import matplotlib.pyplot as plt





%matplotlib inline
#read csv document

df = pd.read_csv('../input/2007-2016-Homelessnewss-USA.csv')
#explore the head

df.head()
#explore the tail

df.tail()
#what are the types of data

df.dtypes
#change count to int/float



df['Count'] = df['Count'].str.replace(',','').astype(np.int64)



#change year to date

df['Year'] = pd.to_datetime(df['Year'])

#Check the data type 

df.dtypes
#copy a new data in a varible

newdf = df.copy()
newdf.head(2)

#Delete Coc Number and CoC Name

newdf.drop(['CoC Number','CoC Name'],axis=1 )
#plotting a car chart to see max number of homelessness according to state

newdf.State.value_counts().plot(kind='bar',title='Total number of homeless people according to states',figsize=(20,20), colormap= 'pink',fontsize=18)

plt.xlabel('States')

plt.ylabel('Number of people ')

#From the chart above we can see that California has most poplution that are homeless
newdf.Measures.value_counts()
newdf.Measures.value_counts().plot(kind='bar',title='Meseasures of homelessness ',figsize=(20,20),colormap= 'pink',fontsize=18)

plt.xlabel('Measures')

plt.ylabel('Number of people ')
newdf.Year.value_counts().plot('bar',colormap= 'pink',figsize=(20,20),fontsize=18)
# we can see from the chart above that most people were homeless on 2015. 
#exploring DC

explr_dc = newdf[newdf['State']=='DC']



explr_dc.groupby(['Year', 'Measures']).sum().unstack().plot.bar(stacked = True,title='Years with measures', colormap= 'Dark2',figsize=(30,30),fontsize=24)





#total number of people in DC who were homeless

explr_dc['Count'].sum()
