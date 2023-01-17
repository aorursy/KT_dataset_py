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
df=pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
df.head() #check exp default 5
df.corr() #show chart 

f,ax=plt.subplots(figsize=(12,12))

sns.heatmap(df.corr(),annot=True,linewidth=.5,fmt='.2f',ax=ax) #annot =numbered notation,fmt=digit
df.columns
df['NA_Sales'].plot(kind='line',figsize = (12,6),color='yellow',label='NA',linewidth=1,alpha = 0.5,grid = True,linestyle = ':',fontsize='12')

df['EU_Sales'].plot(kind='line',figsize = (12,6),color='black',label='EU',linewidth=1,alpha = 0.5,grid = True,linestyle = '-.',fontsize='12')

plt.legend(loc='upper right') 

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
df.plot(kind='scatter', x='NA_Sales', y='EU_Sales',alpha = 0.5,color = 'red',figsize = (5,5))

plt.xlabel('NA_Sales')              # label = name of label

plt.ylabel('EU_Sales')

plt.title('NA EU Sales - Scatter Plot') 
df['NA_Sales'].plot(kind = 'hist',bins = 50,figsize = (5,5))

plt.show()
df['EU_Sales'].plot(kind = 'hist',bins = 50,figsize = (5,5))

plt.show()
#df[(df['NA_Sales']>40) & (df['EU_Sales']<40)]

df[np.logical_and((df['NA_Sales']>40) , (df['EU_Sales']<40))]

for index,value in df[['NA_Sales']][0:4].iterrows():

    print(index," : ",value)
average=sum(df['NA_Sales'])/len(df['NA_Sales'])

print(average)

df['price-level']=["expensive" if i > average else "cheap" for i in df['NA_Sales']]

df.loc[0:,['price-level','NA_Sales']]['price-level']
df.info()
print(df['Genre'].value_counts(dropna=False))
df.describe()
df.boxplot(column='Year')

plt.show()
df.info()
data=df.head()

data
melted=pd.melt(frame=data,id_vars='Name',value_vars=['Genre','price-level'])

melted
melted.pivot(index='Name',columns='variable',values='value')
data1=df.tail()

data2=df.head()



conc_data_row=pd.concat([data1,data2],axis=0,ignore_index=True)# axis = 0 : adds dataframes in row

conc_data_row



data1=df['Genre'].head()

data2=df['Year'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
df["Year"].value_counts(dropna =False)

df[df['Year'].isnull()]

dat=df

dat['Year'].dropna(inplace=True)
assert  dat['Year'].notnull().all() # returns nothing because we drop nan values
