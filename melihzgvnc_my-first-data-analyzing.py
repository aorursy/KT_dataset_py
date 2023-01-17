# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization with graphics

import seaborn as sns # visualization

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/vgsales.csv")
data.info()
data.head()
data.tail()
data.columns
data.isna().sum() #Found missing values in datasets and that count by sum() function.
data.dropna(inplace=True,axis=0) #inplace -> Bool value. Default False. 

                                 #If True, do operation inplace and return None.

                                 #axis=0 drop rows which contain missing values. 

                                 #axis=1 drop columns which contain missing values.
data.isna().sum() 
data.info()
size=[15,15]

plt.figure(figsize=size)

plt.title('Correlation Map')

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.2f',linecolor='black', cmap='Blues')

plt.show()
plt.figure(figsize=(15,15))

plt.bar(data.Genre,data.Global_Sales,alpha=0.1)

plt.title('Bar Plot')

plt.xlabel("Genres")

plt.ylabel("Global Sales")

plt.show()
plt.figure(figsize=(10,5))

plt.scatter(data.Global_Sales,data.Year,color="r",alpha=0.1)

plt.title('Scatter Plot')

plt.xlabel("Global Sales")

plt.ylabel("Years")

plt.show()

# axis=0 --> for line

# axis=1 --> for column 

dataGenre1=data.groupby("Genre")[['Global_Sales']].mean()

dataGenre2=data.groupby("Genre")[['NA_Sales']].mean()

dataGenre3=data.groupby("Genre")[['EU_Sales']].mean()

dataGenre4=data.groupby("Genre")[['JP_Sales']].mean()



dataGenre = pd.concat([dataGenre1,dataGenre2,dataGenre3,dataGenre4],axis=1)

dataGenre.info()
dataGenre
dataGenre['Sales_Status']=['Successful' if(i>0.5) else 'Unsuccessful' for i in dataGenre['Global_Sales']]

dataGenre
dataPlatform=data.groupby("Platform")[['Platform']].count()

dataPlatform.rename(columns={'Platform':'Counted'})

print(data['Genre'].value_counts(dropna=False)) #value_counts() -> Boolean, default True.

                                                #Don't include counts of NaN
data.describe()
data['Sales_Status']=['Successful' if(i>0.5) else 'Unsuccessful' for i in data['Global_Sales']]

data.head()
data.boxplot(column='Year',by='Sales_Status')
data.head()
data_new = data.head()

data_new
melted = pd.melt(frame= data_new, id_vars='Name', value_vars=['Genre','Publisher'])

melted
melted.pivot(index='Name', columns='variable',values='value')
data1 = data.head()

data2 = data.tail()

concat_data_row = pd.concat([data1,data2], axis=0, ignore_index=True)

concat_data_row
data1=data['Genre'].head()

data2=data['Sales_Status'].head()

concat_data_column = pd.concat([data1,data2], axis=1, ignore_index=True)

concat_data_column
data['Year'] = data['Year'].astype('int') #convert float to int
data['Genre'].value_counts(dropna = False)
#assert 1==1 # return nothing because it is true

#assert 1==2 # return error because it is false
assert data['Genre'].notnull().all() #returns nothing because we drop missing values

assert data.columns[1] == 'Name'
country = ['Greece', 'Italy']

city = ['Athen', 'Milano']

title = ['country', 'city']

list_col = [country,city]

zipped = list(zip(title, list_col))

data_dict = dict(zipped)

df = pd.DataFrame(data_dict)

df
#broadcasting

df['income']=1000

df
data.plot(kind='hist', y='Global_Sales', bins=50, range=(0,1), normed=True)

plt
data.plot(kind='hist', y='Global_Sales', bins=50, range=(0,1), normed=True, cumulative=True) # cumulative=True -> 

plt
data.head()
import warnings

warnings.filterwarnings('ignore')



data_indexing = data.head()

date_list=['1997-01-02', '1997-05-12', '1997-09-23', '1997-10-12', '1997-12-27']

date_object = pd.to_datetime(date_list)

data_indexing['date']=date_object



data_indexing = data_indexing.set_index('date')

data_indexing
print(data_indexing.loc['1997-01-02'])
print(data_indexing.loc['1997-01-02':'1997-12-27'])
data_indexing.head()
data_indexing.resample("M").mean()
data_indexing.resample("M").mean().interpolate("linear")