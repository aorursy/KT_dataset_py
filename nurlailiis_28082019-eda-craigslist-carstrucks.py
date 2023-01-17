# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#import numpy and pandas

import numpy as np

import pandas as pd



#import visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



#import preprocessing

from sklearn import preprocessing
#import data craigslistVehiclesFull.csv

cvehiclesfull = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/craigslistVehiclesFull.csv')
#data information

cvehiclesfull.info()
#check length of columns and rows

cvehiclesfull.shape
#check statistical details

cvehiclesfull.describe()
#detect missing value

cvehiclesfull.isnull().sum().sort_values(ascending=False) 
#change data type of year from float to string

cvehiclesfull['year'] = cvehiclesfull.year.astype(str)

cvehiclesfull.info()
#detect percentage of missing value which above 60%

checknull = round(cvehiclesfull.isnull().sum()/len(cvehiclesfull)*100,2).sort_values(ascending=False)

checknull
#drop column url, image url, size, and vin

#drop url and image_url because unique value

df = cvehiclesfull.drop(columns=['url','image_url', 'size', 'vin'])

df.head()
#separate numerical and categorical feature

category = ['city','manufacturer','make','cylinders','fuel','title_status','transmission','drive','type','paint_color','county_name','condition','state_code','state_name','year']

numerical = df.drop(category, axis=1)

categorical = df[category]

numerical.head()
categorical.head()
categorical.info()
numerical.info()
#fill value in numerical with mean

for num in numerical:

    mean = numerical[num].mean()

    numerical[num]=numerical[num].fillna(mean) 
#detect numerical missing value

numerical.isnull().sum().sort_values(ascending=False) 
#fill value in categorical with mode

for cat in categorical:

    mode = categorical[cat].mode().values[0]

    categorical[cat]=df[cat].fillna(mode)
#detect categorical missing value

categorical.isnull().sum().sort_values(ascending=False) 
categorical.head()
numerical.head()
#concat table categorical and numerical to create new table without missing values

df2 = pd.concat([categorical,numerical],axis=1)

df2.head()
#outlier detection in numerical

fig=plt.figure(figsize=(13,12))

axes=330

#put data numerical

for num in numerical:

    axes += 1

    fig.add_subplot(axes)

    #set title of num

    sns.boxplot(data = numerical, x=num, color="y") 

plt.show()
filterprice=df2[df2['price']<15000]

fig=plt.figure(figsize=(15,5))

fig.add_subplot(1,2,1)

sns.boxplot(numerical['price'])

plt.title('Price before dropped under 15000')

fig.add_subplot(1,2,2)

sns.boxplot(filterprice['price'], color="y")

plt.title('Price after dropped under 15000')
#drop column long lat and odometer because too many outliers

dfinal = df2.drop(columns=['long', 'lat', 'odometer'])

dfinal.head()
#filter by year 2010

data_year = dfinal[dfinal['year']=='2010.0']

top=data_year.sort_values('price',ascending=False).head(5)

toplabel=top[['manufacturer','price']]



plt.figure(figsize=(12,6))



x=range(5)

plt.bar(x,top['price']/6**9, color=['y', 'y', 'y', 'y', 'y'])

plt.xticks(x,top['manufacturer'])

plt.xlabel('Manufacturer of Cars')

plt.ylabel('Price')

plt.title('5 Most Highest Price of Manufacturer Cars in 2010')

plt.show()

toplabel.head()
joins = dfinal[['manufacturer','type','price']]

join_group = joins.groupby('type').mean().head(5)

join_group
plt.figure(figsize=(12,6))



x=range(5)

plt.bar(x,join_group['price']/6**9, color="y")

plt.xticks(x,join_group.index)

plt.xlabel('Type of Cars')

plt.ylabel('Price')

plt.title('5 Most Highest Price Based on Type of Cars')

plt.show()
#create correlation with hitmap



#create correlation

corr = dfinal.corr(method = 'pearson')



#convert correlation to numpy array

mask = np.array(corr)



#to mask the repetitive value for each pair

mask[np.tril_indices_from(mask)] = False

fig, ax = plt.subplots(figsize = (15,12))

fig.set_size_inches(20,5)

sns.heatmap(corr, mask = mask, vmax = 0.9, square = True, annot = True, color="y")
data_type = dfinal[dfinal['type']=='sedan']



from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,6))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(data_type['year'],data_type['county_fips'],data_type['price'], s=30, color="y")

plt.show()
fig=plt.figure(figsize=(15,10))

fig.add_subplot(2,2,1)

sns.distplot(filterprice['county_fips'], color="y", kde=False)

plt.title('Histogram of Federal Information Processing Standards code')



fig.add_subplot(2,2,2)

sns.distplot(filterprice['weather'], color="y", kde=False)

plt.title('Histogram of historical average temperature for location in October/November')



fig.add_subplot(2,2,3)

sns.boxplot(filterprice['price'], color="y")

plt.title('Histogram of vehicles price')



plt.show()