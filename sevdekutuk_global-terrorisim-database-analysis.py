# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/gtd/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1') 

#The csv file was read, transferred to the data.
print(type(data)) #type of data
data.dtypes #content of data and data types
data.info() # Information about data
data.columns # Column names of data
data.corr() #return correlations between features
f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True,linewidth=.7,fmt= '.1f',ax=ax)

plt.show()
data.head(15) #Indicates the first 10 values.
data.latitude.plot(kind = 'line', color = 'g',label = 'latitude',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.longitude.plot(kind = 'line',color = 'r',label = 'longitude',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     

plt.xlabel('latitude')              

plt.ylabel('longitude')

plt.title('Line Plot')           

plt.show()
#Relationship between latitude and longitude



data.plot(kind='scatter', x='latitude', y='longitude',alpha = 0.5,color = 'red')

plt.xlabel('latitude')              

plt.ylabel('longitude')

plt.title('latitude longitude Scatter Plot')   



plt.show()
# A different use of it scatter plot

plt.scatter(data.latitude,data.longitude,color="red",alpha=0.5)

plt.show()
# Histogram

data.longitude.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
data.longitude.plot(kind = 'hist',bins = 50)

plt.clf() #cleans it up again you can start a fresh