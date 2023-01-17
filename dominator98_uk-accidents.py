# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/Accidents0515.csv"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/Accidents0515.csv')
df.head()
#Extracting year from date

df['Year']=df['Date'].map(lambda x: x[6:10])
df.shape
del df['Date']
df.info()
#as year is an object we first convert it into integer type

df['Year']=df['Year'].astype(int)
#checking the null values and dropping them

df.isnull().sum()

df=df.dropna()

df.shape
# as the dataset is too large I'm using just from 2010-2015

df=df[(df.Year>=2010)&(df.Year<2016)]
# The latitude and longitude are arranged according to same city in the index, therefore I shuffle the data to get random cities 

df=df.sample(frac=1)
#Now I extract only 2500 rows as there is a limitation of only 2500 requests per day

df1=df[0:2500]

df1.shape
df1.head()
from pygeocoder import Geocoder # added a PR for this 

arr=[]

for i, row in df1.iterrows():

        result = Geocoder.reverse_geocode(df1['Latitude'][i], df1['Longitude'][i])

        print(result.city)

        arr.append(result.city)

        
list(arr)

se = pd.Series(arr)

df1['cities'] = se.values

df1=df1.dropna()

#city wise visualisation can be done