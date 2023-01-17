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
cars = pd.read_csv ('/kaggle/input/craigslist-carstrucks-data/vehicles.csv')
cars.info()

cars.head()
#Dropping year column since whole data was for the year 1970

del  cars['id'],cars['url'],cars ['region_url'],cars['description'],cars['lat'],cars['long'],cars['image_url'], cars['vin'],cars['county'], cars['title_status'],cars['size'],cars['condition']
cars.info()

cars.head()
cars['year'] = pd.to_datetime (cars['year'])
cars['year'] = cars['year'].astype ('str')

cars['year'] = cars['year'].apply (lambda x: x.split()[0])

cars['year'] = pd.to_datetime (cars['year'])
Total = cars.isnull().sum().sort_values(ascending=False)

percent = ((cars.isnull().sum()/cars.isnull().count())*100).sort_values(ascending=False)

missing_data = pd.concat([Total,percent],axis=1,keys=['Total','percent'])

missing_data
cars.dropna(inplace = True)

cars['cylinders'] = cars['cylinders'].apply (lambda x: x.replace('other','0 other'))

cars['cylinders'] = cars['cylinders'].apply (lambda x: x.split()[0])
cars['cylinders'] = cars['cylinders'].astype('int')

cars.drop (cars [cars['cylinders'] == 0 ].index,inplace = True)
#Since while data is extracted for the year 1970, dropping that column 

del cars['year']
cars.shape
cars.head()
cars.describe()
print ("Mean : " , int (cars['price'].mean()))

print ("Mode : " , cars['price'].mode())

print ("Median : " , cars['price'].median())
cars.head()
import matplotlib.pyplot as plt

%matplotlib inline
plt.figure (figsize = (12,5))

plt.ylabel ('Price',size =15)

plt.xlabel ('# of Cylinders',size=15)

plt.title('Price Variation of Cycinders',size=25)

plt.xticks(size=10)

plt.yticks(size=10)

Cylin_df = cars.groupby ('cylinders')['price'].count()

plt.plot(Cylin_df)

plt.show()

plt.figure (figsize = (12,5))

plt.ylabel ('Price',size =15)

plt.xlabel ('Transmission Type',size=15)

plt.title('Price Variation of Transmission Type',size=25)

plt.xticks(size=10)

plt.yticks(size=10)

Cylin_df = cars.groupby ('transmission')['price'].count()

plt.plot(Cylin_df)

plt.show()
plt.figure (figsize = (12,5))

plt.ylabel ('Price',size =15)

plt.xlabel ('Drive Type',size=15)

plt.title('Price Variation of Drive Type',size=25)

plt.xticks(size=10)

plt.yticks(size=10)

Cylin_df = cars.groupby ('drive')['price'].count()

plt.plot(Cylin_df)

plt.show()
plt.figure (figsize = (15,5))

plt.ylabel ('Price',size =15)

plt.xlabel ('Type',size=15)

plt.title('Price Variation Vs Type',size=25)

plt.xticks(size=10)

plt.yticks(size=10)

Cylin_df = cars.groupby ('type')['price'].count()

plt.plot(Cylin_df)

plt.show()
plt.figure (figsize = (20,5))

plt.ylabel ('Price',size =15)

plt.xlabel ('Manufacturer',size=15)

plt.title('Price Variation Vs Manufacturer',size=25)

plt.xticks(size=10)

plt.yticks(size=10)

plt.xticks(rotation=45)

Cylin_df = cars.groupby ('manufacturer')['price'].count()

plt.plot(Cylin_df)

plt.show()