# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_path = '/kaggle/input/dwdm-petrol-prices/Petrol Prices.csv'

data = pd.read_csv(data_path) 

data.head(7)
data.tail(4)
data['Date']=data.Date.astype('datetime64')
Month = data.Date.dt.month

data['Month'] = Month

result = data.groupby('Month').Date.count()

result
import matplotlib.pyplot as plt
data.groupby('Month').Date.count().plot(kind="bar",

    title="Total Record per Month",

    figsize=(12,8)

)

plt.ylabel("Number of Records")
from datetime import time
Month = data.Date.dt.month_name()

data['Month'] = Month

Day = data.Date.dt.day

data['Day'] = Day

Year = data.Date.dt.year

data['Year'] = Year
data['TimeStamp']= data.Date.dt.time
data.describe()
Interesting_data = data[['Gasolene_87','Gasolene_90','Auto_Diesel','Kerosene','Propane','Butane','HFO','Asphalt','ULSD','87_Change','Ex_Refinery']]

df = pd.DataFrame(Interesting_data)

df

x=df.set_index(data['Date'])

x.plot(kind='line', figsize=(12,8))
print(x['Gasolene_90'].pct_change(periods=4))
x['Gasolene_90'].pct_change(periods=4).plot(kind='line', figsize=(12,8))
data1 = pd.DataFrame({

    'Gasolene_87':data['Gasolene_87'],

    'Gasolene_90':data['Gasolene_90'],

    'Month':data['Month'],

    'Day':data['Day'],

    'Year':data['Year'],

    'Timestamp':data['TimeStamp'],

})

data1
from sklearn.cluster import KMeans
data_values = data1.iloc[ :, [0,1]].values

data_values
wcss = []

for i in range( 1, 15 ):

    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300) 

    kmeans.fit_predict( data_values )

    wcss.append( kmeans.inertia_ )

    

plt.plot( wcss, 'ro-', label="WCSS")

plt.title("Computing WCSS for KMeans++")

plt.xlabel("Number of clusters")

plt.ylabel("WCSS")

plt.show()
data2 = data[['Gasolene_87','Gasolene_90','Auto_Diesel','Kerosene','Propane','Butane','HFO','Asphalt','ULSD','87_Change','Ex_Refinery']]
missing_data_results = data2.isnull().sum()

print(missing_data_results)
data2 = data2.fillna( data.median() )
data_values1 = data2.iloc[ :, :].values

data_values1


kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=300) 

data1["cluster"] = kmeans.fit_predict( data_values )

data1.head(25)
import seaborn as sns
sns.pairplot( data1, hue="cluster")
average = data.groupby('Year').mean()

average