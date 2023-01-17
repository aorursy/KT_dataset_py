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
import pandas as pd

prices = pd.read_csv("../input/dwdm-petrol-prices/Petrol Prices.csv")
df = pd.DataFrame(prices)

df.head(7)

df.tail(4)
import matplotlib.pyplot as plt

data3 = df.copy()

data3["Date"]= data3["Date"].str.split(" ", expand = True)

data3['Date'].value_counts().plot(kind='bar' , title=" Records for each month",

           figsize=(12,8))

plt.ylabel("Fequency")

plt.xlabel("Months")
#y=[]

#data4.at[143,'Date']= 'Aug 18 2016'

#s = data4['Date']

#for i in range(229):

 #  l =time.mktime(datetime.datetime.strptime(s[i],"%b %d %Y").timetuple())

  #  y.append(l)

#for i in range(3):    

 #   y.append('NaN')

#print(y)
data4 = df.copy()



# new data frame with split value columns 

new = data4["Date"].str.split(" ",n = 3, expand = True) 

  

# making separate first name column from new data frame 

data4["Month"]= new[0] 

 

# making separate last name column from new data frame 

data4["Day"]= new[1] 

data4["Year"]= new[2]

#data4['TimeStamp'] = y



   

 

# df display 

data4
data4 = data4.fillna(0)

data4
data4['Day']= data4['Day'].apply(int)

data4['Year'] = data4['Year'].apply(int)



data4.dtypes

#use if lamba to change nan to 0 or change to int


import datetime

y=[]

length = len(data4)

date_text = data4['Date']

for i in range(length):

    if (date_text[i] == 0 or date_text[i] =='ug 18 2016'): 

        y.append(0)

    else: 

        l = datetime.datetime.strptime(date_text[i],"%b %d %Y")

        y.append(l)



data4['TimeStamp'] = y

data4
data4.replace(0, np.nan ,inplace=True)

data4['TimeStamp']=pd.to_datetime(data4['Date'], errors='coerce')

data4.dtypes
data2 = data4[['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery','TimeStamp']]

data2
data2.reindex(columns=['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery','TimeStamp'])

#lines = data2.plot.line()


data2.plot(kind="line", # or `us_gdp.plot.line(`

    x='TimeStamp',     

    y=['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery'],

     

    title="Gas Prices per Period",

    figsize=(25,20)

)

#plt.title("From %d to %d" % (

 #   data2['TimeStamp'].min(),

  #data2['TimeStamp'].max()

#),size=8)

plt.suptitle("Gas Prices per Period",size=12)

plt.ylabel("Gas Prices")
data2['Propane'].pct_change(periods= 4,fill_method='ffill')
data2['Propane'].pct_change(periods= 4,fill_method='ffill').plot( title="percentage change for every 4 time periods",

           figsize=(25,8))

plt.ylabel("Fequency of change")

plt.xlabel("Number of Change")
kdata= data4[['Gasolene_87', 'Gasolene_90', 'Month','Day','Year','TimeStamp']]

kdata
cluster_data = kdata[['Gasolene_87', 'Gasolene_90']]





cluster_data = cluster_data.fillna( kdata.median() )

#Get rid of missing data

missing_data_results = cluster_data.isnull().sum()



print(missing_data_results)
data_values = cluster_data.iloc[ :, :].values

data_values



from sklearn.cluster import KMeans



# Use the Elbow method to find a good number of clusters using WCSS (within-cluster sums of squares)

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
kmeans = KMeans(n_clusters=5, init="k-means++", n_init=10, max_iter=300) 

cluster_data["cluster"] = kmeans.fit_predict( data_values )

cluster_data
cluster_data['cluster'].value_counts()
cluster_data['cluster'].value_counts().plot(kind='bar',title='Distribution of Gas Prices across groups')

plt.xlabel("Clusters")

plt.ylabel("Frequency")
import seaborn as sns

sns.pairplot( cluster_data, hue="cluster")
grouped_cluster_data = cluster_data.groupby('cluster')

grouped_cluster_data
grouped_cluster_data.describe()


grouped_cluster_data.plot(subplots=True,)
#Average per year

kdata.groupby('Year')['Gasolene_87','Gasolene_90'].mean()