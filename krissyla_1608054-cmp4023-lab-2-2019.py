# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
path = '/kaggle/input/dwdm-petrol-prices/Petrol Prices.csv'

data = pd.read_csv(path)
data.head(7)
data.tail(4)
from pandasql import sqldf # XXXXXXX
data['Months'] = pd.DatetimeIndex(data['Date']).month 

data
print(data.loc[143]) #locate record
data.iat[143,0] ='Aug 18 2016' #change value
print(data.loc[143])#locate cell
data['Months'] = pd.DatetimeIndex(data['Date']).month 

data
groupy = data.groupby("Months").count()

groupy
#import calendar
#data['Months'] = calendar.month_name('Months')

#data
groupy.plot(kind='bar',title='Data Grouping by Month',figsize=(20,20))
data['Day']=pd.DatetimeIndex(data['Date']).day

data['Year']=pd.DatetimeIndex(data['Date']).year

data['Months'].fillna('NULL',inplace = True)

data['Day'].fillna('NULL',inplace = True)

data['Year'].fillna('NULL',inplace = True)

data
data['Months'] = data['Months'].astype(str)

data.dtypes
data['Timestamp'] = " "
data['Timestamp']= pd.to_datetime(data['Date'])
from datetime import datetime
data['Timestamp'] = pd.to_datetime(data['Date']).values.astype(datetime)
data
extract = data[['Gasolene_87', 'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery']]

extract
extract = data[['Gasolene_87' ,'Gasolene_90', 'Auto_Diesel', 'Kerosene', 'Propane', 'Butane', 'HFO', 'Asphalt', 'ULSD', 'Ex_Refinery','Timestamp']]

extract
extract.plot(kind='line',title='Gas prices over the Period',x='Timestamp',

             y=['Gasolene_87' ,'Gasolene_90','Kerosene', 'Propane' ,'Butane' ,'HFO', 'Asphalt' ,'ULSD' ,'Ex_Refinery'],

             figsize=(18,18))
percentage = data['Kerosene'].pct_change(periods = 4) #pandas.DataFrame

percentage
percentage.plot(kind='line', title='percentage change(y) for four time periods(x)',figsize=(18,18))

newdataset = {'Propane' :data.Propane ,'Gasolene_90' :data.Gasolene_90, 'Month' :data.Months,

               'Day' :data.Day, 'Year' :data.Year, 'Timestamp' :data.Timestamp} 

newdataset

data2 = pd.DataFrame(newdataset)

data2
data3 = data2.drop(data2.index[[229,230,231]])#error in using null values

data3  #drop last 3 null rows
from sklearn.cluster import KMeans
interestdataset = {'Propane' :data3.Propane ,'Gasolene_90' :data3.Gasolene_90} #dataset with just two columns for kmmeans

interestdataset
data4 = pd.DataFrame(interestdataset) #dataframe with that dataset

data4
Mean=KMeans(n_clusters=10, init='k-means++', n_init=10, max_iter=6)

data3['Cluster#'] = Mean.fit_predict(data4) #assign result from kmeans to the extra row

data3
frame2 = frame.drop(frame.index[[229,230,231]])#error in using null values

frame2  #drop last 3 null rows
#suppose to be cluster# column from previous output
import seaborn as sns
plt.style.use('dark_background')
frame2.dtypes #an error was discovered after the plot ran
frame2['Propane']= frame2['Propane'].astype(int)

frame2['Gasolene_90']= frame2['Gasolene_90'].astype(int)

frame2
frame2.plot(kind='scatter', x='Propane' , y= 'Gasolene_90', c=Mean.labels_, cmap='rainbow', figsize=(16,16))
#a

grouping = data.groupby('Year')
#grouping.mean() gave an error

#changing columns to int

data.dtypes
data
data['Propane']= data['Propane'].astype(int)

data['Gasolene_90']= data['Gasolene_90'].astype(int)

data['Gasolene_87']= data['Gasolene_87'].astype(int)

data['Auto_Diesel']= data['Auto_Diesel'].astype(int)

data['Kerosene']= data['Kerosene'].astype(int)

data['Butane']= data['Butane'].astype(int)

data['HFO']= data['HFO'].astype(int)

data['Asphalt']= data['Asphalt'].astype(int)

data['ULSD']= data['ULSD'].astype(int)

data['87_Change']= data['87_Change'].astype(int)

data['SCT']= data['SCT'].astype(int)

data['Ex_Refinery']= data['Ex_Refinery'].astype(int)

data
grouping.mean()