# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
city_data=pd.read_csv('../input/air-quality-data-in-india/city_day.csv')
print('Shape of data is :  ',city_data.shape) #Shape of data.
print('Total null reocrds in Data :\n',city_data.isnull().sum())
city_data[city_data['AQI_Bucket']=='Severe']['City'].value_counts() #Citiwise instamces of Severe bucket of Air Quality index.
#here Ahmedabad,Delhi,Patna seems to be highets in case and Ahemdabad to be very much higher than others.
pd.to_datetime(city_data[(city_data['AQI_Bucket']=='Severe')&(city_data['City']=='Ahmedabad')]['Date']).dt.year.value_counts()
city_data['Date']=pd.to_datetime(city_data['Date'])
city_data[(city_data['AQI_Bucket']=='Severe')&(city_data['City']=='Ahmedabad')&(city_data.Date.dt.year!=2020)].Date.dt.month.value_counts()
city_data[(city_data.City=='Ahmedabad')&(city_data.Date.dt.year==2018)].groupby(city_data.Date.dt.month)['AQI'].sum()
city_data[(city_data.City=='Ahmedabad')&(city_data.Date.dt.year==2019)].groupby(city_data.Date.dt.month)['AQI'].sum()
mis_val=city_data.isnull().sum()
mis_val_percent=100*mis_val/len(city_data)
print(mis_val_percent)
Mis_val=pd.concat([mis_val,mis_val_percent],axis=1)
Mis_val=Mis_val.rename(columns={0 : 'Missing Values', 1 : '% of Total Values'})
Mis_val=Mis_val[Mis_val.iloc[:,1]!=0].sort_values(by='% of Total Values',ascending=False).style.background_gradient(cmap='Reds')
Mis_val
for i in range(2015,2020): 
    print('year :',i,'Missing Values\n',100*(city_data.groupby(city_data.Date.dt.year).get_group(i).isnull().sum()/city_data.groupby(city_data.Date.dt.year).get_group(i).shape[0]))
city_data['PM2.5'].describe()
100*(city_data['PM2.5'].isnull().sum()/city_data.shape[0]) #16% of the data points have PM2.5 values missing in whole dataset.
by_year=city_data.groupby([city_data.Date.dt.year]).mean()
by_month=city_data.groupby([city_data.Date.dt.month]).mean()
plt.xlabel('Month')
plt.ylabel('Mean_AQI')
plt.plot(by_month.index.get_level_values(0),by_month['PM2.5'])
plt.xlabel('Year')
plt.ylabel('Mean_AQI')
plt.plot(by_year.index.get_level_values(0),by_year['PM2.5']) #Ignore 2020 as data is whole year data is not yet available.
city_data_not_2020=city_data[city_data.Date.dt.year!=2020]
by_month_not_2020=city_data_not_2020.groupby([city_data_not_2020.Date.dt.month]).mean()
plt.plot(by_month_not_2020.index.get_level_values(0),by_month_not_2020['PM2.5']) #Data in march,April is higher as it was brought down by 2020 in previous graph.
plt.plot(city_data[city_data.City=='Chennai'].groupby([city_data.Date.dt.month]).mean().index.get_level_values(0),city_data[city_data.City=='Chennai'].groupby([city_data.Date.dt.month]).mean()['PM2.5'])
plt.plot(city_data[city_data.City=='Delhi'].groupby([city_data.Date.dt.month]).mean().index.get_level_values(0),city_data[city_data.City=='Delhi'].groupby([city_data.Date.dt.month]).mean()['PM2.5'])
plt.plot(city_data_not_2020[city_data_not_2020.City=='Delhi'].groupby([city_data_not_2020.Date.dt.month]).mean().index.get_level_values(0),city_data_not_2020[city_data_not_2020.City=='Delhi'].groupby([city_data_not_2020.Date.dt.month]).mean()['PM2.5'])
plt.plot(city_data[city_data.City=='Mumbai'].groupby([city_data.Date.dt.month]).mean().index.get_level_values(0),city_data[city_data.City=='Mumbai'].groupby([city_data.Date.dt.month]).mean()['PM2.5'])
plt.plot(city_data_not_2020[city_data_not_2020.City=='Mumbai'].groupby([city_data_not_2020.Date.dt.month]).mean().index.get_level_values(0),city_data_not_2020[city_data_not_2020.City=='Mumbai'].groupby([city_data_not_2020.Date.dt.month]).mean()['PM2.5'])
plt.plot(city_data_not_2020.groupby([city_data_not_2020.Date.dt.month]).mean().index.get_level_values(0),city_data_not_2020.groupby([city_data_not_2020.Date.dt.month]).mean()['CO'])
plt.plot(city_data_not_2020.groupby([city_data_not_2020.Date.dt.month]).mean().index.get_level_values(0),city_data_not_2020.groupby([city_data_not_2020.Date.dt.month]).mean()['CO'])
plt.plot(city_data_not_2020.groupby([city_data_not_2020.Date.dt.year]).mean().index.get_level_values(0),city_data_not_2020.groupby([city_data_not_2020.Date.dt.year]).mean()['CO'])
city_data_not_2020.groupby([city_data_not_2020.Date.dt.year]).mean()['CO']
plt.plot(city_data[city_data.Date.dt.year==2020].groupby([city_data.Date.dt.month]).mean().index.get_level_values(0),city_data[city_data.Date.dt.year==2020].groupby([city_data.Date.dt.month]).mean()['CO'])
plt.figure(figsize=(10,10))
city_data_not_2020.boxplot()
plt.figure(figsize=(15,10))
plt.plot(city_data[(city_data.Date.dt.month==2)&(city_data.Date.dt.year==2020)&(city_data.City=='Mumbai')]['Date'],city_data[(city_data.Date.dt.month==2)&(city_data.Date.dt.year==2020)&(city_data.City=='Mumbai')]['CO'])
plt.figure(figsize=(15,10))
plt.plot(city_data[(city_data.Date.dt.month==3)&(city_data.Date.dt.year==2020)&(city_data.City=='Mumbai')]['Date'],city_data[(city_data.Date.dt.month==3)&(city_data.Date.dt.year==2020)&(city_data.City=='Mumbai')]['CO'])
plt.figure(figsize=(15,10))
plt.plot(city_data[(city_data.Date.dt.month==4)&(city_data.Date.dt.year==2020)&(city_data.City=='Mumbai')]['Date'],city_data[(city_data.Date.dt.month==4)&(city_data.Date.dt.year==2020)&(city_data.City=='Mumbai')]['CO'])
Mean_co_for_Mumbai_Feb_2020=city_data[(city_data.Date.dt.month==2)&(city_data.Date.dt.year==2020)&(city_data.City=='Mumbai')]['CO'].mean()
Mean_co_for_Mumbai_March_2020=city_data[(city_data.Date.dt.month==3)&(city_data.Date.dt.year==2020)&(city_data.City=='Mumbai')]['CO'].mean()
Mean_co_for_Mumbai_April_2020=city_data[(city_data.Date.dt.month==4)&(city_data.Date.dt.year==2020)&(city_data.City=='Mumbai')]['CO'].mean()
print('Mean Carbon oxide in Feb 2020 in Mumbai',Mean_co_for_Mumbai_Feb_2020)
print('Mean Carbon oxide in March 2020 in Mumbai',Mean_co_for_Mumbai_March_2020)
print('Mean Carbon oxide in April 2020 in Mumbai',Mean_co_for_Mumbai_April_2020)
plt.figure(figsize=(15,10))
plt.plot(city_data[(city_data.Date.dt.month==2)&(city_data.Date.dt.year==2020)&(city_data.City=='Delhi')]['Date'],city_data[(city_data.Date.dt.month==2)&(city_data.Date.dt.year==2020)&(city_data.City=='Delhi')]['CO'])
plt.figure(figsize=(15,10))
plt.plot(city_data[(city_data.Date.dt.month==3)&(city_data.Date.dt.year==2020)&(city_data.City=='Delhi')]['Date'],city_data[(city_data.Date.dt.month==3)&(city_data.Date.dt.year==2020)&(city_data.City=='Delhi')]['CO'])
plt.figure(figsize=(15,10))
plt.plot(city_data[(city_data.Date.dt.month==4)&(city_data.Date.dt.year==2020)&(city_data.City=='Delhi')]['Date'],city_data[(city_data.Date.dt.month==4)&(city_data.Date.dt.year==2020)&(city_data.City=='Delhi')]['CO'])
Mean_co_for_Delhi_Feb_2020=city_data[(city_data.Date.dt.month==2)&(city_data.Date.dt.year==2020)&(city_data.City=='Delhi')]['CO'].mean()
Mean_co_for_Delhi_March_2020=city_data[(city_data.Date.dt.month==3)&(city_data.Date.dt.year==2020)&(city_data.City=='Delhi')]['CO'].mean()
Mean_co_for_Delhi_April_2020=city_data[(city_data.Date.dt.month==4)&(city_data.Date.dt.year==2020)&(city_data.City=='Delhi')]['CO'].mean()
print('Mean Carbon oxide in Feb 2020 in Delhi',Mean_co_for_Delhi_Feb_2020)
print('Mean Carbon oxide in March 2020 in Delhi',Mean_co_for_Delhi_March_2020)
print('Mean Carbon oxide in April 2020 in Delhi',Mean_co_for_Delhi_April_2020)
plt.figure(figsize=(15,10))
plt.title('Co emmision in Chennai in Month of February')
plt.plot(city_data[(city_data.Date.dt.month==2)&(city_data.Date.dt.year==2020)&(city_data.City=='Chennai')]['Date'],city_data[(city_data.Date.dt.month==2)&(city_data.Date.dt.year==2020)&(city_data.City=='Chennai')]['CO'])
plt.figure(figsize=(15,10))
plt.title('Co emmision in Chennai in Month of March')
plt.plot(city_data[(city_data.Date.dt.month==3)&(city_data.Date.dt.year==2020)&(city_data.City=='Chennai')]['Date'],city_data[(city_data.Date.dt.month==3)&(city_data.Date.dt.year==2020)&(city_data.City=='Chennai')]['CO'])
plt.figure(figsize=(15,10))
plt.title('Co emmision in Chennai in Month of April')
plt.plot(city_data[(city_data.Date.dt.month==4)&(city_data.Date.dt.year==2020)&(city_data.City=='Chennai')]['Date'],city_data[(city_data.Date.dt.month==4)&(city_data.Date.dt.year==2020)&(city_data.City=='Chennai')]['CO'])
Mean_co_for_Chennai_Feb_2020=city_data[(city_data.Date.dt.month==2)&(city_data.Date.dt.year==2020)&(city_data.City=='Chennai')]['CO'].mean()
Mean_co_for_Chennai_March_2020=city_data[(city_data.Date.dt.month==3)&(city_data.Date.dt.year==2020)&(city_data.City=='Chennai')]['CO'].mean()
Mean_co_for_Chennai_April_2020=city_data[(city_data.Date.dt.month==4)&(city_data.Date.dt.year==2020)&(city_data.City=='Chennai')]['CO'].mean()
print('Mean Carbon oxide in Feb 2020 in Chennai',Mean_co_for_Chennai_Feb_2020)
print('Mean Carbon oxide in March 2020 in Chennai',Mean_co_for_Chennai_March_2020)
print('Mean Carbon oxide in April 2020 in Chennai',Mean_co_for_Chennai_April_2020)
Mean_co_in_Feb_March_April_2020=pd.DataFrame()
for i in city_data.City.unique():
    if i!='Ahmedabad':
        Mean_co_in_Feb_March_April_2020[i]=[city_data[(city_data.Date.dt.month==2)&(city_data.Date.dt.year==2020)&(city_data.City==i)]['CO'].mean(),city_data[(city_data.Date.dt.month==3)&(city_data.Date.dt.year==2020)&(city_data.City==i)]['CO'].mean(),city_data[(city_data.Date.dt.month==4)&(city_data.Date.dt.year==2020)&(city_data.City==i)]['CO'].mean()]
Mean_co_in_Feb_March_April_2020.transpose().plot(figsize=(20,20),kind='bar')
city_data[city_data.City=='Talcher'].isnull().sum()/len(city_data[city_data.City=='Talcher']) #Missingh values related to CO is about 16% in whole dataset
city_data[(city_data.City=='Talcher')&(city_data.Date.dt.year==2019)].isnull().sum() 
#About 15 values are missing in year 2019 in record of CO.this is small value so it won't affect our mean that much.
plt.figure(figsize=(15,10))
plt.plot(city_data[(city_data.City=='Talcher')&(city_data.Date.dt.year==2019)]['Date'],city_data[(city_data.City=='Talcher')&(city_data.Date.dt.year==2019)]['CO'])
city_data[(city_data.City=='Talcher')&(city_data.Date.dt.year==2019)]['CO'].mean()
city_data[(city_data.City=='Talcher')&(city_data.Date.dt.year==2020)]['CO'].mean()
plt.plot(city_data_not_2020.groupby([city_data_not_2020.Date.dt.month]).mean().index.get_level_values(0),city_data_not_2020.groupby([city_data_not_2020.Date.dt.month]).mean()['AQI'])
Mean_AQI_in_Feb_March_April_2020=pd.DataFrame()
for i in city_data.City.unique():
    if i!='Ahmedabad':
        Mean_AQI_in_Feb_March_April_2020[i]=[city_data[(city_data.Date.dt.month==2)&(city_data.Date.dt.year==2020)&(city_data.City==i)]['AQI'].mean(),city_data[(city_data.Date.dt.month==3)&(city_data.Date.dt.year==2020)&(city_data.City==i)]['AQI'].mean(),city_data[(city_data.Date.dt.month==4)&(city_data.Date.dt.year==2020)&(city_data.City==i)]['AQI'].mean()]
Mean_AQI_in_Feb_March_April_2020.transpose().plot(figsize=(20,20),kind='bar')
Mean_AQI_in_Feb_March_April_2019=pd.DataFrame()
for i in city_data.City.unique():
    if i!='Ahmedabad':
        Mean_AQI_in_Feb_March_April_2019[i]=[city_data[(city_data.Date.dt.month==2)&(city_data.Date.dt.year==2019)&(city_data.City==i)]['AQI'].mean(),city_data[(city_data.Date.dt.month==3)&(city_data.Date.dt.year==2019)&(city_data.City==i)]['AQI'].mean(),city_data[(city_data.Date.dt.month==4)&(city_data.Date.dt.year==2019)&(city_data.City==i)]['AQI'].mean()]
Mean_AQI_in_Feb_March_April_2019.transpose().plot(figsize=(20,20),kind='bar')
plt.plot(city_data[city_data.Date.dt.year!=2020].groupby([city_data.Date.dt.year]).mean().index.get_level_values(0),city_data[city_data.Date.dt.year!=2020].groupby([city_data.Date.dt.year]).mean()['AQI'])
plt.plot(city_data[city_data.Date.dt.year==2020].groupby([city_data.Date.dt.month]).mean().index.get_level_values(0),city_data[city_data.Date.dt.year==2020].groupby([city_data.Date.dt.month]).mean()['AQI'])
Mean_no_in_Feb_March_April_2020=pd.DataFrame()
for i in city_data.City.unique():
    if i!='Ahmedabad':
        Mean_no_in_Feb_March_April_2020[i]=[city_data[(city_data.Date.dt.month==2)&(city_data.Date.dt.year==2020)&(city_data.City==i)]['NO'].mean(),city_data[(city_data.Date.dt.month==3)&(city_data.Date.dt.year==2020)&(city_data.City==i)]['NO'].mean(),city_data[(city_data.Date.dt.month==4)&(city_data.Date.dt.year==2020)&(city_data.City==i)]['NO'].mean()]
Mean_no_in_Feb_March_April_2020.transpose().plot(figsize=(20,20),kind='bar')
Mean_no_in_Feb_March_April_2019=pd.DataFrame()
for i in city_data.City.unique():
    if i!='Ahmedabad':
        Mean_no_in_Feb_March_April_2019[i]=[city_data[(city_data.Date.dt.month==2)&(city_data.Date.dt.year==2019)&(city_data.City==i)]['NO'].mean(),city_data[(city_data.Date.dt.month==3)&(city_data.Date.dt.year==2020)&(city_data.City==i)]['NO'].mean(),city_data[(city_data.Date.dt.month==4)&(city_data.Date.dt.year==2019)&(city_data.City==i)]['NO'].mean()]
Mean_no_in_Feb_March_April_2019.transpose().plot(figsize=(20,20),kind='bar')
Mean_no_in_Feb_March_April_2018=pd.DataFrame()
for i in city_data.City.unique():
    if i!='Ahmedabad':
        Mean_no_in_Feb_March_April_2018[i]=[city_data[(city_data.Date.dt.month==2)&(city_data.Date.dt.year==2018)&(city_data.City==i)]['NO'].mean(),city_data[(city_data.Date.dt.month==3)&(city_data.Date.dt.year==2020)&(city_data.City==i)]['NO'].mean(),city_data[(city_data.Date.dt.month==4)&(city_data.Date.dt.year==2018)&(city_data.City==i)]['NO'].mean()]
Mean_no_in_Feb_March_April_2018.transpose().plot(figsize=(20,20),kind='bar')
plt.xlabel('Months')
plt.ylabel('AQI of India')
plt.plot(city_data[city_data.Date.dt.year==2020].groupby(city_data.Date.dt.month)['AQI'].mean(),label='2020')
plt.plot(city_data[city_data.Date.dt.year==2019].groupby(city_data.Date.dt.month)['AQI'].mean(),label='2019')
plt.plot(city_data[city_data.Date.dt.year==2018].groupby(city_data.Date.dt.month)['AQI'].mean(),label='2018')
plt.plot(city_data[city_data.Date.dt.year==2017].groupby(city_data.Date.dt.month)['AQI'].mean(),label='2017')
plt.plot(city_data[city_data.Date.dt.year==2016].groupby(city_data.Date.dt.month)['AQI'].mean(),label='2016')
plt.plot(city_data[city_data.Date.dt.year==2015].groupby(city_data.Date.dt.month)['AQI'].mean(),label='2015')

plt.legend()
plt.show()
#CO levels
plt.xlabel('Months')
plt.ylabel('Co levels of India')

plt.plot(city_data[city_data.Date.dt.year==2020].groupby(city_data.Date.dt.month)['CO'].mean(),label='2020')
plt.plot(city_data[city_data.Date.dt.year==2019].groupby(city_data.Date.dt.month)['CO'].mean(),label='2019')
plt.plot(city_data[city_data.Date.dt.year==2018].groupby(city_data.Date.dt.month)['CO'].mean(),label='2018')
plt.plot(city_data[city_data.Date.dt.year==2017].groupby(city_data.Date.dt.month)['CO'].mean(),label='2017')
plt.plot(city_data[city_data.Date.dt.year==2016].groupby(city_data.Date.dt.month)['CO'].mean(),label='2016')
plt.plot(city_data[city_data.Date.dt.year==2015].groupby(city_data.Date.dt.month)['CO'].mean(),label='2015')

plt.legend()
plt.show()
city_data[city_data.City=='Mumbai'][(city_data.Date.dt.year==2020)&(city_data.Date.dt.month==3)]['AQI']
plt.figure(figsize=(20,10))
plt.plot(city_data[city_data.City=='Bengaluru'][(city_data.Date.dt.year==2020)]['AQI'].reset_index(drop='True'),label='Bengaluru')
plt.plot(city_data[city_data.City=='Mumbai'][(city_data.Date.dt.year==2020)]['AQI'].reset_index(drop='True'),label='Mumbai')
plt.plot(city_data[city_data.City=='Delhi'][(city_data.Date.dt.year==2020)]['AQI'].reset_index(drop='True'),label='Delhi')
plt.legend()
plt.show()
plt.figure(figsize=(20,10))
plt.plot(city_data[city_data.City=='Bengaluru'][(city_data.Date.dt.year==2020)]['CO'].reset_index(drop='True'),label='Bengaluru')
plt.plot(city_data[city_data.City=='Mumbai'][(city_data.Date.dt.year==2020)]['CO'].reset_index(drop='True'),label='Mumbai')
plt.plot(city_data[city_data.City=='Delhi'][(city_data.Date.dt.year==2020)]['CO'].reset_index(drop='True'),label='Delhi')
plt.legend()
plt.show()
plt.figure(figsize=(20,10))
plt.plot(city_data[city_data.City=='Bengaluru'][(city_data.Date.dt.year==2020)]['CO'].reset_index(drop='True'),label='Chennai')
plt.plot(city_data[city_data.City=='Mumbai'][(city_data.Date.dt.year==2020)]['CO'].reset_index(drop='True'),label='Amritsar')
plt.plot(city_data[city_data.City=='Delhi'][(city_data.Date.dt.year==2020)]['CO'].reset_index(drop='True'),label='Chandigad')
plt.legend()
plt.show()
plt.figure(figsize=(20,10))
plt.plot(city_data[city_data.City=='Bengaluru'][(city_data.Date.dt.year==2020)]['AQI'].reset_index(drop='True'),label='Chennai')
plt.plot(city_data[city_data.City=='Mumbai'][(city_data.Date.dt.year==2020)]['AQI'].reset_index(drop='True'),label='Amritsar')
plt.plot(city_data[city_data.City=='Delhi'][(city_data.Date.dt.year==2020)]['AQI'].reset_index(drop='True'),label='Chandigad')
plt.legend()
plt.show()