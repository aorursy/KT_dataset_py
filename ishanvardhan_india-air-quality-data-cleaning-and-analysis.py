# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#importing the data as a Pandas DataFrame
dataset=pd.read_csv('../input/data.csv',encoding="ISO-8859-1")
dataset.describe()


dataset.columns
#Apart from the major pollutants, there are columns that refer to the respective states, agencies, sampling dates and the type.
#We will now have a look at what kind of data each of the columns consists of.

dataset.info()

#Now, we can immediatly see that there are quite a few nulls in various columns, which need work and first need a closer inspection.
dataset.head()

dataset.drop(['stn_code','agency','sampling_date','location_monitoring_station'],axis=1,inplace=True)
dataset.info()
dataset.head()
#Fixing the missing values firstly for all the pollutants.
#We will consider taking mean for all the pollutants columns and make use of the Imputer class
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(dataset.iloc[:, 3:8].values)
dataset.iloc[:,3:8] = imputer.transform(dataset.iloc[:, 3:8].values)
dataset.info()
dataset.head()
#Fixing the missing values in the column 'type'
dataset['type'].describe()
#With 10 Unique labels, we will fill the null values by the most common type, which is 'Residential, Rural and Other Areas'.
common_value='Residential,Rural and other Areas'
dataset['type']=dataset['type'].fillna(common_value)
dataset.info()
#Importing libraries
import matplotlib.pyplot as plt
import seaborn as sns
#We will start with pairplots to undestand the statistics and get a general idea about the interdependence of pollutants.
sns.pairplot(dataset[['so2','no2','pm2_5']])


sns.pairplot(dataset[['so2','no2','spm']])

sns.pairplot(dataset[['so2','no2','rspm']])

sns.pairplot(dataset[['rspm','spm','pm2_5']])


fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
sns.distplot(dataset['no2'],hist=True,kde=True,
             color='darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth':4},
             ax=axarr[0][0])

sns.distplot(dataset['so2'],hist=True,kde=True,
             color='red',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth':4},
             ax=axarr[0][1])

sns.distplot(dataset['rspm'],hist=True,kde=True,
             color='green',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth':4},
             ax=axarr[1][0])

sns.distplot(dataset['spm'],hist=True,kde=True,
             color='black',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth':4},
             ax=axarr[1][1])
sns.heatmap(
    dataset.loc[:, ['state','so2', 'no2', 'rspm', 'spm', 'pm2_5']].corr(),
    annot=True
)


statewise_emmissions = dataset.groupby('state').mean()[['so2', 'no2', 'rspm', 'spm', 'pm2_5']]
statewise_emmissions.plot.area()
 

statewise_emmissions.describe()

Top10States_with_highest_No2=statewise_emmissions.sort_values(by='no2',ascending=False).head(10)
Top10States_with_highest_No2_sorted=Top10States_with_highest_No2.loc[:,['no2']]
Top10States_with_highest_No2_sorted.head()
Top10states_with_highest_So2=statewise_emmissions.sort_values(by='so2',ascending=False).head(10)
Top10states_with_highest_So2_sorted=Top10states_with_highest_So2.loc[:,['so2']]
Top10states_with_highest_So2_sorted.head()
Top10states_with_highest_rspm=statewise_emmissions.sort_values(by='rspm',ascending=False).head(10)
Top10states_with_highest_rspm_sorted=Top10states_with_highest_rspm.loc[:,['rspm']]
Top10states_with_highest_rspm_sorted.head()
Top10states_with_highest_spm=statewise_emmissions.sort_values(by='spm',ascending=False).head(10)
Top10states_with_highest_spm_sorted=Top10states_with_highest_spm.loc[:,['spm']]
Top10states_with_highest_spm_sorted.head()
Top10states_with_highest_pm2_5=statewise_emmissions.sort_values(by='pm2_5',ascending=False).head(10)
Top10states_with_highest_pm2_5_sorted=Top10states_with_highest_pm2_5.loc[:,['pm2_5']]
Top10states_with_highest_pm2_5_sorted.head()
#Getting the statistics citywise for the pollutants
locationwise_emmissions=dataset.groupby('location').mean()[['so2','no2','rspm','spm','pm2_5']]
Top10Cities_with_highest_NO2=locationwise_emmissions.sort_values(by='no2',ascending=False).head(10)
Top10Cities_with_highest_NO2_sorted=Top10Cities_with_highest_NO2.loc[:,['no2']]
Top10Cities_with_highest_NO2_sorted.head()


Top10Cities_with_highest_So2=locationwise_emmissions.sort_values(by='so2',ascending=False).head(10)
Top10Cities_with_highest_So2_sorted=Top10Cities_with_highest_So2.loc[:,['so2']]
Top10Cities_with_highest_So2_sorted.head()


Top10Cities_with_highest_rspm=locationwise_emmissions.sort_values(by='rspm',ascending=False).head(10)
Top10Cities_with_highest_rspm_sorted=Top10Cities_with_highest_rspm.loc[:,['rspm']]
Top10Cities_with_highest_rspm_sorted.head()
Top10Cities_with_highest_spm=locationwise_emmissions.sort_values(by='spm',ascending=False).head(10)
Top10Cities_with_highest_spm_sorted=Top10Cities_with_highest_spm.loc[:,['spm']]
Top10Cities_with_highest_spm_sorted.head()
Top10Cities_with_highest_pm2_5=locationwise_emmissions.sort_values(by='pm2_5',ascending=False).head(10)
Top10Cities_with_highest_pm2_5_sorted=Top10Cities_with_highest_pm2_5.loc[:,['pm2_5']]
Top10Cities_with_highest_pm2_5_sorted.head()
#Visualising the emmissions according to the type and getting the relevant statistics
type_emmissions=dataset.groupby('type').mean()[['so2','no2','rspm','spm','pm2_5']]
type_emmissions.head()

fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(18,14))
ax = sns.barplot("so2", y="type",
                 data=dataset,
                 ax=axes[0,0]
                )
ax = sns.barplot("no2", y="type",
                 data=dataset,
                 ax=axes[0,1]
                )
ax = sns.barplot("rspm", y="type",
                 data=dataset,
                 ax=axes[1,0]
                )
ax = sns.barplot("spm", y="type",
                 data=dataset,
                 ax=axes[1,1]
                )
#Understanding the emmissions with time
dataset['date'].describe()

dataset.head()
common_value_date='2015-03-19'
dataset['date']=dataset['date'].fillna(common_value_date)
dataset.tail()
datewise_emmissions_SO2=dataset.groupby('date').mean()['so2']
datewise_emmissions_NO2=dataset.groupby('date').mean()['no2']
datewise_emmissions_rspm=dataset.groupby('date').mean()['rspm']
datewise_emmissions_spm=dataset.groupby('date').mean()['spm']



fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(14,10))
datewise_emmissions_SO2.plot(style='k.',legend=True,ax=axes[0,0])
datewise_emmissions_NO2.plot(style='b.',legend=True,ax=axes[0,1])
datewise_emmissions_rspm.plot(style='r.',legend=True,ax=axes[1,0])
datewise_emmissions_spm.plot(style='g.', legend=True,ax=axes[1,1])

dataset.groupby('so2').max()[['state','date']].tail(20)

dataset.groupby('no2').max()[['state','date']].tail(20)

dataset.groupby('spm').max()[['state','date']].tail(20)

dataset.groupby('rspm').max()[['state','date']].tail(20)
dataset['date'] = pd.to_datetime(dataset.date, format='%Y-%m-%d')
dataset.info()

#As it can be seen now, the date column is converted into datetime, instead of an object. This method is useful for 
#anlysing trends with time within the dataset.

#Making the date column as the index of the dataframe to make plotting and visulaisation easier.
dataset=dataset.set_index('date')
dataset.head()

yearly = dataset.resample('Y').mean()

monthly=dataset.resample('M').mean()

weekly=dataset.resample('W').mean()

daily=dataset.resample('D').mean()

#All the above dataframes will be grouped together and plotted together in a sinlge frame using subplots.

fig,axes=plt.subplots(nrows=2,ncols=2, figsize=(14,10))
yearly.plot(style=[':', '--', '-','.','*'],
            ax=axes[0,0],
            title='Yearly Emmissions')

monthly.plot(style=[':', '--', '-','.','*'],
             ax=axes[0,1],
             title='Monthly Emmissions')

weekly.plot(style=[':', '--', '-','.','*'],
            ax=axes[1,0],
            title='Weekly Emmissions')

daily.plot(style=[':', '--', '-','.','*'],
            ax=axes[1,1],
            title='Daily Emmissions')

#Putting together all the emissions data, datewise and visualising the data distributions, outliers and median values
fig,axes2=plt.subplots(nrows=2,ncols=2, figsize=(14,10))
yearly.plot.box(
                ax=axes2[0,0],
                title='Yearly Emmissions Distribution')

monthly.plot.box(
                ax=axes2[0,1],
                title='Monthly Emmissions Distribution')

weekly.plot.box(
                ax=axes2[1,0],
                title='Weekly Emmissions Distribution')

daily.plot.box(
                ax=axes2[1,1],
                title='Daily Emmissions Distribution')

Top5Years_highest_SO2=yearly.sort_values(by='so2', ascending=False).head(5)
Top5Years_highest_SO2.loc[:,'so2']



Top5Years_highest_NO2=yearly.sort_values(by='no2', ascending=False).head(5)
Top5Years_highest_NO2.loc[:,'no2']


Top10Years_highest_spm=yearly.sort_values(by='spm', ascending=False).head(10)
Top10Years_highest_spm.loc[:,'spm']

Top10Years_highest_rspm=yearly.sort_values(by='rspm', ascending=False).head(10)
Top10Years_highest_rspm.loc[:,'rspm']
#Getting the statistics out of each of the monthly merged datasets
Top10Months_highest_SO2=monthly.sort_values(by='so2', ascending=False).head(10)
Top10Months_highest_SO2.loc[:,'so2']





Top10Months_highest_NO2=monthly.sort_values(by='no2', ascending=False).head(10)
Top10Months_highest_NO2.loc[:,'no2']
Top20Months_highest_spm=monthly.sort_values(by='spm', ascending=False).head(20)
Top20Months_highest_spm.loc[:,'spm']
Top20Months_highest_rspm=monthly.sort_values(by='rspm', ascending=False).head(20)
Top20Months_highest_rspm.loc[:,'rspm']