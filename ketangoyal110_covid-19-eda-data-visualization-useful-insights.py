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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

from pandas import Series

%matplotlib inline 

from matplotlib.pylab import rcParams

rcParams['figure.figsize']=10,6

plt.style.use(['ggplot'])
df=pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",parse_dates=['ObservationDate'])

df.head()
#finding the missing data in DF

missing_data=df.isnull()

for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print(" ")
#filtering out top 5 cities of china 

china=df.loc[df['Country/Region']=='Mainland China',['Country/Region','Confirmed','ObservationDate','Deaths','Recovered']]

china=china.drop(['Country/Region'],axis=1)

china=china.groupby(['ObservationDate'],as_index=False).sum()

china.head()
#creating the bins of the data

bins_dt = pd.date_range('2020-01-21', freq='6D', periods=11)

labels=['Week-1','Week-2','Week-3','Week-4','Week-5','Week-6','Week-7','Week-8','Week-9','Week-10']

china['Confirmed-bins'] = pd.cut(china['ObservationDate'], bins=bins_dt, labels=labels)

china.head()
china=china.groupby(['Confirmed-bins']).sum()

china.plot(kind='bar')

plt.xlabel('Weekly developement in cases')

plt.ylabel('#No of Cases ')

plt.title('Weekly Developement of COVID_19 in China from January 22 2020')
#filtering out top 5 cities of Italy 

Italy=df.loc[df['Country/Region']=='Italy',['Country/Region','Confirmed','ObservationDate','Deaths','Recovered']]

Italy=Italy.drop(['Country/Region'],axis=1)

Italy=Italy.groupby(['ObservationDate'],as_index=False).sum()

#creating the bins of the data

bins_dt = pd.date_range('2020-01-21', freq='6D', periods=11)

labels=['Week-1','Week-2','Week-3','Week-4','Week-5','Week-6','Week-7','Week-8','Week-9','Week-10']

Italy['Confirmed-bins'] = pd.cut(Italy['ObservationDate'], bins=bins_dt, labels=labels)

#plotting the data of china

Italy=Italy.groupby(['Confirmed-bins']).sum()

Italy.plot(kind='bar')

plt.xlabel('Weekly developement in cases')

plt.ylabel('#No of Cases ')

plt.title('Weekly Developement of COVID_19 in Italy from January 22 2020')
#filtering out top 5 cities of UK 

UK=df.loc[df['Country/Region']=='UK',['Country/Region','Confirmed','ObservationDate','Deaths','Recovered']]

UK=UK.drop(['Country/Region'],axis=1)

UK=UK.groupby(['ObservationDate'],as_index=False).sum()

#creating the bins of the data

bins_dt = pd.date_range('2020-01-21', freq='6D', periods=11)

labels=['Week-1','Week-2','Week-3','Week-4','Week-5','Week-6','Week-7','Week-8','Week-9','Week-10']

UK['Confirmed-bins'] = pd.cut(UK['ObservationDate'], bins=bins_dt, labels=labels)

#plotting the data of china

UK=UK.groupby(['Confirmed-bins']).sum()

UK.plot(kind='bar')

plt.xlabel('Weekly developement in cases')

plt.ylabel('#No of Cases ')

plt.title('Weekly Developement of COVID_19 in UK from January 22 2020')
#filtering out top 5 cities of US

US=df.loc[df['Country/Region']=='UK',['Country/Region','Confirmed','ObservationDate','Deaths','Recovered']]

US=US.drop(['Country/Region'],axis=1)

US=US.groupby(['ObservationDate'],as_index=False).sum()

#creating the bins of the data

bins_dt = pd.date_range('2020-01-21', freq='6D', periods=11)

labels=['Week-1','Week-2','Week-3','Week-4','Week-5','Week-6','Week-7','Week-8','Week-9','Week-10']

US['Confirmed-bins'] = pd.cut(US['ObservationDate'], bins=bins_dt, labels=labels)

#plotting the data of china

US=US.groupby(['Confirmed-bins']).sum()

US.plot(kind='bar')

plt.xlabel('Weekly developement in cases')

plt.ylabel('#No of Cases ')

plt.title('Weekly Developement of COVID_19 in US from January 22 2020')
df_time_series=pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

df_cleaned_line=df_time_series.drop(['Lat','Long'],axis=1)

df_cleaned_line.head()
df_cleaned_line.rename(columns={'Country/Region':'Country','Province/State':'State'},inplace=True)

df_cleaned_line.head()


df_cleaned_line.set_index('State', inplace=True)

df_cleaned_line.index.name = None
df_cleaned_line['Total']=df_cleaned_line.sum(axis=1)

df_cleaned_line.head()
china1=df_cleaned_line.loc[df_cleaned_line['Country']=='China',['1/22/20', '1/23/20', '1/24/20', '1/25/20',

       '1/26/20', '1/27/20', '1/28/20', '1/29/20', '1/30/20', '1/31/20',

       '2/1/20', '2/2/20', '2/3/20', '2/4/20', '2/5/20', '2/6/20', '2/7/20',

       '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20', '2/13/20',

       '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20', '2/19/20',

       '2/20/20', '2/21/20', '2/22/20', '2/23/20', '2/24/20', '2/25/20',

       '2/26/20', '2/27/20', '2/28/20', '2/29/20', '3/1/20', '3/2/20',

       '3/3/20', '3/4/20', '3/5/20', '3/6/20', '3/7/20', '3/8/20', '3/9/20',

       '3/10/20', '3/11/20', '3/12/20', '3/13/20', '3/14/20', '3/15/20',

       '3/16/20', '3/17/20', '3/18/20', '3/19/20']]

china1.head()
china_time= china1.loc[['Hubei','Guangdong','Henan','Zhejiang','Hunan'], ['1/22/20', '1/23/20', '1/24/20', '1/25/20',

       '1/26/20', '1/27/20', '1/28/20', '1/29/20', '1/30/20', '1/31/20',

       '2/1/20', '2/2/20', '2/3/20', '2/4/20', '2/5/20', '2/6/20', '2/7/20',

       '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20', '2/13/20',

       '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20', '2/19/20',

       '2/20/20', '2/21/20', '2/22/20', '2/23/20', '2/24/20', '2/25/20',

       '2/26/20', '2/27/20', '2/28/20', '2/29/20', '3/1/20', '3/2/20',

       '3/3/20', '3/4/20', '3/5/20', '3/6/20', '3/7/20', '3/8/20', '3/9/20',

       '3/10/20', '3/11/20', '3/12/20', '3/13/20', '3/14/20', '3/15/20',

       '3/16/20', '3/17/20', '3/18/20', '3/19/20']]

china_time=pd.DataFrame(china_time)

china_time=china_time.transpose()

china_time.head()
china_time.plot(kind='line')

plt.title('Top 5 cities in China with Confirmed cases')

plt.xlabel('#Days')

plt.ylabel('No of Confirmed cases')
Us_time=df_cleaned_line.loc[df_cleaned_line['Country']=='US',['1/22/20', '1/23/20', '1/24/20', '1/25/20',

       '1/26/20', '1/27/20', '1/28/20', '1/29/20', '1/30/20', '1/31/20',

       '2/1/20', '2/2/20', '2/3/20', '2/4/20', '2/5/20', '2/6/20', '2/7/20',

       '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20', '2/13/20',

       '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20', '2/19/20',

       '2/20/20', '2/21/20', '2/22/20', '2/23/20', '2/24/20', '2/25/20',

       '2/26/20', '2/27/20', '2/28/20', '2/29/20', '3/1/20', '3/2/20',

       '3/3/20', '3/4/20', '3/5/20', '3/6/20', '3/7/20', '3/8/20', '3/9/20',

       '3/10/20', '3/11/20', '3/12/20', '3/13/20', '3/14/20', '3/15/20',

       '3/16/20', '3/17/20', '3/18/20', '3/19/20']]

Us_time.head()
#code for comparing first 100 cases in Italy

x=df.loc[df['Country/Region']=='Italy',['Country/Region','Confirmed','ObservationDate']]

x2=x.drop(['Country/Region'],axis=1)

Italy=x2.groupby(['ObservationDate'],as_index=False).sum()

Italy=Italy.drop(Italy.index[0:23])

Italy=Italy.reset_index()

Italy
x=df.loc[df['Country/Region']=='US',['Country/Region','Confirmed','ObservationDate']]

x2=x.drop(['Country/Region'],axis=1)

US=x2.groupby(['ObservationDate'],as_index=False).sum()

US=US.drop(US.index[0:40])

US=US.reset_index()

US
x=df.loc[df['Country/Region']=='Iran',['Country/Region','Confirmed','ObservationDate']]

x2=x.drop(['Country/Region'],axis=1)

Iran=x2.groupby(['ObservationDate'],as_index=False).sum()

Iran=Iran.drop(Iran.index[0:7])

Iran=Iran.reset_index()

Iran
x=df.loc[df['Country/Region']=='UK',['Country/Region','Confirmed','ObservationDate']]

x2=x.drop(['Country/Region'],axis=1)

UK=x2.groupby(['ObservationDate'],as_index=False).sum()

UK=UK.drop(UK.index[0:34])

UK=UK.reset_index()

UK
x=df.loc[df['Country/Region']=='India',['Country/Region','Confirmed','ObservationDate']]

x2=x.drop(['Country/Region'],axis=1)

India=x2.groupby(['ObservationDate'],as_index=False).sum()

India=India.drop(India.index[0:44])

India=India.reset_index()

India
#comparing first 100 cases of Italy and US

sns.barplot(x=US.index,y='Confirmed',data=US,color='lightgreen')

sns.barplot(x=Italy.index,y='Confirmed',data=Italy,color='coral')

plt.xticks(rotation=90)

plt.xlabel('Days')

plt.ylabel('#No of confirmed cases')

plt.title('Comparision of cases after first 100 cases in Italy and US')
sns.barplot(x=Iran.index,y='Confirmed',data=Iran,color='lightgreen')

sns.barplot(x=Italy.index,y='Confirmed',data=Italy,color='coral')

plt.xticks(rotation=90)

plt.xlabel('Days')

plt.ylabel('#No of confirmed cases')

plt.title('Comparision of cases after first 100 cases in Italy and Iran')
sns.barplot(x=UK.index,y='Confirmed',data=UK,color='lightgreen')

sns.barplot(x=Italy.index,y='Confirmed',data=Italy,color='coral')

plt.xticks(rotation=90)

plt.xlabel('Days')

plt.ylabel('#No of confirmed cases')

plt.title('Comparision of cases after first 100 cases in Italy and UK')
#comparing first 100 cases of Italy and US

sns.barplot(x=India.index,y='Confirmed',data=India,color='lightgreen')

sns.barplot(x=Italy.index,y='Confirmed',data=Italy,color='coral')

plt.xticks(rotation=90)

plt.xlabel('Days')

plt.ylabel('#No of confirmed cases')

plt.title('Comparision of cases after first 100 cases in Italy and India')
conf_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

deaths_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

recv_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
conf_df.columns[4:]
dates = conf_df.columns[4:]



conf_df_long = conf_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Confirmed')



deaths_df_long = deaths_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Deaths')



recv_df_long = recv_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Recovered')



full_table = pd.concat([conf_df_long, deaths_df_long['Deaths'], recv_df_long['Recovered']], 

                       axis=1, sort=False)



full_table.head()

#refrence:https://www.kaggle.com/imdevskp/covid-19-analysis-visualization-comparisons
full_latest = full_table[full_table['Date'] == max(full_table['Date'])].reset_index()

full_latest_grouped = full_latest.groupby(['Country/Region'])['Confirmed', 'Deaths', 'Recovered'].sum()

full_latest_grouped.head()
final_grouped = full_latest_grouped.sort_values(by='Confirmed', ascending=False).head(15)

final_grouped.head()
final_grouped['Confirmed'].plot('barh')

plt.xlabel('#no of Confirmed Cases')

plt.ylabel('Countries')

plt.title('No of Confirmed cases in top 15 countries affected by COVID_19')
final_grouped['Deaths'].plot('barh')

plt.xlabel('#no of deaths Cases')

plt.ylabel('Countries')

plt.title('No of deaths in top 15 countries affected by COVID_19')
final_grouped['Recovered'].plot('barh')

plt.xlabel('#no of Recovered Cases')

plt.ylabel('Countries')

plt.title('No of recovered in top 15 countries affected by COVID_19')


x=df.loc[df['Country/Region']=='Mainland China',['Country/Region','Confirmed','ObservationDate','Deaths','Recovered']]

x2=x.drop(['Country/Region'],axis=1)

x2=x2.groupby(['ObservationDate'],as_index=False).sum()

x2.head()
x2['ObservationDate'] = pd.to_datetime(x2.ObservationDate)



cumval=0

fig = plt.figure(figsize=(12,8))

for col in x2.columns[~x2.columns.isin(['ObservationDate'])]:

    plt.bar(x2.ObservationDate, x2[col], bottom=cumval, label=col)

    cumval = cumval+x2[col]



_ = plt.xticks(rotation=30)

_ = plt.legend(fontsize=18) 

_ = plt.title('Stacked bar chart showing Confirmed,Deaths,Recovered of china') 



x=df.loc[df['Country/Region']=='Italy',['Country/Region','Confirmed','ObservationDate','Deaths','Recovered']]

x2=x.drop(['Country/Region'],axis=1)

x2=x2.groupby(['ObservationDate'],as_index=False).sum()

x2.head()
x2['ObservationDate'] = pd.to_datetime(x2.ObservationDate)



cumval=0

fig = plt.figure(figsize=(12,8))

for col in x2.columns[~x2.columns.isin(['ObservationDate'])]:

    plt.bar(x2.ObservationDate, x2[col], bottom=cumval, label=col)

    cumval = cumval+x2[col]



_ = plt.xticks(rotation=30)

_ = plt.legend(fontsize=18)

_ = plt.title('Stacked bar chart showing Confirmed,Deaths,Recovered of Italy')
#Stacked bar chart for Iran

x=df.loc[df['Country/Region']=='Iran',['Country/Region','Confirmed','ObservationDate','Deaths','Recovered']]

x2=x.drop(['Country/Region'],axis=1)

x2=x2.groupby(['ObservationDate'],as_index=False).sum()

x2.head()
x2['ObservationDate'] = pd.to_datetime(x2.ObservationDate)



cumval=0

fig = plt.figure(figsize=(12,8))

for col in x2.columns[~x2.columns.isin(['ObservationDate'])]:

    plt.bar(x2.ObservationDate, x2[col], bottom=cumval, label=col)

    cumval = cumval+x2[col]



_ = plt.xticks(rotation=90)

_ = plt.legend(fontsize=18) 

_ = plt.title('Stacked bar chart showing Confirmed,Deaths,Recovered of Iran')
#Stacked bar chart for US

x=df.loc[df['Country/Region']=='US',['Country/Region','Confirmed','ObservationDate','Deaths','Recovered']]

x2=x.drop(['Country/Region'],axis=1)

x2=x2.groupby(['ObservationDate'],as_index=False).sum()

x2.head()
x2['ObservationDate'] = pd.to_datetime(x2.ObservationDate)



cumval=0

fig = plt.figure(figsize=(12,8))

for col in x2.columns[~x2.columns.isin(['ObservationDate'])]:

    plt.bar(x2.ObservationDate, x2[col], bottom=cumval, label=col)

    cumval = cumval+x2[col]



_ = plt.xticks(rotation=90)

_ = plt.legend(fontsize=18)

_ = plt.title('Stacked bar chart showing Confirmed,Deaths,Recovered of US')
#Stacked bar chart for India

x=df.loc[df['Country/Region']=='India',['Country/Region','Confirmed','ObservationDate','Deaths','Recovered']]

x2=x.drop(['Country/Region'],axis=1)

x2=x2.groupby(['ObservationDate'],as_index=False).sum()

x2.head()
x2['ObservationDate'] = pd.to_datetime(x2.ObservationDate)



cumval=0

fig = plt.figure(figsize=(12,8))

for col in x2.columns[~x2.columns.isin(['ObservationDate'])]:

    plt.bar(x2.ObservationDate, x2[col], bottom=cumval, label=col)

    cumval = cumval+x2[col]



_ = plt.xticks(rotation=90)

_ = plt.legend(fontsize=18)



_ = plt.title('Stacked bar chart showing Confirmed,Deaths,Recovered of India')