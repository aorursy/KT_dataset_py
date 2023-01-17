import pandas as pd

import numpy as np
df_hs_exp_2010 = pd.read_csv('../input/all_house_senate_2010.csv', na_values='0', index_col='tra_id', sep=',', low_memory=False)



# Filter rows having null or NaN in the date and amount field

df_hs_exp_2010 = df_hs_exp_2010.dropna(subset = ['dis_dat', 'dis_amo'])



# Filter records relevant to year 2010

df_hs_exp_2010 = df_hs_exp_2010[df_hs_exp_2010['dis_dat'].apply(lambda x: x[0:4]) == '2010']



# Format the datatime column for time series analysis

df_hs_exp_2010['dis_dat'] = pd.to_datetime(df_hs_exp_2010['dis_dat'], format='%Y-%m-%d')



# Format the currency field to convert them to float

df_hs_exp_2010['dis_amo'] = (df_hs_exp_2010['dis_amo'].replace( '[\$,)]','', regex=True ).replace( '[(]','-',   regex=True ).astype(float))
#Applying same modifications to 2012, 2014 and 2016 datasets

df_hs_exp_2012 = pd.read_csv('../input/2012.csv', na_values='0', index_col='tra_id', sep=',', parse_dates=['dis_dat'],low_memory=False)

df_hs_exp_2012 = df_hs_exp_2012.dropna(subset = ['dis_dat', 'dis_amo'])

df_hs_exp_2012 = df_hs_exp_2012[df_hs_exp_2012['dis_dat'].apply(lambda x: x[0:4]) == '2012'] #relevant 2012 records

df_hs_exp_2012['dis_dat'] = pd.to_datetime(df_hs_exp_2012['dis_dat'], format='%Y-%m-%d')

df_hs_exp_2012['dis_amo'] = (df_hs_exp_2012['dis_amo'].replace( '[\$,)]','', regex=True ).replace( '[(]','-',   regex=True ).astype(float))

df_hs_exp_2014 = pd.read_csv('../input/all_house_senate_2014.csv', na_values='0', index_col='tra_id', sep=',',low_memory=False)

df_hs_exp_2014 = df_hs_exp_2014.dropna(subset = ['dis_dat', 'dis_amo'])

df_hs_exp_2014 = df_hs_exp_2014[df_hs_exp_2014['dis_dat'].apply(lambda x: x[0:4]) == '2014'] #relevant 2014 records

df_hs_exp_2014['dis_dat'] = pd.to_datetime(df_hs_exp_2014['dis_dat'], format='%Y-%m-%d')

df_hs_exp_2014['dis_amo'] = (df_hs_exp_2014['dis_amo'].replace( '[\$,)]','', regex=True ).replace( '[(]','-',   regex=True ).astype(float))
df_hs_exp_2016 = pd.read_csv('../input/all_house_senate_2016.csv', na_values='0', index_col='tra_id', sep=',',low_memory=False, encoding = "ISO-8859-1")

df_hs_exp_2016 = df_hs_exp_2016.dropna(subset = ['dis_dat', 'dis_amo'])

df_hs_exp_2016 = df_hs_exp_2016[df_hs_exp_2016['dis_dat'].apply(lambda x: x[0:4]) == '2016'] #relevant 2016 records

df_hs_exp_2016['dis_dat'] = pd.to_datetime(df_hs_exp_2016['dis_dat'], format='%Y-%m-%d')

df_hs_exp_2016['dis_amo'] = (df_hs_exp_2016['dis_amo'].replace( '[\$,)]','', regex=True ).replace( '[(]','-',   regex=True ).astype(float))
df_hs_exp_2010.isnull().sum()
df_hs_exp_2012.isnull().sum()
df_hs_exp_2014.isnull().sum()
df_hs_exp_2016.isnull().sum()
#Summing amounts (in million dollars) by date to create a timeseries trendline for 2010

df_2010 = pd.DataFrame(df_hs_exp_2010.groupby('dis_dat')['dis_amo'].sum()).reset_index()

df_series_2010 = pd.Series(np.array(df_2010['dis_amo']/1000000), np.array(df_2010['dis_dat']))



#Summing amounts (in million dollars) by date to create a timeseries trendline for 2012

df_2012 = pd.DataFrame(df_hs_exp_2012.groupby('dis_dat')['dis_amo'].sum()).reset_index()

df_series_2012 = pd.Series(np.array(df_2012['dis_amo']/1000000), np.array(df_2012['dis_dat']))



#Summing amounts (in million dollars) by date to create a timeseries trendline for 2014

df_2014 = pd.DataFrame(df_hs_exp_2014.groupby('dis_dat')['dis_amo'].sum()).reset_index()

df_series_2014 = pd.Series(np.array(df_2014['dis_amo']/1000000), np.array(df_2014['dis_dat']))



#Summing amounts (in million dollars) by date to create a timeseries trendline for 2016

df_2016 = pd.DataFrame(df_hs_exp_2016.groupby('dis_dat')['dis_amo'].sum()).reset_index()

df_series_2016 = pd.Series(np.array(df_2016['dis_amo']/1000000), np.array(df_2016['dis_dat']))
import datetime

import matplotlib.pyplot as plt

import numpy as np



import plotly.plotly as py

import plotly.tools as tls
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 4, ncols = 1, figsize=(25, 40))



#plt.rcParams["figure.figsize"] = [15, 30]

ax1.set_ylim(0,60)

ax1.set_title(' Trendline for Disbursements in 2010')

ax1.set_ylabel('Amount in Million Dollars ($)')



ax2.set_ylim(0,60)

ax2.set_title('Trendline for Disbursements in 2012')

ax2.set_ylabel('Amount in Million Dollars ($)')



ax3.set_ylim(0,60)

ax3.set_title(' Trendline for Disbursements in 2014')

ax3.set_ylabel('Amount in Million Dollars ($)')



ax4.set_ylim(0,60)

ax4.set_title('Trendline for Disbursements in 2016')

ax4.set_ylabel('Amount in Million Dollars ($)')



df_series_2010.plot(ax=ax1)

df_series_2012.plot(ax=ax2)

df_series_2014.plot(ax=ax3)

df_series_2016.plot(ax=ax4)



plt.show()
def getWeekDay(dt):

    year, month, day = (int(x) for x in dt.split('-'))    

    weekday = datetime.date(year, month, day).weekday()

    return weekday
df_2010_week = df_series_2010.to_frame().reset_index()

df_2010_week.rename(columns={'index':'Date', 0:'Disbursement'}, inplace=True)

df_2010_week['DayOfWeek'] = df_2010_week['Date'].dt.dayofweek



df_2012_week = df_series_2012.to_frame().reset_index()

df_2012_week.rename(columns={'index':'Date', 0:'Disbursement'}, inplace=True)

df_2012_week['DayOfWeek'] = df_2012_week['Date'].dt.dayofweek



df_2014_week = df_series_2014.to_frame().reset_index()

df_2014_week.rename(columns={'index':'Date', 0:'Disbursement'}, inplace=True)

df_2014_week['DayOfWeek'] = df_2014_week['Date'].dt.dayofweek



df_2016_week = df_series_2016.to_frame().reset_index()

df_2016_week.rename(columns={'index':'Date', 0:'Disbursement'}, inplace=True)

df_2016_week['DayOfWeek'] = df_2016_week['Date'].dt.dayofweek
fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 2, figsize=(15, 10))

weekday_map= {0:'MON', 1:'TUE', 2:'WED', 3:'THU',

              4:'FRI', 5:'SAT', 6:'SUN'}

import seaborn as sns



ax1[0].set_ylim(0,50)

ax1[0].set_title('Disbursements by weekday in 2010')

ax1[0].set_ylabel('Amount in Million Dollars ($)')

ax1[0].set_xticklabels(['M'])



ax1[1].set_ylim(0,50)

ax1[1].set_title('Disbursements by weekday in 2012')

ax1[1].set_ylabel('Amount in Million Dollars ($)')



ax2[0].set_ylim(0,50)

ax2[0].set_title('Disbursements by weekday in 2014')

ax2[0].set_ylabel('Amount in Million Dollars ($)')



ax2[1].set_ylim(0,50)

ax2[1].set_title('Disbursements by weekday in 2016')

ax2[1].set_ylabel('Amount in Million Dollars ($)')





sns.boxplot(x="DayOfWeek", y="Disbursement", data=df_2010_week,palette='rainbow', ax=ax1[0])

sns.boxplot(x="DayOfWeek", y="Disbursement", data=df_2012_week,palette='rainbow', ax=ax1[1])

sns.boxplot(x="DayOfWeek", y="Disbursement", data=df_2014_week,palette='rainbow', ax=ax2[0])

sns.boxplot(x="DayOfWeek", y="Disbursement", data=df_2016_week,palette='rainbow', ax=ax2[1])