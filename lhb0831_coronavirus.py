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
import matplotlib.pyplot as plt

import seaborn as sns



'''Set the color palette'''

sns.set_style(style='darkgrid')

sns.set_context(context='poster',font_scale=0.5)

sns.set_palette(sns.color_palette("muted"))



df_all=pd.read_csv('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')

'''Data Cleaning'''

df_all=df_all.fillna(0)

df_all=df_all.drop(columns=['Sno','Last Update'])

df_all['Date']=df_all['Date'].str.split(' ').str[0]

df_all['Country']=df_all['Country'].where(df_all['Country']!='China')

df_all['Country']=df_all['Country'].fillna('Mainland China')

df_all
#lineplot

df=df_all.copy()

df=df[df['Province/State']!=0][df['Country'].isin(['Mainland China','China','Hong Kong','Taiwan','Macau'])]

df=df.sort_values(by=['Date'])

df_group_sum=pd.DataFrame(df.groupby('Date').sum())

plt.figure(figsize=(25,10))

sns.lineplot(data=df_group_sum['Confirmed'])

plt.xticks(rotation=30)

plt.xlabel('Time')

plt.ylabel('Number')

plt.title('Accumulation of Virus Confirmed Cases Over Time(In China)')

x_date=df_group_sum.index

for x,y in zip(x_date,df_group_sum['Confirmed']):

    plt.text(x,y,y,ha='left',va='top')



plt.figure(figsize=(25,10))

sns.lineplot(data=df_group_sum[['Deaths','Recovered']])

plt.xticks(rotation=30)

plt.xlabel('Time')

plt.ylabel('Number')

plt.title('Accumulation of Virus Deathes& Recovered Cases Over Time(In China)')

x_date=df_group_sum.index

y_num=['Recovered','Deaths']

temp = 0

for i in y_num:#Disply the text on the line

    for x,y in zip(x_date,df_group_sum[i]):

        h_pos=['left','right']

        v_pos=['bottom','top']

        plt.text(x,y,y,ha=h_pos[temp],va=v_pos[temp])

    temp+=1

df_group_sum
df_group_sum['Motality']=df_group_sum['Deaths']/df_group_sum['Confirmed']

df_group_sum['Recovered Rate']=df_group_sum['Recovered']/df_group_sum['Confirmed']

df_MS=df_group_sum[['Motality','Recovered Rate']]

plt.figure(figsize=(20,5))

sns.lineplot(data=df_MS)

plt.xticks(rotation=30)

plt.xlabel('Time')

plt.ylabel('Rate')

plt.title('Motality & Recovered Rate Over Time(In China)')

df_group_sum
df_group_sum=pd.DataFrame(df.groupby('Date').sum())

rate=['New Confirmed','New Deaths','New Recovered']

col=['Confirmed','Deaths','Recovered']

df_group_sum[rate]=df_group_sum[['Confirmed','Deaths','Recovered']]

df_group_sum=df_group_sum.reset_index()

for i in range(len(rate)):

    for j in range(len(df_group_sum['Date'])):

        if j==0:

            df_group_sum.at[j,rate[i]]=0

        else:

            df_group_sum.at[j,rate[i]]=df_group_sum.at[j,col[i]]-df_group_sum.at[j-1,col[i]]

df_group_sum=df_group_sum[['Date','New Confirmed','New Deaths','New Recovered']].set_index('Date')



plt.figure(figsize=(20,10))

sns.lineplot(data=df_group_sum['New Confirmed'])

plt.xlabel('Time')

plt.ylabel('New Number')

plt.title('The new confirmed cases over the time(In China)')

x_date=df_group_sum.index

for x,y in zip(x_date,df_group_sum['New Confirmed']):

    plt.text(x,y,y,ha='left',va='top')



plt.figure(figsize=(20,10))

sns.lineplot(data=df_group_sum[['New Deaths','New Recovered']])

plt.xlabel('Time')

plt.ylabel('New Number')

plt.title('The new recovered and death cases over the time(In China)')

x_date=df_group_sum.index

y_num=['New Deaths','New Recovered']

temp=0

for i in y_num:

    for x,y in zip(x_date,df_group_sum[i]):

        h_pos=['right','left']

        v_pos=['top','bottom']

        plt.text(x,y,y,ha=h_pos[temp],va=v_pos[temp])

    temp+=1

df_group_sum
'''Country explorations'''

df=df_all.copy()

unique_date=list(df['Date'].unique())

type_list=['Confirmed','Recovered','Deaths']

df=df[df['Country']!='Mainland China'][df['Date']==unique_date[-1]].groupby('Country').sum().sort_values(by='Confirmed',ascending=False).reset_index()

for i in type_list:

    plt.figure(figsize=(20,10))

    sns.barplot(x=i,y='Country',data=df)

    plt.xlabel(i)

    plt.ylabel('Country')

    plt.title(i+' Number(Over Country)')

df
'''Province/State explorations'''

df=df_all.copy()

unique_date=list(df['Date'].unique())

type_list=['Confirmed','Recovered','Deaths']

df=df[df['Province/State']!=0][df['Country'].isin(['Mainland China','China','Hong Kong','Taiwan','Macau'])]

df=df[df['Date']==unique_date[-1]].groupby('Province/State').sum().sort_values(by='Confirmed',ascending=False).reset_index()

for i in type_list:

    plt.figure(figsize=(20,10))

    sns.barplot(x=i,y='Province/State',data=df)

    plt.xlabel(i)

    plt.ylabel('Province/State')

    plt.title(i+' Number(In China)')