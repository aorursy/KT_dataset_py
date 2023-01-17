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
data=pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

data.tail()
subset_booked=data[data['is_canceled']==0]

subset_canceled=data[data['is_canceled']==1]

print('Number of bookings made =',subset_booked['hotel'].count())

print('Number of cancellations made =',subset_canceled['hotel'].count())
import matplotlib.pyplot as plt

import seaborn as sns

sns.boxplot(x=data['is_canceled'],y=data['lead_time'])

total=data[(data['lead_time']>214)&(data['lead_time']<365)]['hotel'].count()

cases=subset_canceled[(subset_canceled['lead_time']>214)&(subset_canceled['lead_time']<365)]['hotel'].count()

print('Probability of booking getting canceled given that lead time is between 214 days and 300 days= ',cases/total)

print('Probability of booking getting canceled= ',cases/data['hotel'].count())
x=[]

y=[]

for i in range(400):

    total=data[(data['lead_time']>214)&(data['lead_time']<i)]['hotel'].count()

    cases=subset_canceled[(subset_canceled['lead_time']>214)&(subset_canceled['lead_time']<i)]['hotel'].count()

    x.append(i)

    y.append((cases/total))

plt.scatter(x,y)

plt.title('Probability of booking cancelation vs max. lead time')

#plt.xlabel('Number of days')

#plt.ylabel('Probability of booking getting canceled')

plt.show()
subset_booked.groupby('arrival_date_year')['hotel'].count()

subset_booked_2015=subset_booked[subset_booked['arrival_date_year']==2015]

subset_booked_2016=subset_booked[subset_booked['arrival_date_year']==2016]

subset_booked_2017=subset_booked[subset_booked['arrival_date_year']==2017]

months=['January','February','March','April','May','June','July','August','September','October','November','December']

booking_2015=[]

booking_2016=[]

booking_2017=[]

for i in months:

    num_15=subset_booked_2015[subset_booked_2015['arrival_date_month']==i]['hotel'].count()

    num_16=subset_booked_2016[subset_booked_2016['arrival_date_month']==i]['hotel'].count()

    num_17=subset_booked_2017[subset_booked_2017['arrival_date_month']==i]['hotel'].count()

    booking_2015.append(num_15)

    booking_2016.append(num_16)

    booking_2017.append(num_17)

plt.figure(figsize=(20,20))

plt.plot(months,booking_2015,label='2015')

plt.plot(months,booking_2016,label='2016')

plt.plot(months,booking_2017,label='2017')

plt.legend()

plt.show()

print('Mean bookings in 2016= ',round(np.array(booking_2016).mean()),0)

print('Median bookings in 2016= ',np.median(np.array(booking_2016)))



data.groupby('is_repeated_guest')['hotel'].count()
country=subset_booked[['hotel','country']]

c=country.groupby('country')['hotel'].count()

df=pd.DataFrame(c)

x=list(c.index)

y = list(c)

#plt.figure(figsize=(20,20))

#plt.barh(x[:50],y[:50])

top=np.quantile(c,0.95)

df[df['hotel']>top]

total=df['hotel'].sum()

df['% of stays']=(df['hotel']/total)*100

top_coutries=list(df[df['hotel']>top].index)

df[df['hotel']>top]



#prt=subset_booked[subset_booked['country']=='PRT']

#prt=prt[['hotel','arrival_date_month','arrival_date_year']]

#prt=prt[prt['arrival_date_year']==2017]

#prt.groupby('arrival_date_month')['hotel'].count()
subset_PRT=subset_booked[subset_booked['country']=='PRT']

subset_NLD=subset_booked[subset_booked['country']=='NLD']

subset_ITA=subset_booked[subset_booked['country']=='ITA']

subset_IRL=subset_booked[subset_booked['country']=='IRL']

subset_GBR=subset_booked[subset_booked['country']=='GBR']

subset_FRA=subset_booked[subset_booked['country']=='FRA']

subset_ESP=subset_booked[subset_booked['country']=='ESP']

subset_DEU=subset_booked[subset_booked['country']=='DEU']

subset_BEL=subset_booked[subset_booked['country']=='BEL']
City_Hotel=[]

City_Hotel.append(subset_PRT[subset_PRT['hotel']=='City Hotel']['hotel'].count())

City_Hotel.append(subset_NLD[subset_NLD['hotel']=='City Hotel']['hotel'].count())

City_Hotel.append(subset_ITA[subset_ITA['hotel']=='City Hotel']['hotel'].count())

City_Hotel.append(subset_IRL[subset_IRL['hotel']=='City Hotel']['hotel'].count())

City_Hotel.append(subset_GBR[subset_GBR['hotel']=='City Hotel']['hotel'].count())

City_Hotel.append(subset_FRA[subset_FRA['hotel']=='City Hotel']['hotel'].count())

City_Hotel.append(subset_ESP[subset_ESP['hotel']=='City Hotel']['hotel'].count())

City_Hotel.append(subset_DEU[subset_DEU['hotel']=='City Hotel']['hotel'].count())

City_Hotel.append(subset_BEL[subset_BEL['hotel']=='City Hotel']['hotel'].count())



Resort_Hotel=[]

Resort_Hotel.append(subset_PRT[subset_PRT['hotel']=='Resort Hotel']['hotel'].count())

Resort_Hotel.append(subset_NLD[subset_NLD['hotel']=='Resort Hotel']['hotel'].count())

Resort_Hotel.append(subset_ITA[subset_ITA['hotel']=='Resort Hotel']['hotel'].count())

Resort_Hotel.append(subset_IRL[subset_IRL['hotel']=='Resort Hotel']['hotel'].count())

Resort_Hotel.append(subset_GBR[subset_GBR['hotel']=='Resort Hotel']['hotel'].count())

Resort_Hotel.append(subset_FRA[subset_FRA['hotel']=='Resort Hotel']['hotel'].count())

Resort_Hotel.append(subset_ESP[subset_ESP['hotel']=='Resort Hotel']['hotel'].count())

Resort_Hotel.append(subset_DEU[subset_DEU['hotel']=='Resort Hotel']['hotel'].count())

Resort_Hotel.append(subset_BEL[subset_BEL['hotel']=='Resort Hotel']['hotel'].count())



top_countries=['PRT','NLD','ITA','IRL','GBR','FRA','ESP','DEU','BEL']

top_data=subset_booked[subset_booked['country'].isin(top_countries)]

hotel=pd.DataFrame({'Country':top_countries,'City hotel preffered':City_Hotel,'Resort hotel preffered': Resort_Hotel})

hotel

#plt.bar(top_countries,City_Hotel,label='City Hotel')

#plt.title('City Hotel')

#plt.show()

#plt.bar(top_countries,Resort_Hotel,label='Resort Hotel')

#plt.title('Resort Hotel')

#plt.show()
subset_PRT=data[data['country']=='PRT']

subset_NLD=data[data['country']=='NLD']

subset_ITA=data[data['country']=='ITA']

subset_IRL=data[data['country']=='IRL']

subset_GBR=data[data['country']=='GBR']

subset_FRA=data[data['country']=='FRA']

subset_ESP=data[data['country']=='ESP']

subset_DEU=data[data['country']=='DEU']

subset_BEL=data[data['country']=='BEL']



City_Hotel=[]

City_Hotel.append(subset_PRT[subset_PRT['hotel']=='City Hotel']['hotel'].count())

City_Hotel.append(subset_NLD[subset_NLD['hotel']=='City Hotel']['hotel'].count())

City_Hotel.append(subset_ITA[subset_ITA['hotel']=='City Hotel']['hotel'].count())

City_Hotel.append(subset_IRL[subset_IRL['hotel']=='City Hotel']['hotel'].count())

City_Hotel.append(subset_GBR[subset_GBR['hotel']=='City Hotel']['hotel'].count())

City_Hotel.append(subset_FRA[subset_FRA['hotel']=='City Hotel']['hotel'].count())

City_Hotel.append(subset_ESP[subset_ESP['hotel']=='City Hotel']['hotel'].count())

City_Hotel.append(subset_DEU[subset_DEU['hotel']=='City Hotel']['hotel'].count())

City_Hotel.append(subset_BEL[subset_BEL['hotel']=='City Hotel']['hotel'].count())



Resort_Hotel=[]

Resort_Hotel.append(subset_PRT[subset_PRT['hotel']=='Resort Hotel']['hotel'].count())

Resort_Hotel.append(subset_NLD[subset_NLD['hotel']=='Resort Hotel']['hotel'].count())

Resort_Hotel.append(subset_ITA[subset_ITA['hotel']=='Resort Hotel']['hotel'].count())

Resort_Hotel.append(subset_IRL[subset_IRL['hotel']=='Resort Hotel']['hotel'].count())

Resort_Hotel.append(subset_GBR[subset_GBR['hotel']=='Resort Hotel']['hotel'].count())

Resort_Hotel.append(subset_FRA[subset_FRA['hotel']=='Resort Hotel']['hotel'].count())

Resort_Hotel.append(subset_ESP[subset_ESP['hotel']=='Resort Hotel']['hotel'].count())

Resort_Hotel.append(subset_DEU[subset_DEU['hotel']=='Resort Hotel']['hotel'].count())

Resort_Hotel.append(subset_BEL[subset_BEL['hotel']=='Resort Hotel']['hotel'].count())



top_countries=['PRT','NLD','ITA','IRL','GBR','FRA','ESP','DEU','BEL']

hotel=pd.DataFrame({'Country':top_countries,'City hotel bookings':City_Hotel,'Resort hotel bookings': Resort_Hotel})

hotel

plt.hist(subset_PRT['adults'],bins=3,range=[1,3],label='Portugal')

plt.hist(subset_NLD['adults'],bins=3,range=[1,3],label='Netherlands',alpha=0.3)

plt.hist(subset_ITA['adults'],bins=3,range=[1,3],label='Italy',alpha=0.3)

plt.hist(subset_IRL['adults'],bins=3,range=[1,3],label='Ireland',alpha=0.3)

plt.hist(subset_GBR['adults'],bins=3,range=[1,3],label='Britain',alpha=0.3)

plt.hist(subset_FRA['adults'],bins=3,range=[1,3],label='France',alpha=0.3)

plt.hist(subset_ESP['adults'],bins=3,range=[1,3],label='Spain',alpha=0.3)

plt.hist(subset_DEU['adults'],bins=3,range=[1,3],label='Germany',alpha=0.3)

plt.hist(subset_BEL['adults'],bins=3,range=[1,3],label='Belgium',alpha=0.3)

plt.legend()

plt.title('Distribution of adult customers staying in hotel ')

plt.xlabel='Number of adults booked'

plt.xlabel='Count'

plt.show()



avg=subset_booked['children'].mean()

med=subset_booked['children'].median()

print('Mean of children staying=',round(avg,0))

print('Median of children staying=',med)

company_bookings=[]

company_bookings.append(subset_PRT[subset_PRT['company'] != np.nan]['hotel'].count())

company_bookings.append(subset_NLD[subset_NLD['company'] != np.nan]['hotel'].count())

company_bookings.append(subset_ITA[subset_ITA['company'] != np.nan]['hotel'].count())

company_bookings.append(subset_IRL[subset_IRL['company'] != np.nan]['hotel'].count())

company_bookings.append(subset_GBR[subset_GBR['company'] != np.nan]['hotel'].count())

company_bookings.append(subset_FRA[subset_FRA['company'] != np.nan]['hotel'].count())

company_bookings.append(subset_ESP[subset_ESP['company'] != np.nan]['hotel'].count())

company_bookings.append(subset_DEU[subset_DEU['company'] != np.nan]['hotel'].count())

company_bookings.append(subset_BEL[subset_BEL['company'] != np.nan]['hotel'].count())



total_bookings=[]

total_bookings.append(subset_PRT['hotel'].count())

total_bookings.append(subset_NLD['hotel'].count())

total_bookings.append(subset_ITA['hotel'].count())

total_bookings.append(subset_IRL['hotel'].count())

total_bookings.append(subset_GBR['hotel'].count())

total_bookings.append(subset_FRA['hotel'].count())

total_bookings.append(subset_ESP['hotel'].count())

total_bookings.append(subset_DEU['hotel'].count())

total_bookings.append(subset_BEL['hotel'].count())



bookings=pd.DataFrame({'Country':top_countries,'Company bookings':company_bookings,'Total bookings':total_bookings})

bookings
chrt=sns.boxplot(x=top_data['adr'],y=top_data['country'])

chrt.axvline(top_data['adr'].median())

np.quantile(top_data[top_data['country']=='ITA']['adr'],0.5)
sns.boxplot(x=top_data['total_of_special_requests'],y=top_data['country'])

subset_booked.groupby('market_segment')['hotel'].count()

segment_data=pd.DataFrame(top_data.groupby(['market_segment'])['hotel'].count())

total = segment_data['hotel'].sum()

segment_data['% share']= round(((segment_data['hotel']/total)*100),0)

segment_data.sort_values(by='hotel',ascending=False)







segment_data_PRT=pd.DataFrame(subset_PRT.groupby(['market_segment'])['hotel'].count())

total = segment_data_PRT['hotel'].sum()

segment_data_PRT['% share']= round(((segment_data_PRT['hotel']/total)*100),0)

segment_data_PRT=segment_data_PRT.sort_values(by='hotel',ascending=False)



segment_data_NLD=pd.DataFrame(subset_NLD.groupby(['market_segment'])['hotel'].count())

total = segment_data_NLD['hotel'].sum()

segment_data_NLD['% share']= round(((segment_data_NLD['hotel']/total)*100),0)

segment_data_NLD=segment_data_NLD.sort_values(by='hotel',ascending=False)



segment_data_ITA=pd.DataFrame(subset_ITA.groupby(['market_segment'])['hotel'].count())

total = segment_data_ITA['hotel'].sum()

segment_data_ITA['% share']= round(((segment_data_ITA['hotel']/total)*100),0)

segment_data_ITA=segment_data_ITA.sort_values(by='hotel',ascending=False)



segment_data_IRL=pd.DataFrame(subset_IRL.groupby(['market_segment'])['hotel'].count())

total = segment_data_IRL['hotel'].sum()

segment_data_IRL['% share']= round(((segment_data_IRL['hotel']/total)*100),0)

segment_data_IRL=segment_data_IRL.sort_values(by='hotel',ascending=False)



segment_data_GBR=pd.DataFrame(subset_GBR.groupby(['market_segment'])['hotel'].count())

total = segment_data_GBR['hotel'].sum()

segment_data_GBR['% share']= round(((segment_data_GBR['hotel']/total)*100),0)

segment_data_GBR=segment_data_GBR.sort_values(by='hotel',ascending=False)



segment_data_FRA=pd.DataFrame(subset_FRA.groupby(['market_segment'])['hotel'].count())

total = segment_data_FRA['hotel'].sum()

segment_data_FRA['% share']= round(((segment_data_FRA['hotel']/total)*100),0)

segment_data_FRA=segment_data_FRA.sort_values(by='hotel',ascending=False)



segment_data_DEU=pd.DataFrame(subset_DEU.groupby(['market_segment'])['hotel'].count())

total = segment_data_DEU['hotel'].sum()

segment_data_DEU['% share']= round(((segment_data_DEU['hotel']/total)*100),0)

segment_data_DEU=segment_data_DEU.sort_values(by='hotel',ascending=False)



segment_data_ESP=pd.DataFrame(subset_ESP.groupby(['market_segment'])['hotel'].count())

total = segment_data_ESP['hotel'].sum()

segment_data_ESP['% share']= round(((segment_data_ESP['hotel']/total)*100),0)

segment_data_ESP=segment_data_ESP.sort_values(by='hotel',ascending=False)



segment_data_BEL=pd.DataFrame(subset_BEL.groupby(['market_segment'])['hotel'].count())

total = segment_data_BEL['hotel'].sum()

segment_data_BEL['% share']= round(((segment_data_BEL['hotel']/total)*100),0)

segment_data_BEL=segment_data_BEL.sort_values(by='hotel',ascending=False)



x=np.arange(9)

w=0.5

plt.bar(segment_data_PRT.index,segment_data_PRT['% share'],label='Portugal',alpha=0.3)

plt.legend()

plt.xticks(rotation=90)

plt.show()

plt.bar(segment_data_NLD.index,segment_data_NLD['% share'],label='Netherlands',alpha=0.3)

plt.legend()

plt.xticks(rotation=90)

plt.show()

plt.bar(segment_data_ITA.index,segment_data_ITA['% share'],label='Italy',alpha=0.3)

plt.legend()

plt.xticks(rotation=90)

plt.show()

plt.bar(segment_data_IRL.index,segment_data_IRL['% share'],label='Ireland',alpha=0.3)

plt.legend()

plt.xticks(rotation=90)

plt.show()

plt.bar(segment_data_GBR.index,segment_data_GBR['% share'],label='UK',alpha=0.3)

plt.legend()

plt.xticks(rotation=90)

plt.show()

plt.bar(segment_data_FRA.index,segment_data_FRA['% share'],label='France',alpha=0.3)

plt.legend()

plt.xticks(rotation=90)

plt.show()

plt.bar(segment_data_DEU.index,segment_data_DEU['% share'],label='Germany',alpha=0.3)

plt.legend()

plt.xticks(rotation=90)

plt.show()

plt.bar(segment_data_ESP.index,segment_data_ESP['% share'],label='Spain',alpha=0.3)

plt.legend()

plt.xticks(rotation=90)

plt.show()

plt.bar(segment_data_BEL.index,segment_data_BEL['% share'],label='Belgium',alpha=0.3)

plt.legend()

plt.xticks(rotation=90)

plt.show()
agent=top_data.groupby(['agent','country']).count()

agent=pd.DataFrame(agent)

agent[agent['hotel']==agent['hotel'].max()]['hotel']
agent.loc[9]['hotel']
plt.bar(subset_PRT.groupby('agent')['hotel'].count().index,subset_PRT.groupby('agent')['hotel'].count(),label='PRT')

plt.legend()

plt.show()

plt.bar(subset_NLD.groupby('agent')['hotel'].count().index,subset_NLD.groupby('agent')['hotel'].count(),label='NLD')

plt.legend()

plt.show()

plt.bar(subset_ITA.groupby('agent')['hotel'].count().index,subset_ITA.groupby('agent')['hotel'].count(),label='ITA')

plt.legend()

plt.show()

plt.bar(subset_IRL.groupby('agent')['hotel'].count().index,subset_IRL.groupby('agent')['hotel'].count(),label='IRL')

plt.legend()

plt.show()

plt.bar(subset_GBR.groupby('agent')['hotel'].count().index,subset_GBR.groupby('agent')['hotel'].count(),label='GBR')

plt.legend()

plt.show()

plt.bar(subset_FRA.groupby('agent')['hotel'].count().index,subset_FRA.groupby('agent')['hotel'].count(),label='FRA')

plt.legend()

plt.show()

plt.bar(subset_ESP.groupby('agent')['hotel'].count().index,subset_ESP.groupby('agent')['hotel'].count(),label='ESP')

plt.legend()

plt.show()

plt.bar(subset_DEU.groupby('agent')['hotel'].count().index,subset_DEU.groupby('agent')['hotel'].count(),label='DEU')

plt.legend()

plt.show()

plt.bar(subset_BEL.groupby('agent')['hotel'].count().index,subset_BEL.groupby('agent')['hotel'].count(),label='BEL')

plt.legend()

plt.show()
adr_mean=subset_booked.groupby('country')['adr'].mean()

adr_mean=pd.DataFrame(adr_mean)

adr_count=subset_booked.groupby('country')['adr'].count()

adr_count=pd.DataFrame(adr_count)

adr_std=subset_booked.groupby('country')['adr'].std()

adr_std=pd.DataFrame(adr_std)

adr_median=subset_booked.groupby('country')['adr'].median()

adr_median=pd.DataFrame(adr_median)

adr_mean.columns=['mean']

adr_median.columns=['median']

adr_count.columns=['count']

adr_std.columns=['std']

adr_total=((adr_mean.join(adr_median)).join(adr_count)).join(adr_std)

adr_total=adr_total.sort_values(by='count',ascending=False)

sns.scatterplot(x=adr_total['count'],y=adr_total['median'])

sns.heatmap(adr_total.corr(),center=0)