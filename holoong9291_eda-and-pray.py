import numpy as np

import pandas as pd



from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
cov_data = pd.read_csv('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')

cov_data.Date = pd.to_datetime(cov_data.Date)

cov_data['Last Update'] = pd.to_datetime(cov_data['Last Update'])



whole_china = ['China']

parts_china = ['Mainland China','Macau','Hong Kong','Taiwan']

data_whole = cov_data[~cov_data.Country.isin(parts_china)].copy()

data_parts = cov_data[~cov_data.Country.isin(whole_china)].copy()



world_coordinates = pd.read_csv('../input/world-coordinates/world_coordinates.csv')
data_parts.sort_values(by='Confirmed', ascending=False)
data_parts['Province'] = data_parts['Province/State']

data_parts['LastUpdateFixed'] = data_parts['Last Update'].astype('object')

data_parts = data_parts.drop('Last Update', axis=1)

data_parts['LastUpdateFixed'] = data_parts['LastUpdateFixed'].apply(

    lambda lu: lu if not ('2020-03' in str(lu) or '2020-04' in str(lu)) else ((str(lu).split(' ')[0].split('-')[0])+'-'+(str(lu).split(' ')[0].split('-')[2])+'-'+(str(lu).split(' ')[0].split('-')[1]))+' '+str(lu).split(' ')[1])
tmp = data_parts.drop('Sno',axis=1).groupby('Country').max().sort_values(by='Confirmed', ascending=False)



print('Confirmed Mainland Chind / World : {%d}/{%d}, about %d%% in Mainland Chine.' % (tmp.iloc[0]['Confirmed'],tmp['Confirmed'].sum(),int(tmp.iloc[0]['Confirmed']/tmp['Confirmed'].sum()*100)))



tmp.iloc[:10]
tmp = data_parts.query('Country=="Mainland China"').drop('Sno',axis=1).groupby('Province/State').max().sort_values(by='Confirmed', ascending=False)



print('Confirmed Hubei / Mainland China : {%d}/{%d}, about %d%% in Hubei.' % (tmp.iloc[0]['Confirmed'],tmp['Confirmed'].sum(),int(tmp.iloc[0]['Confirmed']/tmp['Confirmed'].sum()*100)))

print('Provinces with more than 100 confirmed people : {%s}' % tmp.query('Confirmed>100').index.tolist())
plt.subplots(figsize=(20, 12))



plt.subplot(3,1,1)

plt.xticks(rotation=40)

sns.barplot(x="Province", y="Confirmed", data=tmp.query('Confirmed>0'))



plt.subplot(3,1,2)

plt.xticks(rotation=40)

tmp['Deaths_Confirmed'] = tmp.Deaths / tmp.Confirmed

sns.barplot(x="Province", y="Deaths_Confirmed", data=tmp.query('Deaths>0'))



plt.subplot(3,1,3)

plt.xticks(rotation=40)

tmp['Recovered_Confirmed'] = tmp.Recovered / tmp.Confirmed

sns.barplot(x="Province", y="Recovered_Confirmed", data=tmp.query('Recovered>0'))
tmp = data_parts.query('Country=="Mainland China" and Province=="Hubei"').drop(['Sno','Province/State'],axis=1).sort_values(by='Date')
plt.subplots(figsize=(18, 8))



plt.subplot(2,3,1)

plt.xticks(rotation=40)

sns.lineplot(x="Date", y="Confirmed", data=tmp)



plt.subplot(2,3,2)

plt.xticks(rotation=40)

sns.lineplot(x="Date", y="Deaths", data=tmp)



plt.subplot(2,3,3)

plt.xticks(rotation=40)

sns.lineplot(x="Date", y="Recovered", data=tmp)



plt.subplot(2,3,4)

plt.xticks(rotation=40)

tmp['Deaths/Confirmed'] = tmp['Deaths'] / tmp['Confirmed']

sns.lineplot(x="Date", y="Deaths/Confirmed", data=tmp)



plt.subplot(2,3,5)

plt.xticks(rotation=40)

tmp['Recovered/Confirmed'] = tmp['Recovered'] / tmp['Confirmed']

sns.lineplot(x="Date", y="Recovered/Confirmed", data=tmp)



plt.subplot(2,3,6)

plt.xticks(rotation=40)

tmp['Deaths/Recovered'] = tmp['Deaths'] / tmp['Recovered']

sns.lineplot(x="Date", y="Deaths/Recovered", data=tmp)