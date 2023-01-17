

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
suratk = pd.read_excel('../input/Surat Kiosk Walk-in Register 15-12-2018.xlsx', 'Financial')

suratk.head()

suratn = pd.read_excel('../input/Surat Kiosk Walk-in Register 15-12-2018.xlsx', 'Non-Financial')

suratn.head()
suratk.drop(['Unnamed: 9', 'customer feedback if any'], axis =1, inplace=True)

suratk = suratk.rename(columns={'Query if any (account balance, fee query, charges query, account opening query)': 'purpose',

								'Type of Cutomer(CBG/IBG)': 'Type',

								'resolved (Y or N)': 'resolved',

								'type of customer (office boy/accountant/promoter)': 'visitor'})

suratk['Date'] = pd.to_datetime(suratk['Date'])

suratk['purpose']=suratk['purpose'].str.replace('[0-9]+[*]','').str.strip()

suratk['purpose'] = suratk['purpose'].str.lower()
suratn = suratn.rename(columns={'Query if any (account balance, fee query, charges query, account opening query)': 'purpose',

								'Type of Cutomer(CBG/IBG)': 'Type',

								'resolved (Y or N)': 'resolved',

								'type of customer (office boy/accountant/promoter)': 'visitor'})

suratn['Date'] = pd.to_datetime(suratn['Date'])

suratn['purpose']=suratn['purpose'].str.replace('[0-9]+[*]','').str.strip()

suratn['purpose'] = suratn['purpose'].str.lower()
data = suratk.groupby(['customer name', 'purpose', 'resolved'], as_index=False).agg(

						{'Date':['first', 'count', lambda x: x.max() - x.min()],

                         'Type': 'first',

                         'visitor': pd.Series.nunique

						 })
datan = suratn.groupby(['Name', 'purpose', 'resolved'], as_index=False).agg(

						{'Date':['first', 'count', lambda x: x.max() - x.min()],

                         'Type': 'first',

                         'visitor': pd.Series.nunique

						 })
data['firstVisit'] = data['Date']['first']

data['totalVisit'] = data['Date']['count']

data['totalDays'] = data['Date']['<lambda>']

data['uniqueVisiotors'] = data['visitor']['nunique']

data.drop(columns=['Date', 'visitor'], inplace=True)

data.columns = data.columns.droplevel(1)

data.head(20)
datan['firstVisit'] = datan['Date']['first']

datan['totalVisit'] = datan['Date']['count']

datan['totalDays'] = datan['Date']['<lambda>']

datan['uniqueVisiotors'] = datan['visitor']['nunique']

datan.drop(columns=['Date', 'visitor'], inplace=True)

datan.columns = datan.columns.droplevel(1)

datan.head(20)
data.plot(x='firstVisit', y='totalVisit')
purin = data.groupby('purpose', as_index=False).count();

purin.head()

purin.totalVisit.plot.hist(title = 'Customer Unique Purposes of Visit');
pitfallsk = suratk.groupby(['Date', 'Type'], as_index=False).count()

pitfallsk.head()
pitfallsk.plot(x='Date', y='customer name', label='Footfalls')
datan.plot(x='firstVisit', y='totalVisit')
purposein = datan.groupby('purpose', as_index=False).count();

purposein.head()
purposein.totalVisit.plot.hist(title = 'Customer Unique Purposes of Visit');
pitfallsn = suratn.groupby(['Date', 'Type'], as_index=False).count()

pitfallsn.head()
pitfallsn.plot(x='Date', y='Name', label='Footfalls')
data[data['totalVisit']>5]
datan[datan['totalVisit']>5]