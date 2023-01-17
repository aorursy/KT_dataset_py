import pandas as pd

import numpy as np

import datetime





import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)





dateparse = lambda x: datetime.datetime.strptime(x,'%m/%d/%Y %I:%M:00 %p')



 



# Read data 

d=pd.read_csv("../input/crime.csv",parse_dates=['incident_datetime'],date_parser=dateparse)



d['year_month']=d['incident_datetime'].apply(lambda x:  x.strftime('%Y-%m-01 00:00:00'))

d['year']=d['incident_datetime'].apply(lambda x:  x.strftime('%Y'))
d=d[d['address_1'].str.contains('500 Block RICES MILL', na=False) ]

d.head()
g=d.groupby(['year']).size().reset_index()

g.columns = ['year','count']

g
g=d.groupby(['year','parent_incident_type']).size().reset_index()

g.columns = ['year', 'parent_incident_type','count']

g[g['parent_incident_type']=='Assault']
#d[(d['parent_incident_type']=='Assault') & (d['year']=='2017')][['incident_description','incident_datetime','case_number']]
d['e']=1

g=d[d['address_1'].str.contains('500 Block RICES MILL', na=False) ]



g=g[g['parent_incident_type'].isin(['Assault','Disorder','Drugs','Theft'])]





p=pd.pivot_table(g, values='e', index=['incident_datetime'], columns=['parent_incident_type'], aggfunc=np.sum)



# Resampling year end 'A'.  This is very powerful

pp=p.resample('A', how=[np.sum]).reset_index()

pp.fillna(0, inplace=True)

pp.columns = pp.columns.get_level_values(0)

pp
g=d[d['address_1'].str.contains('500 Block RICES MILL', na=False) ]



g=g[g['parent_incident_type'].isin(['Assault','Disorder','Drugs','Theft'])]



g[g['incident_datetime'] >= '2017-01-01'][['incident_datetime','case_number','incident_description','address_1']]