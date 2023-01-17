import pandas as pd

import numpy as np

import datetime





import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)





dateparse = lambda x: datetime.datetime.strptime(x,'%m/%d/%y %H:%M')



# Read data 

d=pd.read_csv("../input/crime.csv",parse_dates=['incident_datetime'],date_parser=dateparse)
d['year_month']=d['incident_datetime'].apply(lambda x:  x.strftime('%Y-%m-01 00:00:00'))

d['year']=d['incident_datetime'].apply(lambda x:  x.strftime('%Y'))
d.head()
# Use sum. We're summing up all  Dc_Dist

g=d[(d['city']=='ELKINS PARK')].groupby(['year_month','parent_incident_type']).size().reset_index()

g2=d[(d['city']=='ELKINS PARK')].groupby(['year','parent_incident_type']).size().reset_index()



# Take a look at 'Burglary Residential'

#gg=g[(g['Text_General_Code'] == 'Burglary Residential')]

#gg.head()
g=g[g['year_month']> '1969-12-01 00:00:00']

g2=g2[g2['year']> '1969']
g.columns = ['year_month', 'parent_incident_type','count']

g2.columns = ['year', 'parent_incident_type','count']

g.head()
g[g['parent_incident_type']=='Breaking & Entering'].head()
g2[g2['parent_incident_type']=='Breaking & Entering']