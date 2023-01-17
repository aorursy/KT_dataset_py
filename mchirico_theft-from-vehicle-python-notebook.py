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



d=d[d['incident_datetime'] >= "2000-01-01 00:00:00"]



d['timeStamp'] = d.incident_datetime
d['parent_incident_type'].value_counts()
t=d[(d.parent_incident_type == "Theft from Vehicle")]

t.head()



g = d.groupby(['parent_incident_type'])

#t['timeStamp'] = pd.Timestamp(t['incident_datetime'])



d['timeStamp']=pd.DatetimeIndex(d['timeStamp'])



#

#t['timeStamp']
d['e']=1



# Look at subset

t=d[d['parent_incident_type'].isin(['Breaking & Entering',

                                    'Disorder',

                                    'Drugs',

                                    'Theft from Vehicle',

                                   'Property Crime'])]



p=pd.pivot_table(t, values='e', index=['timeStamp'], columns=['parent_incident_type'], aggfunc=np.sum)



# Resampling every week 'W' or 'M'  This is very powerful

pp=p.resample('6W', how=[np.sum]).reset_index()

pp.sort_values(by='timeStamp',ascending=False,inplace=True)



# Let's flatten the columns 

pp.columns = pp.columns.get_level_values(0)



# Show values

# Note, last week might not be a full week

#pp.tail(3)

pp
d['e']=1



# Look at subset

t=d[d['parent_incident_type'].isin(['Breaking & Entering',

                                    'Disorder',

                                    'Drugs',

                                    'Theft from Vehicle',

                                   'Property Crime'])]



p=pd.pivot_table(t, values='e', index=['timeStamp'], columns=['parent_incident_type'], aggfunc=np.sum)



# Resampling every week 'W' or 'M'  This is very powerful

pp=p.resample('2W', how=[np.sum]).reset_index()

pp.sort_values(by='timeStamp',ascending=False,inplace=True)



# Let's flatten the columns 

pp.columns = pp.columns.get_level_values(0)



# Show values

# Note, last week might not be a full week

#pp.tail(3)

pp