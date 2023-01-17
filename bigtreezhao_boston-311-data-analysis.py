import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from datetime import datetime, date, time

plt.style.use('ggplot') # Look Pretty
yiwu =pd.read_csv('../input/311__2015-2016.csv', index_col='OPEN_DT', parse_dates=True)



yiwu2=yiwu.drop(['CASE_ENQUIRY_ID', 'TARGET_DT', 'CLOSED_DT',

       'CASE_STATUS', 'CLOSURE_REASON', 'CASE_TITLE', 'QUEUE', 'Department', 

       'SubmittedPhoto', 'ClosedPhoto',

       'Location', 'fire_district', 'pwd_district', 'city_council_district',

       'police_district','ward', 'precinct', 'land_usage', 'LOCATION_STREET_NAME',

       'Property_Type', 'Property_ID', 'LATITUDE',

       'LONGITUDE', 'Geocoded_Location'],axis=1)
yiwu2=yiwu2.sort_index(ascending=True)
yiwu201=yiwu2.reset_index()

yiwu202=yiwu201.reset_index()

list = [ datetime.date(datee).isocalendar()[1] for datee in yiwu202['OPEN_DT']]

yiwu202['week']=list  # work great # 
yiwu202['case']=1

yiwu_SUBJECT = yiwu202.pivot_table(index='week',

                                  values='case',

                                  columns='SUBJECT',

                                  aggfunc='count')

yiwu_REASON = yiwu202.pivot_table(index='week',

                                  values='case',

                                  columns='REASON',

                                  aggfunc='count') 



yiwu_SUBJECT2 = yiwu202.pivot_table(index='week',

                                  values='case',

                                  columns=['SUBJECT','REASON'],

                                  aggfunc='count') 



yiwu_SOURCE = yiwu202.pivot_table(index='week',

                                  values='case',

                                  columns=['Source'],

                                  aggfunc='count')

                                 

yiwu_neighbor = yiwu202.pivot_table(index='week',

                                  values='case',

                                  columns=['neighborhood'],

                                  aggfunc='count')
subject_total=yiwu_SUBJECT.sum(axis=0).sort_values(ascending=False)

subject_total.reset_index().plot(kind='bar',x='SUBJECT',y=0,figsize=(8, 8))
# select public work column

pub_work=yiwu_SUBJECT.columns[-4]

Trans_Inspec=yiwu_SUBJECT.columns[[-3,-9]]

yiwu_SUBJECT.plot(x=yiwu_SUBJECT.index,y=pub_work, kind='bar',figsize=(12, 8))
yiwu_SUBJECT.plot(x=yiwu_SUBJECT.index,y=Trans_Inspec, kind='bar',figsize=(18, 8))
yiwu_SUBJECT.plot(x=yiwu_SUBJECT.index,y=['Animal Control', 'Boston Police Department',

       'Boston Water & Sewer Commission', 'City Hall Truck', 'Civil Rights',

       'Disability Department',

       "Mayor's 24 Hour Hotline", 'Neighborhood Services',

       'Parks & Recreation Department', 'Property Management',

               'Veterans', 'Youthline'], kind='line',figsize=(18, 10))

## 311 calling reason graph results  

reason_total=yiwu_REASON.sum(axis=0)

reason_total=reason_total.sort_values(ascending=True)

yiwu_REASON.plot(use_index=True,y=reason_total.index[-1], kind='line',figsize=(8, 8))

yiwu_REASON.plot(use_index=True,y=reason_total.index[-3:-1], kind='line',figsize=(8, 8))

yiwu_REASON.plot(use_index=True,y=reason_total.index[-8:-3], kind='area',figsize=(15, 8))

yiwu_REASON.plot(use_index=True,y=reason_total.index[-15:-8], kind='line',figsize=(15, 8))


## 311 calling neighborhoods graph results  

neighbor_total=yiwu_neighbor.sum(axis=0)

neighbor_total=neighbor_total.sort_values(ascending=True)

# total number cases indicate that feb and march are most busy month

yiwu_neighbor.plot(use_index=True,y=neighbor_total.index[-1], kind='line',figsize=(8, 8))

# 6 neighborhoods with most cases in 2015

yiwu_neighbor.plot(use_index=True,y=neighbor_total.index[-6:], kind='line',figsize=(8, 8))
## 311 calling subject+reason graph

sub_rea_total=yiwu_SUBJECT2.sum(axis=0)

sub_rea_total=sub_rea_total.sort_values(ascending=True)

sub_rea_total.plot(kind='bar',use_index=True,figsize=(15, 10))

yiwu_SUBJECT2.plot(use_index=True,y=sub_rea_total.index[-1], kind='line',figsize=(8, 8))

yiwu_SUBJECT2.plot(use_index=True,y=sub_rea_total.index[-6:-1], kind='line',figsize=(14, 8))

## 311 calling source graph results 

yiwu_SOURCE.plot(x=yiwu_SOURCE.index,y=['Citizens Connect App', 'City Worker App',

       'Constituent Call', 'Employee Generated','Self Service'], kind='line',figsize=(12, 8))

# plot exclude first 9 weeks

yiwu_SOURCE.iloc[9:].plot(x=yiwu_SOURCE.iloc[9:].index,y=['Citizens Connect App', 'City Worker App',

       'Constituent Call', 'Employee Generated','Self Service'], kind='line',figsize=(12, 8))



# area plot

yiwu_SOURCE.plot(use_index=True,y=['Citizens Connect App', 'City Worker App',

       'Constituent Call', 'Employee Generated','Self Service'], kind='area')



# pie plot

source_total=yiwu_SOURCE.sum(axis=0)

source_total[:4].plot.pie(figsize=(8, 8))

yiwu3=yiwu.drop(['CASE_ENQUIRY_ID','CLOSURE_REASON', 'CASE_TITLE', 'QUEUE', 'Department', 

       'SubmittedPhoto', 'ClosedPhoto','Location', 'fire_district',

       'pwd_district', 'city_council_district',

       'police_district','ward', 'precinct', 'land_usage', 'LOCATION_STREET_NAME',

       'Property_Type', 'Property_ID', 'Geocoded_Location'],axis=1)



yiwu3=yiwu3.sort_index(ascending=True)

yiwu301=yiwu3.reset_index()
## calculate duration 

yiwu301['OPEN_DT']=pd.to_datetime(yiwu301['OPEN_DT'])

yiwu301['CLOSED_DT']=pd.to_datetime(yiwu301['CLOSED_DT'])

yiwu301['Duration']=yiwu301['CLOSED_DT']-yiwu301['OPEN_DT']



yiwu301['SOLVE_DAY']=yiwu301['Duration'].dt.days

yiwu301['SOLVE_HOUR']=yiwu301['Duration'].dt.hour ## why no attribute 'hour'?


