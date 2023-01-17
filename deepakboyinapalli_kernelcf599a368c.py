import pandas as pd

import numpy as np

from decimal import Decimal

import matplotlib.pyplot as plt

% matplotlib inline

import seaborn as sn
f = pd.read_csv("h1b_kaggle.csv")

del f['Unnamed: 0']

f.head()
f = f.dropna()

f.reset_index()

lng=len(f)

print(lng)
f
f.EMPLOYER_NAME.value_counts() .head(15)
f['EMPLOYER_NAME'].value_counts() .head(15) .plot(kind='bar',title='Top15 hiring companies')
f.PREVAILING_WAGE.value_counts() .sort_values(ascending=False) .head()
f.PREVAILING_WAGE.mean()
awp =f.groupby(['EMPLOYER_NAME']).mean()['PREVAILING_WAGE'].nlargest(15).plot(kind='bar')
f.WORKSITE.value_counts() .head(20)
f.WORKSITE.value_counts() .head(20).plot(kind='bar')
f.loc[:,'WORKSITE']=f.loc[:,'WORKSITE'].apply(lambda rec:rec.split(',')[1][1:]) 



def change_NA(rec):

    if (rec=='NA'):

        return 'MARIANA ISLANDS'

    return rec

f.loc[:,'WORKSITE']=f.loc[:,'WORKSITE'].apply(lambda rec: change_NA(rec))

print(len(f['WORKSITE'].unique()))



        
f.rename(columns={'EMPLOYER_NAME':'EMPLOYER', 'FULL_TIME_POSITION':'FULL_T', 

                  'PREVAILING_WAGE':'PREV_WAGE','WORKSITE':'STATE',

                 'lon':'LON', 'lat':'LAT'}, inplace=True)

f.columns
f=f[['CASE_STATUS','YEAR','STATE','SOC_NAME','JOB_TITLE','FULL_T',

                'PREV_WAGE','EMPLOYER','LON','LAT']]

f.columns
f['LON']=f['LON'].apply(lambda lon: float("%.2f" %lon))

f['LAT']=f['LAT'].apply(lambda lat: float("%.2f" %lat))

f['YEAR']=f['YEAR'].apply(lambda year: '%g' % (Decimal(str(year))))

f['PREV_WAGE']=f['PREV_WAGE'].apply(lambda wage: '%g' % (Decimal(str(wage))))

f.head(2)
f['CASE_STATUS'].unique()
status_freq=[0]*7

statuses=['CERTIFIED-WITHDRAWN', 'WITHDRAWN', 'CERTIFIED', 'DENIED',

       'REJECTED', 'INVALIDATED',

       'PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED']

for i in range (0,7):

   status_freq[i]= f[f.CASE_STATUS==statuses[i]]['CASE_STATUS'].count()

status_freq
from matplotlib.pyplot import pie, axis,show

import matplotlib as mlp

plt.figure(figsize=(4.5,4.5))

plt.title('PETITIONS BY CASE STATUS')

axis('equal');

pie(status_freq[:4], labels=statuses[:4]);

show()
years=['2011','2012','2013','2014','2015','2016']

year_count=[0]*6

for i in range(0,6):

    year_count[i]=f[f.YEAR==years[i]]['YEAR'].count()

year_count

sn.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(13,3))

plt.title('PETITION DISTRIBUTION BY YEAR')

sn.countplot(f['YEAR'])
denied=f[f.CASE_STATUS=='DENIED']

len(denied)
del denied['CASE_STATUS']

denied=denied.reset_index()

denied.head(2)
denied_year_count=[0]*6

for i in range(0,6):

    denied_year_count[i]=denied[denied.YEAR==years[i]]['YEAR'].count()

denied_year_count    
sn.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(13,3))

plt.title('DENIED PETITION DISTRIBUTION BY YEAR')

sn.countplot(denied['YEAR'])
denied_year_rate=[0]*6

for i in range(0,6):

    denied_year_rate[i]=float("%.2f" %((denied_year_count[i]/year_count[i])*100))

    

ratio=pd.DataFrame()

ratio['year']=years

ratio['denied rate %']=denied_year_rate

ratio=ratio.set_index(['year'])

ratio.T
US_states=['ALABAMA','ALASKA','ARIZONA','ARKANSAS','CALIFORNIA','COLORADO',

           'CONNECTICUT','DELAWARE','DISTRICT OF COLUMBIA','FLORIDA','GEORGIA',

           'HAWAII','IDAHO','ILLINOIS','INDIANA','IOWA','KANSAS','KENTUCKY',

           'LOUISIANA','MAINE','MARIANA ISLANDS','MARYLAND','MASSACHUSETTS',

           'MICHIGAN','MINNESOTA','MISSISSIPPI','MISSOURI','MONTANA','NEBRASKA',

           'NEVADA','NEW HAMPSHIRE','NEW JERSEY','NEW MEXICO','NEW YORK',

           'NORTH CAROLINA','NORTH DAKOTA','OHIO','OKLAHOMA','OREGON',

           'PENNSYLVANIA','PUERTO RICO','RHODE ISLAND','SOUTH CAROLINA',

           'SOUTH DAKOTA','TENNESSEE','TEXAS','UTAH','VERMONT','VIRGINIA',

           'WASHINGTON','WEST VIRGINIA','WISCONSIN','WYOMING']

petitions_by_state=[0]*53

for i in range(0,53):

   petitions_by_state[i]=f[f.STATE==US_states[i]]['STATE'].count()

pet_state=pd.DataFrame()

pet_state['STATE']=US_states

pet_state['FILED PETITIONS']=petitions_by_state

print(sum(petitions_by_state))
sn.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(13,5))

plt.title('FILED PETITION BY STATE')

v=sn.barplot(x='STATE',y='FILED PETITIONS',data=pet_state)

rotg=v.set_xticklabels(v.get_xticklabels(),rotation=90)
denied_by_state=[0]*53

for i in range (0,53):

    denied_by_state[i]=denied[denied.STATE==US_states[i]]['STATE'].count()

den_state=pd.DataFrame()

den_state['STATE']=US_states

den_state['DENIED PETITIONS']=denied_by_state

print(sum(denied_by_state))
sn.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(13,5))

plt.title('DENIED PETITIONS BY STATE')

v=sn.barplot(x='STATE',y='DENIED PETITIONS',data=den_state)

rotg=v.set_xticklabels(v.get_xticklabels(),rotation=90)
denied_state_rate=[0]*53

for i in range(0,53):

    denied_state_rate[i]=float("%.2f" %((denied_by_state[i]/petitions_by_state[i])*100))

    

ratios=pd.DataFrame()

ratios['STATE']=US_states

ratios['DENIED PETITIONS %']=denied_state_rate
sn.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(13,5))

plt.title('DENIED PETITIONS RATE IN % BY STATE')

v=sn.barplot(x='STATE',y='DENIED PETITIONS %',data=ratios)

rotg=v.set_xticklabels(v.get_xticklabels(),rotation=90)
pet_state['DENIED PETITIONS']=denied_by_state

pet_state['DENIED PETITIONS %']= denied_state_rate

pet_state=pet_state.sort_values(by='DENIED PETITIONS %',ascending=False)

pet_state
illinois=f[f.STATE=='ILLINOIS']

del illinois['STATE']

del illinois['LAT']

del illinois['LON']

illinois=illinois.reset_index()

lnh=len(illinois)

print(lnh,'petitions were filed in illinois and 3612 of them were denied')
def get_data(rec):

    if(rec.find('CHIEF PROCESS OFFICER')<0):

        return 'NO_DATA'

    return rec

illinois.loc[:,'JOB_TITLE']=illinois.loc[:,"JOB_TITLE"].apply(lambda rec:get_data(rec))

illinois_data=illinois[(illinois.JOB_TITLE!='NO_DATA')&(illinois.CASE_STATUS!='DENIED')]

print(len(illinois_data),'petitions for the job of "CHIEF PROCESS OFFICER"in the state of','illinois were certified')
illinois_data
f.JOB_TITLE.value_counts().sort_values(ascending=False).head(25)
f.JOB_TITLE.value_counts().sort_values(ascending=False).head(25).plot(kind='bar')