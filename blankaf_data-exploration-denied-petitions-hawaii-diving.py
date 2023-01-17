import pandas as pd

import numpy as np

from decimal import Decimal

import matplotlib.pyplot as plt

% matplotlib inline

import seaborn as sns   
h1b = pd.read_csv('../input/h1b_kaggle.csv')

del h1b['Unnamed: 0']

len(h1b)
h1b = h1b.dropna()                         # little rough cleaning

h1b.reset_index()

lng = len(h1b)

print(lng)
h1b.loc[:,'WORKSITE']=h1b.loc[:,'WORKSITE'].apply(lambda rec:rec.split(',')[1][1:]) 

                               # for getting the state of WORKSITE it is necessary

                               # to split the string and remove the space after comma

def change_NA(rec):            # There are 53 "states" incl. D.C., Puerto Rico and

    if (rec=='NA'):               # Mariana Islands, which were abbreviated as "NA"

        return 'MARIANA ISLANDS'

    return rec

h1b.loc[:,'WORKSITE']=h1b.loc[:,'WORKSITE'].apply(lambda rec: change_NA(rec))

print(len(h1b['WORKSITE'].unique()))
h1b.rename(columns={'EMPLOYER_NAME': 'EMPLOYER', 'FULL_TIME_POSITION': 'FULL_T',

                    'PREVAILING_WAGE': 'PREV_WAGE','WORKSITE': 'STATE',

                    'lon':'LON', 'lat':'LAT'}, inplace=True )

columns_to_keep=['CASE_STATUS','YEAR','STATE','SOC_NAME','JOB_TITLE','FULL_T',

                 'PREV_WAGE','EMPLOYER','LON', 'LAT']

h1b = h1b[columns_to_keep] 

h1b.columns
h1b['LON'] = h1b['LON'].apply(lambda lon: float("%.2f" %lon))           # rounding 

h1b['LAT'] = h1b['LAT'].apply(lambda lat: float("%.2f" %lat))

h1b['YEAR'] = h1b['YEAR'].apply(lambda year:'%g' % (Decimal(str(year))))

h1b['PREV_WAGE'] = h1b['PREV_WAGE'].apply(lambda wage:'%g' % (Decimal(str(wage))))

h1b.head(2)
h1b['CASE_STATUS'].unique()
status_freq = [0]*7                        # petitions distributions by status

statuses=['CERTIFIED-WITHDRAWN','CERTIFIED','DENIED','WITHDRAWN','REJECTED', 'INVALIDATED',

            'PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED']

for i in range(0,7):

    status_freq[i] = h1b[h1b.CASE_STATUS==statuses[i]]['CASE_STATUS'].count()

status_freq
from matplotlib.pyplot import pie, axis, show

import matplotlib as mpl

sns.set_context("notebook",font_scale=1.2)

plt.figure(figsize=(4.5,4.5))

plt.title('PETITIONS BY CASE STATUS')

axis('equal');

pie(status_freq[:4], labels=statuses[:4]);

show()
years = ['2011','2012','2013','2014','2015','2016']  

year_count = [0]*6                              # petitions distributions by year        

for i in range(0,6):

    year_count[i] = h1b[h1b.YEAR==years[i]]['YEAR'].count()

year_count       
sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(13,3))

plt.title('PETITIONS DISTRIBUTION BY YEAR')

sns.countplot(h1b['YEAR'])
denied = h1b[h1b.CASE_STATUS=='DENIED']           # subset of denied petitions

len(denied)
del denied['CASE_STATUS']                         

denied = denied.reset_index()

denied.head(2)
denied_year_count = [0]*6                     # denied petitions distributions by year  

for i in range(0,6):

    denied_year_count[i] = denied[denied.YEAR==years[i]]['YEAR'].count()

denied_year_count  
sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(13,3))

plt.title('DENIED PETITIONS DISTRIBUTION BY YEAR')

sns.countplot(denied['YEAR'])
denied_year_rate = [0]*6                # rate of denied petitions distributions by year  

for i in range(0,6):

    denied_year_rate[i] = float("%.2f" %((denied_year_count[i]/year_count[i])*100))

    

ratio = pd.DataFrame()

ratio['year'] = years

ratio['denied rate %'] = denied_year_rate

ratio = ratio.set_index(['year'])

ratio.T
ratio = ratio.reset_index()

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(13,3))

plt.title('DENIED PETITIONS RATE BY YEAR')

g = sns.barplot( x='year', y='denied rate %', data=ratio)
US_states = ['ALABAMA','ALASKA','ARIZONA','ARKANSAS','CALIFORNIA','COLORADO',

             'CONNECTICUT','DELAWARE','DISTRICT OF COLUMBIA','FLORIDA','GEORGIA',

             'HAWAII','IDAHO','ILLINOIS','INDIANA','IOWA','KANSAS','KENTUCKY',

             'LOUISIANA','MAINE','MARIANA ISLANDS','MARYLAND','MASSACHUSETTS',

             'MICHIGAN','MINNESOTA','MISSISSIPPI','MISSOURI','MONTANA','NEBRASKA',

             'NEVADA','NEW HAMPSHIRE','NEW JERSEY','NEW MEXICO','NEW YORK',

             'NORTH CAROLINA','NORTH DAKOTA','OHIO','OKLAHOMA','OREGON',

             'PENNSYLVANIA','PUERTO RICO','RHODE ISLAND','SOUTH CAROLINA',

             'SOUTH DAKOTA','TENNESSEE','TEXAS','UTAH','VERMONT','VIRGINIA',

             'WASHINGTON','WEST VIRGINIA','WISCONSIN','WYOMING']

petitions_by_state = [0]*53              # filed petitions distribution by state

for i in range (0,53):

    petitions_by_state[i] = h1b[h1b.STATE==US_states[i]]['STATE'].count() 

pet_state = pd.DataFrame()

pet_state['STATE'] = US_states

pet_state['FILED PETITIONS'] = petitions_by_state 

print(sum(petitions_by_state))
sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(13,5))

plt.title('FILED PETITIONS BY STATE')

g = sns.barplot( x='STATE', y='FILED PETITIONS', data=pet_state)

rotg = g.set_xticklabels(g.get_xticklabels(), rotation=90)
denied_by_state = [0]*53                # denied petitions distributions by state  

for i in range (0,53):

    denied_by_state[i] = denied[denied.STATE==US_states[i]]['STATE'].count()

den_state = pd.DataFrame()

den_state['STATE'] = US_states

den_state['DENIED PETITIONS'] = denied_by_state

print(sum(denied_by_state)) 
sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(13,5))

plt.title('DENIED PETITIONS BY STATE')

g = sns.barplot( x='STATE', y='DENIED PETITIONS', data=den_state)

rotg = g.set_xticklabels(g.get_xticklabels(), rotation=90)
denied_state_rate = [0]*53          # rate of denied petitions distributions by state  

for i in range(0,53):

    denied_state_rate[i] = float("%.2f" %((denied_by_state[i]/petitions_by_state[i])*100))



ratios = pd.DataFrame()

ratios['STATE'] = US_states

ratios['DENIED PETITIONS %'] = denied_state_rate
sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(13,5))

plt.title('DENIED PETITIONS RATE IN % BY STATE')

g = sns.barplot( x='STATE', y='DENIED PETITIONS %', data=ratios)

rotg = g.set_xticklabels(g.get_xticklabels(), rotation=90)
pet_state['DENIED PETITIONS'] = denied_by_state

pet_state['DENIED PETITIONS %'] = denied_state_rate

pet_state = pet_state.sort_values(by='DENIED PETITIONS %',ascending= False)

pet_state
hawaii = h1b[h1b.STATE=='HAWAII']        # subset of Hawaiian petitions

del hawaii['STATE']

del hawaii['LON']

del hawaii['LAT']

hawaii = hawaii.reset_index()

lnh = len(hawaii) 

print(lnh, 'petitions were filed in Hawaii and 309 of them were denied.')
def get_data(rec):           

    if (rec.find('DATA SCIENTIST')<0):

        return 'NO_DATA' 

    return rec

hawaii.loc[:,'JOB_TITLE']=hawaii.loc[:,'JOB_TITLE'].apply(lambda rec: get_data(rec)) 

hawaii_data = hawaii[(hawaii.JOB_TITLE!='NO_DATA')&(hawaii.CASE_STATUS!='DENIED')]

print(len(hawaii_data),'petitions for the job of "DATA SCIENTIST" in the state of',

                        'Hawaii were certified.')
hawaii_data