import pandas as pd

import numpy as np

from decimal import Decimal

import matplotlib.pyplot as plt

% matplotlib inline

import seaborn as sns
H1b = pd.read_csv("../input/h1b_kaggle.csv")

del H1b['Unnamed: 0']

h1b  = H1b.fillna(method='ffill')                      # little rough cleaning

h1b.head(2)
h1b.columns
for col in h1b.columns:                          # renaming columns to shorter names

    if col[:]=='EMPLOYER_NAME':

        h1b.rename(columns={col:'EMPLOYER'}, inplace=True)

    if col[:]=='FULL_TIME_POSITION': 

        h1b.rename(columns={col:'FULL_T'}, inplace=True)

    if col[:]=='PREVAILING_WAGE': 

        h1b.rename(columns={col:'PREV_WAGE'}, inplace=True)

    if col[:]=='lon':

        h1b.rename(columns={col:'LON'}, inplace=True)

    if col[:]=='lat':

        h1b.rename(columns={col:'LAT'}, inplace=True)    

h1b.columns        
lng = len(h1b) 

lng
years = [2011,2012,2013,2014,2015,2016]          # petitions distributions by year        

year_count = [0]*6

for i in range(0,6):

    year_count[i] = h1b[h1b.YEAR==years[i]]['YEAR'].count()

year_count  
sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(13,3))

plt.title('PETITIONS DISTRIBUTION BY YEAR')

sns.countplot(h1b['YEAR'])
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
denied = h1b[h1b.CASE_STATUS=='DENIED']           # subset of denied petitions

len(denied)
del denied['CASE_STATUS']                         

denied = denied.reset_index()

denied.head(2)
years = [2011,2012,2013,2014,2015,2016]

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

    years[i] = '%g' % (Decimal(str(years[i])))



ratio = pd.DataFrame()

ratio['year'] = years

ratio['denied rate %'] = denied_year_rate

ratio = ratio.set_index(['year'])

ratio.T
ratio = ratio.reset_index()

sns.set_context("notebook",font_scale=1.1)

plt.figure(figsize=(13,3))

plt.title('DENIED PETITIONS RATE BY YEAR')

g = sns.barplot( x='year', y='denied rate %', data=ratio)
US_states = ['ALABAMA','ALASKA','ARIZONA','ARKANSAS','CALIFORNIA','COLORADO','CONNECTICUT',

             'DELAWARE','FLORIDA','GEORGIA','HAWAII','IDAHO','ILLINOIS','INDIANA','IOWA',

             'KANSAS','KENTUCKY','LOUISIANA','MAINE','MARYLAND','MASSACHUSETTS','MICHIGAN',

             'MINNESOTA','MISSISSIPPI','MISSOURI','MONTANA','NEBRASKA','NEVADA',

             'NEW HAMPSHIRE','NEW JERSEY','NEW MEXICO','NEW YORK','NORTH CAROLINA',

             'NORTH DAKOTA','OHIO','OKLAHOMA','OREGON','PENNSYLVANIA','RHODE ISLAND',

             'SOUTH CAROLINA','SOUTH DAKOTA','TENNESSEE','TEXAS','UTAH','VERMONT',

             'VIRGINIA','WASHINGTON','WEST VIRGINIA','WISCONSIN','WYOMING']

denied_by_state = [0]*50                # denied petitions distributions by state 

lngd = len(denied) 

for i in range(0,lngd):

    for j in range (0,50):

        if (denied.loc[i,'WORKSITE'].find(US_states[j]) >0):

            denied_by_state[j]+=1 

sum(denied_by_state)          
den_state = pd.DataFrame()

den_state['STATE'] = US_states

den_state['DENIED PETITIONS'] = denied_by_state

den_state = den_state.set_index(['STATE'])

den_state
den_state = den_state.reset_index()

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(13,5))

plt.title('DENIED PETITIONS BY STATE')

g = sns.barplot( x='STATE', y='DENIED PETITIONS', data=den_state)

rotg = g.set_xticklabels(g.get_xticklabels(), rotation=90)
