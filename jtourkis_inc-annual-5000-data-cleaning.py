import pandas as pd
import numpy as np
import matplotlib. pyplot as plt
inc_5000 = pd.read_csv("...Tableau_Data/inc5000_fastest_growing_private_companies_10years.csv", index_col = 0)
inc_5000_2018 = pd.read_csv("...Tableau_Data/inc5000_2018_2.csv", index_col = 0)
inc_5000_2019 = pd.read_csv("...Tableau_Data/inc5000_2019_2.csv", index_col = 0)

inc_5000.reset_index(drop=False, inplace=True)
inc_5000.head()

inc_5000_2018.reset_index(drop=False, inplace=True)
inc_5000_2018['year']=2018
inc_5000_2018.columns = ['rank','city', 'growth', 'workers','company','state_I','state_s','revenue','industry','yrs_on_list', 'metro','year'] 
inc_5000_2018.head()
inc_5000_2019
inc_5000_2019.reset_index(drop=False, inplace=True)
####Convert String Revenue to Numeric
inc_5000_2019['year']=2019
inc_5000_2019.columns = ['rank','company', 'state_s', 'revenue','growth','industry','workers','yrs_on_list','metro','city', 'year'] 

#####Create a Dummy for String Containing

inc_5000_2019['Mill_Dum'] = inc_5000_2019['revenue'].str.contains('Million', regex=True)
inc_5000_2019['Bill_Dum'] = inc_5000_2019['revenue'].str.contains('Billion', regex=True)

inc_5000_2019['revenue']=inc_5000_2019['revenue'].str.replace('Million','')
inc_5000_2019['revenue']=inc_5000_2019['revenue'].str.replace('Billion','')
inc_5000_2019['revenue'] = pd.to_numeric(inc_5000_2019['revenue'])
####Use dummy to multiply numeric by the correct amount
inc_5000_2019['Millions']=inc_5000_2019['Mill_Dum']*1000000
inc_5000_2019['Billions']=inc_5000_2019['Bill_Dum']*1000000000
inc_5000_2019['revenue_multiple']=inc_5000_2019['Millions']+inc_5000_2019['Billions']
inc_5000_2019['revenue']=inc_5000_2019['revenue']*inc_5000_2019['revenue_multiple']
inc_5000_2019['state_I']=np.nan
del inc_5000_2019['Millions']
del inc_5000_2019['Billions']
del inc_5000_2019['revenue_multiple']
del inc_5000_2019['Mill_Dum']
del inc_5000_2019['Bill_Dum']

inc_5000_2019.head()
inc_5000_2019.dtypes

frames = [inc_5000, inc_5000_2018, inc_5000_2019]
inc_5000_2007_2019 = pd.concat(frames, sort=True)
inc_5000_2007_2019.head()
inc_5000_2007_2019['perc_growth']=inc_5000_2007_2019['growth']

inc_5000_2007_2019['Prio_Rev_4_Years']=inc_5000_2007_2019['revenue']/((inc_5000_2007_2019['growth']/100)+1)
inc_5000_2007_2019['growth']=inc_5000_2007_2019['revenue']-inc_5000_2007_2019['Prio_Rev_4_Years']
#####Check to make sure no companies on list have negative revenue growth

print(inc_5000_2007_2019[inc_5000_2007_2019['revenue']<0])
####Check to makes sure categories for industry are the same. 
inc_5000['industry'].unique()
inc_5000_2018['industry'].unique()
inc_5000_2019['industry'].unique()
inc_5000_2007_2019['industry'].unique()
#####Comine IT Services as in Prior Years
inc_5000_2007_2019["industry"].replace({"IT Management": "IT Services", "IT System Development": "IT Services"}, inplace=True)
inc_5000_2007_2019['industry'].unique()
inc_5000_2007_2019['state_I'].unique()
#inc_5000_2007_2019['state_s'].unique()
states = {"AL":"Alabama","AK":"Alaska","PR":"Puerto Rico", "AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut","DE":"Delaware","DC":"District of Columbia","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa","KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan","MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire","NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio","OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota","TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia","WI":"Wisconsin","WY":"Wyoming"}
inc_5000_2007_2019['state_temp']=inc_5000_2007_2019['state_s']
inc_5000_2007_2019['state_temp'].replace(states, inplace=True)
inc_5000_2007_2019['state_I']=inc_5000_2007_2019['state_temp']
del inc_5000_2007_2019['state_temp']
inc_5000_2007_2019['state_I'].unique()
inc_5000_2007_2019.isnull().sum(axis = 0)


inc_5000_2007_2019[inc_5000_2007_2019['state_I'].isnull()]
inc_5000_2007_2019=inc_5000_2007_2019[inc_5000_2007_2019['state_I'].notnull()]
inc_5000_2007_2019.isnull().sum(axis = 0)

#inc_5000_2007_2019.to_csv('inc5000_12_year.csv', index = False, header = True)
inc_multi_year_comps=inc_5000_2007_2019[inc_5000_2007_2019['yrs_on_list']>1]
co_list=inc_multi_year_comps.company.unique()

#####Filter out companies from full set that only appear once

inc_multi_year_comps_2=inc_5000_2007_2019[inc_5000_2007_2019['company'].isin(co_list)]
piv=pd.pivot_table(inc_multi_year_comps,index=["company"],values=["revenue"],
               columns=["yrs_on_list"],aggfunc=[np.sum])
piv

inc_5000_2007_2019['yrs_on_max'] = inc_5000_2007_2019.groupby(['company'])['yrs_on_list'].transform(max)
inc_5000_2007_2019
