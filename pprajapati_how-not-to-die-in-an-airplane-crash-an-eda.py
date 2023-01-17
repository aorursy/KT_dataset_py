import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
#Read the data

airplane =pd.read_csv('../input/Airplane_Crashes_and_Fatalities_Since_1908.csv', parse_dates=['Date'], index_col='Date' )

airplane.head(2)
# Add another column in airplane dataframe for survivors 

airplane['Survivor']=airplane['Aboard']-airplane['Fatalities']

airplane.head(2)
# plot using pandas plot

survior=airplane.resample('YS').agg({'Survivor':'sum',

                                    'Ground':'sum',

                                    'Fatalities':'sum',

                                    'Aboard':'sum'})

sx=survior.plot(kind='line',figsize=(15,5))
#Find % of Total ground fatalities and % of total 

ag=airplane.resample('YS').agg({'Ground':['sum',lambda x: x.sum()/airplane['Ground'].sum()]})



#Label the columns

ag.columns = ag.columns.map('_'.join).str.replace('<lambda>','% of Total')



#Find the year when maximun ground fatalities happened

ag.sort_values(by=['Ground_% of Total'],ascending=False).head()
aa=airplane.resample('YS').agg({'Fatalities':'sum'}).sort_values(by=['Fatalities'], ascending =False)

aa.rename(columns={'Fatalities':'Total Fatalities per year'}).head(10)

# Find sum and % of Total fatalities by Operator

gp=airplane.groupby('Operator').agg({'Fatalities':['sum',lambda x: x.sum()/airplane['Fatalities'].sum()]})



#Flatten the multiindex 

gp.columns=gp.columns.map('_'.join)



#Rename Columns

gp.rename(columns={'Fatalities_sum':'Total Fatalities',

                   'Fatalities_<lambda>':'% of Total Fatalities'},inplace=True)



#Sort by Total Fatalities

gp.sort_values(by='Total Fatalities', ascending=False).head()

filt=airplane['Operator']=='Aeroflot'

airplane2=airplane[filt]

airplane2.resample('YS').agg({'Fatalities':'sum'}).plot(kind='line',figsize=(15,5));
gp.reset_index(inplace=True)
#List of all operators with word aeroflot 

filt=gp['Operator'].str.contains('Aeroflot')

aeroflot=gp[filt]

aeroflot.head()

#aeroflot['Operator'].unique()



#no. of uniques operators with word 'Aeroflot' in it

aeroflot['Operator'].nunique()
aeroflot.groupby('Operator').agg({'Total Fatalities':sum}).sort_values(by='Total Fatalities', ascending=True).plot.barh();
filt=airplane['Operator']=='Aeroflot'

aerotype=airplane.loc[filt,['Operator','Type','Fatalities']]

aerotype.head()
aerotype.groupby('Type').agg({'Fatalities':'sum'}).sort_values('Fatalities',ascending=False).head(10)
gp=airplane.groupby('Location').agg({'Fatalities':['sum',lambda x: x.sum()/airplane['Fatalities'].sum()]})



#Flatten the multiindex 

gp.columns=gp.columns.map('_'.join)



#Rename Columns

gp.rename(columns={'Fatalities_sum':'Total Fatalities',

                   'Fatalities_<lambda>':'% of Total Fatalities'},inplace=True)



#Sort by Total Fatalities

gp.sort_values(by='Total Fatalities', ascending=False).head()
tg=pd.Grouper(freq='YS')

airplane.groupby([tg,'Registration']).agg({'Fatalities':'sum'}).sort_values(by=['Fatalities'],ascending=False).head(10)
#tg=pd.Grouper(freq='YS')

airplane.groupby(['Type']).agg({'Fatalities':'sum'}).sort_values(by=['Fatalities'],ascending=False).head(10).plot.bar()
filt=airplane['Type']=='Douglas DC-3'

doug=airplane.loc[filt,['Operator','Fatalities']].sort_values('Fatalities', ascending=False)

doug.head(10)
airplane['Route'].isna().count()
airplane['Route'].notna().count()
#No. of plane crashes per route

route=airplane.groupby('Route').agg({'Fatalities':['sum',lambda x:x.sum() / airplane['Fatalities'].sum()]})

route.head()
route.columns=route.columns.map(''.join)

route.head()
route.reset_index(inplace=True)

route.head()
route.rename(columns={'Fatalitiessum':'Total Fatalities',

                      'Fatalities<lambda>':'% of Total Fatalities'}, inplace=True)

route.head()
#Sort by desc

route.sort_values(by='Total Fatalities', ascending=False).head()
route=airplane.loc[airplane['Route']=='Training',['Route','Aboard','Fatalities']]

route.head()
# No. of Trainig flights per year

route.resample('YS').size().plot(kind='line', figsize=(10,5));
route.resample('YS').agg({'Fatalities':['sum'],

                         'Aboard':'sum'}).plot(kind='line', figsize=(10,5));