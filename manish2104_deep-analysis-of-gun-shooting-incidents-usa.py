#IMPORTING PACKAGES

import numpy as np
import pandas as pd
import nltk
import plotly
import re
          
plotly.offline.init_notebook_mode() # run at the start of every notebook
import cufflinks as cf
cf.go_offline()
cf.getThemes()
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
%matplotlib inline
import random
import matplotlib.pyplot as plt
import seaborn as sns
path = "../input/gun-violence-data_01-2013_03-2018.csv"
df_Gun = pd.read_csv(path) # loading csv file in a dataframe
df_Gun.isnull().sum()
# 1- Which state has maximum no. of gun shooting incidents ( based on the state name appearing in the count)
# df_Gun.groupby(['state']).count() # another method to perform this operation
df_Gun['state'].value_counts().iplot(kind = 'bar', theme = 'white', title = 'STATES AND NUMBER OF GUN SHOOTONG INCIDENTS')
# 2- Which city/county has maximum no. of gun shooting incidents ( based on the city name appearing in the count)
df_Gun['city_or_county'].value_counts().head(50).iplot(kind = 'bar', theme = 'white', title = 'CITY AND NUMBER OF GUN SHOOTONG INCIDENTS')
# 3- How many people have been killed and injured in these shooting incidents from 2013 - 2018
df_Gun['Total_Killed/Injured']= df_Gun['n_killed']+df_Gun['n_injured'] # creating a new column and adding the killed plus
                                                                       # injured people into it
print('Total killed plus Injured=',df_Gun['Total_Killed/Injured'].sum()) # Total killed + injured
print('Total Killed=',df_Gun['n_killed'].sum()) # Total killed
print('Total Injured=',df_Gun['n_injured'].sum()) # Total injured
# 6- Age group of participants those died or injured
df_Gun['participant_age_group'] = df_Gun['participant_age_group'].fillna('Null')
df_Gun['participant_age_group'] = df_Gun['participant_age_group'].fillna('Null')
df_Gun['participant_age_group'] = df_Gun['participant_age_group'].str.replace('::',',')
df_Gun['participant_age_group'] = df_Gun['participant_age_group'].str.replace('|',' ')
df_Gun['participant_age_group'] = df_Gun['participant_age_group'].str.replace(',',' ')
df_Gun['participant_age_group']= df_Gun['participant_age_group'].str.replace('\d+', '')
agelist=df_Gun['participant_age_group'].tolist()   

afinallist=[]
for i in agelist:
    afinallist.append(i.split())

count=0
for i in afinallist:
    for k in i:
        if ('Adult'==k):
            count = count+1
            
print('Adult=',count)

count=0
for i in afinallist:
    for k in i:
        if ('Teen'==k):
            count = count+1
            
print('Teen=',count)

count=0
for i in afinallist:
    for k in i:
        if ('Child'==k):
            count = count+1
            
print('Child=',count)

# 7- Gender of participants those died or injured
df_Gun['participant_gender'] = df_Gun['participant_gender'].fillna('Null')
df_Gun['participant_gender'] = df_Gun['participant_gender'].str.replace('::',',')
df_Gun['participant_gender'] = df_Gun['participant_gender'].str.replace('|',' ')
df_Gun['participant_gender'] = df_Gun['participant_gender'].str.replace(',',' ')
df_Gun['participant_gender']= df_Gun['participant_gender'].str.replace('\d+', '')
genderlist=df_Gun['participant_gender'].tolist()   

gfinallist=[]
for i in genderlist:
    gfinallist.append(i.split())

count=0
for i in gfinallist:
    for k in i:
        if ('Male'==k):
            count = count+1
            
print('Male=',count)

count=0
for i in gfinallist:
    for k in i:
        if ('Female'==k):
            count = count+1
            
print('Female=',count)
# 8- How many of the participants were arrested ( means they were shot in the police encounter)

df_Gun['participant_status'] = df_Gun['participant_status'].fillna('Null')
df_Gun['participant_status'] = df_Gun['participant_status'].str.replace('::',',')
df_Gun['participant_status'] = df_Gun['participant_status'].str.replace('|',' ')
df_Gun['participant_status'] = df_Gun['participant_status'].str.replace(',',' ')
df_Gun['participant_status']= df_Gun['participant_status'].str.replace('\d+', '')
statuslist=df_Gun['participant_status'].tolist()   
statuslist
sfinallist=[]
for i in statuslist:
    sfinallist.append(i.split())

count=0
for i in sfinallist:
    for k in i:
        if ('Killed'==k):
            count = count+1
            
print('Killed=',count)

count=0
for i in sfinallist:
    for k in i:
        if ('Injured'==k):
            count = count+1
            
print('Injured=',count)

count=0
for i in sfinallist:
    for k in i:
        if ('Arrested'==k):
            count = count+1
            
print('Arrested=',count)

count=0
for i in sfinallist:
    for k in i:
        if ('Unharmed'==k):
            count = count+1
            
print('Unharmed=',count)

# 9- How many participants those died or got injured were suspects or victims
df_Gun['participant_type'] = df_Gun['participant_type'].fillna('Null')
df_Gun['participant_type'] = df_Gun['participant_type'].str.replace('::',',')
df_Gun['participant_type'] = df_Gun['participant_type'].str.replace('|',' ')
df_Gun['participant_type'] = df_Gun['participant_type'].str.replace(',',' ')
df_Gun['participant_type']= df_Gun['participant_type'].str.replace('\d+', '')
typelist=df_Gun['participant_type'].tolist()   
tfinallist=[]
for i in typelist:
    tfinallist.append(i.split())

count=0
for i in tfinallist:
    for k in i:
        if ('Victim'==k):
            count = count+1
            
print('Victims=',count)

count=0
for i in tfinallist:
    for k in i:
        if ('Subject-Suspect'==k):
            count = count+1
            
print('Subject-Suspect=',count)
# 10- Which year from 2013-2015 had maximum no. of incidents
df_Gun['date'] = df_Gun['date'].fillna('Null')
df_Gun['date'] = df_Gun['date'].str.replace('-',' ')
datelist=df_Gun['date'].tolist()   
dfinallist=[]
for i in datelist:
    dfinallist.append(i.split())
dfinallist

count_2013=0
count_2014=0
count_2015=0
count_2016=0
count_2017=0
count_2018=0
for i in dfinallist:
    for k in i:
        if ('2013'==k):
            count_2013 = count_2013+1
        elif ('2014'==k):
            count_2014 = count_2014+1
        elif ('2015'==k):
            count_2015 = count_2015+1
        elif ('2016'==k):
            count_2016 = count_2016+1
        elif ('2017'==k):
            count_2017 = count_2017+1
        elif ('2018'==k):
            count_2018 = count_2018+1
            
print('Year 2013 incidents=',count_2013)
print('Year 2014 incidents=',count_2014)
print('Year 2015 incidents=',count_2015)
print('Year 2016 incidents=',count_2016)
print('Year 2017 incidents=',count_2017)
print('Year 2018 incidents=',count_2018)

# 11- How many got killed from a particular state and county/city
state_killed = df_Gun[['state', 'n_killed']].groupby(['state'], 
                                   as_index=False).sum().sort_values(by='n_killed', ascending=False).head(20)
city_killed = df_Gun[['city_or_county','n_killed']].groupby(['city_or_county'],as_index=False).sum().sort_values(by='n_killed'
                                    ,ascending=False).head(20)
#12 - How many males and females were involved in these incidents for every state
df_Gun['Males'] = df_Gun['participant_gender'].apply(lambda x: x.count('Male'))   
df_Gun['Females'] = df_Gun['participant_gender'].apply(lambda x: x.count('Female'))
df_Gun['Total_MF'] = df_Gun['Males'] + df_Gun['Females']
state_MF=df_Gun[['state','Total_MF','Males','Females']].groupby(['state'],as_index=False).sum().sort_values(by='Total_MF',
                                                                                                           ascending=False)

state_MF.head(20)[['state','Total_MF','Males','Females']].set_index('state').iplot(kind='bar',
                                        title = 'State wise report - Males,Females involved in the gun shooting incidents')
#12 - How many males and females were involved in these incidents for every city/county
city_MF = df_Gun[['city_or_county','Total_MF','Males','Females']].groupby(['city_or_county'],
                                        as_index=False).sum().sort_values(by='Total_MF',ascending=False)

city_MF.head(20)[['city_or_county','Total_MF','Males','Females']].set_index('city_or_county').iplot(kind='bar',
                                    title = 'City wise report - Males,Females involved in the gun shooting incidents')
#13 - How many incidents 
df_Gun['2013'] = df_Gun['date'].apply(lambda x: x.count('2013'))
df_Gun['2014'] = df_Gun['date'].apply(lambda x: x.count('2014'))
df_Gun['2015'] = df_Gun['date'].apply(lambda x: x.count('2015'))
df_Gun['2016'] = df_Gun['date'].apply(lambda x: x.count('2016'))
df_Gun['2017'] = df_Gun['date'].apply(lambda x: x.count('2017'))
df_Gun['2018'] = df_Gun['date'].apply(lambda x: x.count('2018'))

state_year_incidents = df_Gun[['state','Total_Killed/Injured','2013','2014','2015','2016','2017','2018']].groupby(['state'],as_index = False).sum().sort_values(by='Total_Killed/Injured',ascending=False)

state_year_incidents.head(10)[['state','Total_Killed/Injured','2013','2014','2015','2016','2017','2018']].set_index('state').iplot(kind = 'bar',
                                                                        title = 'Year wise report of killed/injured people in top 10 states')


city_year_incidents = df_Gun[['city_or_county','Total_Killed/Injured','2013','2014','2015','2016','2017','2018']].groupby(['city_or_county'],as_index = False).sum().sort_values(by='Total_Killed/Injured',ascending=False)

city_year_incidents.head(10)[['city_or_county','Total_Killed/Injured','2013','2014','2015','2016','2017','2018']].set_index('city_or_county').iplot(kind = 'bar',
                                                                        title = 'Year wise report of killed/injured people in top 10 cities')


yearlist = []
years = ['2013','2014','2015','2016','2017','2018']
for i in dfinallist:
    for k in i:
        
        if k in years:
            yearlist.append(k)

# No. of persons killed, injured by year
df_Gun['Year'] = yearlist

incidents_year = df_Gun[['Year','n_killed','n_injured','Total_Killed/Injured']].groupby(['Year'], as_index = False).sum().sort_values(by='Year',
ascending = True)

incidents_year[['Year','n_killed','n_injured','Total_Killed/Injured']].set_index('Year').iplot(kind = 'bar',
                                        theme='solar', title = 'YEAR WISE- Total killed, injured report')
incidents_year[['Year','n_killed','n_injured','Total_Killed/Injured']].set_index('Year').iplot(kind = 'scatter',
                            title = 'Trend - YEAR WISE- Total killed, injured report',theme = 'solar')
incidents_year[['Year','Total_Killed/Injured']].iplot(kind='pie',labels='Year',values='Total_Killed/Injured',
                                    pull=.2,hole=0.2,textposition='outside',textinfo='value+percent',
                                                      title='TOTAL KILLED/INJURED YEAR WISE')
df_Gun['participant_type'] = tfinallist
df_Gun['type_victim'] = df_Gun['participant_type'].apply(lambda x: x.count('Victim'))
df_Gun['type_subject-suspect'] = df_Gun['participant_type'].apply(lambda x: x.count('Subject-Suspect'))

victim_suspects_year = df_Gun[['Year','type_victim','type_subject-suspect']].groupby(['Year'],as_index = False).sum()

victim_suspects_year[['Year','type_victim','type_subject-suspect']].set_index('Year').iplot(kind = 'bar',theme = 'solar',
                                        title= 'YEAR WISE- VICTIM,SUBJECT SUSPECT REPORT IN THE GUN SHOOTING INCIDENTS')
victim_suspects_state = df_Gun[['state','type_victim','type_subject-suspect']].groupby(['state'],as_index = False).sum().sort_values(by='type_victim'and'type_subject-suspect',ascending = False)

victim_suspects_state.head(20)[['state','type_victim','type_subject-suspect']].set_index('state').iplot(kind='bar', theme ='solar',
                                        title = 'STATE WISE REPORT - Victims and Subject Suspects')
victim_suspects_state.head(10)[['state','type_victim','type_subject-suspect']].iplot(kind='pie',labels='state',values='type_subject-suspect',
                                    pull=.2,hole=0.2,textposition='outside',textinfo='value+percent',
                                        title = 'STATE WISE REPORT - Subject Suspects')

victim_suspects_city = df_Gun[['city_or_county','type_victim','type_subject-suspect']].groupby(['city_or_county'],as_index = False).sum().sort_values(by='type_victim'and'type_subject-suspect',ascending = False)

victim_suspects_city.head(20)[['city_or_county','type_victim','type_subject-suspect']].set_index('city_or_county').iplot(kind='bar', theme ='solar',
                                        title = 'CITY WISE REPORT - Victims and Subject Suspects')
victim_suspects_city.head(10)[['city_or_county','type_victim','type_subject-suspect']].iplot(kind='pie',labels='city_or_county',values='type_subject-suspect',
                                    pull=.2,hole=0.2,textposition='outside',textinfo='value+percent',
                                        title = 'CITY WISE REPORT - Subject Suspects')
df_Gun['status_killed'] = df_Gun['participant_status'].apply(lambda x: x.count('Killed'))
df_Gun['status_Injured'] = df_Gun['participant_status'].apply(lambda x: x.count('Injured'))
df_Gun['status_Arrested'] = df_Gun['participant_status'].apply(lambda x: x.count('Arrested'))
df_Gun['status_Unharmed'] = df_Gun['participant_status'].apply(lambda x: x.count('Unharmed'))

participant_status_year = df_Gun[['Year','status_killed','status_Arrested','status_Injured','status_Unharmed']].groupby(['Year'],as_index = False).sum()

participant_status_year[['Year','status_killed','status_Arrested','status_Injured','status_Unharmed']].set_index('Year').iplot(kind='bar',
                                                            title = 'YEAR WISE REPORT - Killed, Arrested, Injured, Unharmed')
participant_status_year[['Year','status_killed','status_Arrested','status_Injured','status_Unharmed']].iplot(kind='pie',labels='Year',values='status_Arrested',
                                    pull=.2,hole=0.2,textposition='outside',textinfo='value+percent',
                                        title = 'YEAR WISE REPORT - Arrested people in the gun shooting incidents')
df_Gun['group_Adult'] = df_Gun['participant_age_group'].apply(lambda x: x.count('Adult'))
df_Gun['group_Teen'] = df_Gun['participant_age_group'].apply(lambda x: x.count('Teen'))
df_Gun['group_Child'] = df_Gun['participant_age_group'].apply(lambda x: x.count('Child'))

df_Gun[['group_Adult','group_Teen','group_Child']].sum().iplot(kind= 'bar', 
                                    title = 'Age Group of people involved in the Gun shooting incidents')
guns = df_Gun['gun_type']
guns = guns.dropna()
guns = [x for x in guns if x != '0::Unknown' and x!='0:Unknown']

allguns=[]
for i in guns:
    result = re.sub("\d+::", "", i)
    result = re.sub("\d+:", "", result)
    result = result.split("|")
    for j in result:
        allguns.append(j)

allguns = [x for x in allguns if x != 'Unknown']
allguns = [x for x in allguns if x]

df_gtypes = pd.DataFrame()
df_gtypes['gun_types_clean'] = allguns

df_gtypes['gun_types_clean'].value_counts().iplot(kind='bar', title = 'Guns used for shooting in incidents from 2013-2018')
df_Gun['gun_stolen'] = df_Gun['gun_stolen'].fillna('Null')

df_Gun['gun_stolen'] = df_Gun['gun_stolen'].str.replace('::',',')
df_Gun['gun_stolen'] = df_Gun['gun_stolen'].str.replace('|',' ')
df_Gun['gun_stolen'] = df_Gun['gun_stolen'].str.replace(',',' ')
df_Gun['gun_stolen']= df_Gun['gun_stolen'].str.replace('\d+', '')


df_Gun['Stolenguns']=df_Gun['gun_stolen'].apply(lambda x: x.count('Stolen'))
df_Gun['stolenguns']=df_Gun['gun_stolen'].apply(lambda x: x.count('stolen'))
df_Gun['Stolengunstotal'] = df_Gun['Stolenguns'] + df_Gun['stolenguns']

df_year_stolenguns = df_Gun[['Year','Stolengunstotal']].groupby(['Year'], as_index = False).sum()
df_year_stolenguns[['Year','Stolengunstotal']].set_index('Year').iplot(kind = 'line',
                            title = 'Trend - Year wise report of stolen guns',theme = 'solar')
df_year_stolenguns[['Year','Stolengunstotal']].set_index('Year').iplot(kind = 'bar', title = 'Year wise report of stolen guns')