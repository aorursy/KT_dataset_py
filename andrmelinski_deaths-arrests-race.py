# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import plotly.express as px
dar=pd.read_csv('/kaggle/input/police-violence-in-the-us/deaths_arrests_race.csv')
mp=dar.iloc[:,0:10].copy()
mp.head(2)
mp.replace(np.nan, 0, inplace=True)
mp['Total_Killed_by_Police']=mp.iloc[:,3:10].sum(axis=1)
mp.head()
mp.rename(columns={'Black People Killed by Police (1/1/2013-12/31/2019)':'Black',
                   'Hispanic People Killed by Police (1/1/2013-12/31/2019)':'Hispanic', 
                  'Native American People Killed by Police (1/1/2013-12/31/2019)':'Native',
                  'Asian People Killed by Police (1/1/2013-12/31/2019)':'Asian', 'Pacific Islanders Killed by Police (1/1/2013-12/31/2019)':'Pacific_Islanders',
                  'White People Killed by Police (1/1/2013-12/31/2019)':'White', 'Unknown Race People Killed by Police (1/1/2013-12/31/2019)':'Unknown'}, inplace=True)
#murders x State
murders_state=mp.groupby(['State'],as_index=False)[['Total_Killed_by_Police']].sum().sort_values('Total_Killed_by_Police', ascending=False)
fig = px.bar(murders_state, x='State', y='Total_Killed_by_Police', title='State x Total killed by Police',height=500)
fig.update_layout(yaxis=dict(title='Total killed by Police'), xaxis=dict(title=''))
fig.show()
#murder_races x PD
mp.groupby(['PD'])[['Black','Hispanic','Native','Asian','Pacific_Islanders','White','Unknown','Total_Killed_by_Police']].sum().sort_values(['Black','Hispanic','Native','Asian','Pacific_Islanders','White'], ascending=[0,0,0,0,0,0])
#races x state and cities
rcs=dar.iloc[:,[0,1,10,11,12,13,14,15,16,17,18]].copy()
rcs.head(2)
rcs.rename(columns={'Amer. Indian':'Amer_Indian', 'Two or\nmore races':'two_or_more_races'},inplace=True)
rcs.rename(columns={'Amer. Indian':'Amer_Indian', 'Two or\nmore races':'two_or_more_races'},inplace=True)
rcs.info()
rcs['Total']=rcs.Total.apply(lambda x: x.replace(',',''))
rcs['Black']=rcs.Black.apply(lambda x: x.replace(',',''))
rcs['White']=rcs.White.apply(lambda x: x.replace(',',''))
rcs['Amer_Indian']=rcs.Amer_Indian.apply(lambda x: x.replace(',',''))
rcs['Asian']=rcs.Asian.apply(lambda x: x.replace(',',''))
rcs['Hawaiian']=rcs.Hawaiian.apply(lambda x: x.replace(',',''))
rcs['two_or_more_races']=rcs.two_or_more_races.apply(lambda x: x.replace(',',''))
rcs['Hispanic']=rcs.Hispanic.apply(lambda x: x.replace(',',''))
rcs.replace(np.nan,'0',inplace=True)
rcs['Other']=rcs.Other.apply(lambda x: x.replace(',',''))
rcs['Total']=rcs.Total.astype('int64')
rcs['Black']=rcs.Black.astype('int64')
rcs['White']=rcs.White.astype('int64')
rcs['Amer_Indian']=rcs.Amer_Indian.astype('int64')
rcs['Asian']=rcs.Asian.astype('int64')
rcs['Hawaiian']=rcs.Hawaiian.astype('int')
rcs['Other']=rcs.Other.astype('int64')
rcs['two_or_more_races']=rcs.two_or_more_races.astype('int64')
rcs['Hispanic']=rcs.Hispanic.astype('int64')
rcs['Black_per1000']=(rcs['Black'].div(rcs['Total'])).mul(1000).astype('int')
rcs['White_per1000']=(rcs['White'].div(rcs['Total'])).mul(1000).astype('int')
rcs['Amer_Indian_per1000']=(rcs['Amer_Indian'].div(rcs['Total'])).mul(1000).astype('int')
rcs['Asian_per1000']=(rcs['Asian'].div(rcs['Total'])).mul(1000).astype('int')
rcs['Hawaiian_per1000']=(rcs['Hawaiian'].div(rcs['Total'])).mul(1000).astype('int')
rcs['Other_per1000']=(rcs['Other'].div(rcs['Total'])).mul(1000).astype('int')
rcs['two_or_more_races']=(rcs['two_or_more_races'].div(rcs['Total'])).mul(1000).astype('int')
rcs['Hispanic']=(rcs['Hispanic'].div(rcs['Total'])).mul(1000).astype('int')
rcs_1000=rcs.sort_values(['Black_per1000', 'White_per1000'], ascending=[0,0])
rcs_1000
state_races=rcs.groupby(['State'], as_index=False)[['Black_per1000','White_per1000','Amer_Indian_per1000','Asian_per1000','Hawaiian_per1000','Other_per1000']].mean().sort_values(['Black_per1000','White_per1000'],ascending=[0,0])
state_races.head(5)
fig = px.scatter(state_races, x='State', y=['Black_per1000','White_per1000','Amer_Indian_per1000','Asian_per1000','Hawaiian_per1000','Other_per1000'], title='Racer per 1000 hab x State',height=500)
fig.update_layout(yaxis=dict(title='Race per 1000 hab'), xaxis=dict(title=''))
fig.show()
#total of violent crimes and arrests
violent_crimes=dar.iloc[:,[2,10,21,22,23,24,25,26,27,28,29,30,31,32]].copy()
violent_crimes
violent_crimes.rename(columns={'Violent crimes 2013 (if reported by agency)':'Violent_crimes_2013','Violent crimes 2014 (if reported by agency)':'Violent_crimes_2014', 
                               'Violent crimes 2015 (if reported by agency)':'Violent_crimes_2015',
                               'Violent crimes 2016 (if reported by agency)':'Violent_crimes_2016',
                               'Violent crimes 2017 (if reported by agency)':'Violent_crimes_2017',
                               'Violent crimes 2018 (if reported by agency)':'Violent_crimes_2018',
                               '2013 Total Arrests (UCR Data)':'2013_Tot_Arrests','2014 Total Arrests': '2014_Tot_Arrests',
                               '2015 Total Arrests':'2015_Tot_Arrests' ,'2016 Total Arrests':'2016_Tot_Arrests' ,'2017 Total Arrests':'2017_Tot_Arrests','2018 Total Arrests':'2018_Tot_Arrests'}, inplace=True)
violent_crimes=violent_crimes.sort_values(['Violent_crimes_2013','Violent_crimes_2014', 'Violent_crimes_2015', 'Violent_crimes_2016', 'Violent_crimes_2017', 'Violent_crimes_2018'],ascending=[0,0,0,0,0,0])
violent_crimes.replace(np.nan,0, inplace=True)

fig = px.bar(violent_crimes, x='PD', y=['Violent_crimes_2013','Violent_crimes_2014', 'Violent_crimes_2015', 'Violent_crimes_2016', 'Violent_crimes_2017', 'Violent_crimes_2018'],barmode='group')

fig.show()
tot_arrest=violent_crimes.sort_values(['2013_Tot_Arrests', '2014_Tot_Arrests', '2015_Tot_Arrests', '2016_Tot_Arrests', '2017_Tot_Arrests','2018_Tot_Arrests'],ascending=[0,0,0,0,0,0])
fig = px.bar(tot_arrest, x='PD', y=['2013_Tot_Arrests', '2014_Tot_Arrests', '2015_Tot_Arrests', '2016_Tot_Arrests', '2017_Tot_Arrests','2018_Tot_Arrests'],barmode='group')
fig.show()
