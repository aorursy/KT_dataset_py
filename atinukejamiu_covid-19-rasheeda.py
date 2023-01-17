# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
df.head()
df.shape
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
df.head()
countrywise=df[df["ObservationDate"]==df["ObservationDate"].max()].groupby(["Country/Region"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'}).sort_values(["Confirmed"],ascending=False)
top_10_confirmed = countrywise.sort_values(['Confirmed'], ascending=False).head(10)
top_10_deaths= countrywise.sort_values(['Deaths'], ascending=False).head(10)
top_10_recoveries=countrywise.sort_values(['Recovered'], ascending=False).head(10)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(27,10))

sns.barplot(x=top_10_deaths["Deaths"],y=top_10_deaths.index, ax=ax1)
ax1.set_title('top 10 death cases')

sns.barplot(x=top_10_recoveries["Recovered"],y=top_10_recoveries.index,ax=ax2)
ax2.set_title('top 10 recoverd cases')

sns.barplot(x=top_10_deaths["Confirmed"] / 10**2,y=top_10_confirmed.index , ax=ax3)
ax3.set_title('top 10 confimed cases')
#countries across africa
africa = ['Nigeria', 'South Africa', 'Senegal', 'Egypt', 'Kenya']

# This empty dataframe is created so that you can concatenate the results of the aggregations from the for loop
df_2 = pd.DataFrame(columns=["Country/Region","ObservationDate", "Confirmed",  "Recovered",  "Deaths"])

for i in df[df['Country/Region'].isin(africa)]['Country/Region']:
    # The reset_index() function is responsible for flattening the dataframe after the group by has been implemented
    african_aggregations = df[df['Country/Region'] == i].groupby(['Country/Region', 'ObservationDate']).agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'}).reset_index()
    # Lastly, you combine both dataframes together
    df_2 = pd.concat([df_2, african_aggregations])
# I discovered that there were duplicates, so I'm handling that here
df_2 = df_2.drop_duplicates()
#plot africa

plt.rcParams["figure.figsize"] = [16,9]
# Instead of multiple plot statements, a for loop can handle this for you
# "africa" is the list I created previously i.e ['Nigeria', 'South Africa', 'Egypt', 'Senegal', 'Kenya']
for i in africa:
    plt.plot(df_2[df_2['Country/Region'] == i]['ObservationDate'], df_2[df_2['Country/Region'] == i]['Confirmed'])
    plt.legend(['Nigeria', 'South Africa', 'Egypt', 'Senegal', 'Kenya'])
    plt.xlabel('Days')
    plt.ylabel('no of confirmed cases')
    plt.title('growth of cases across africa')
    
recent_data = df[df['ObservationDate'] == df['ObservationDate'].max()]
sorted_cases = recent_data.sort_values('Confirmed', ascending=False)
sorted_cases.head()
top_50 = sorted_cases[0:50]
top_50['Province/State'] = top_50['Province/State'].fillna('')
top_50['Country/Region-Province/State'] = top_50['Country/Region'] + "-" + top_50['Province/State']
top_50.head()
plt.rcParams["figure.figsize"] = [16,9]
top_50.plot(kind='barh',x ='Country/Region-Province/State',y='Deaths', color = 'blue')
recent_data[recent_data['Country/Region'] == 'UK']
#countries across africa
nig_df=df[df['Country/Region']=='Nigeria']
datewise_nig= nig_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})

SA_df = df[df['Country/Region']=='South Africa']
datewise_SA= SA_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})

senegal_df = df[df['Country/Region']=='Senegal']
datewise_senegal= senegal_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})

egypt_df = df[df['Country/Region']=='Egypt']
datewise_egypt= egypt_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})

kenya_df = df[df['Country/Region']=='Kenya']
datewise_kenya= kenya_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})


#across asia
mainCH_df=df[df['Country/Region']=='Mainland China']
datewise_mainCH= mainCH_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})

india_df= df[df['Country/Region']=='India']
datewise_india= india_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})

uae_df = df[df['Country/Region']=='United Arab Emirates']
datewise_uae= uae_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})

saudi_df = df[df['Country/Region']== 'Saudi Arabia']
datewise_saudi= saudi_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})

iran_df = df[df['Country/Region']=='Iran']
datewise_iran= iran_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})



#arcoss europe
uk_df=df[df['Country/Region']=='UK']
datewise_uk= uk_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})

italy_df= df[df['Country/Region']=='Italy']
datewise_italy= italy_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})

germany_df = df[df['Country/Region']=='Germany']
datewise_germany= germany_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})

spain_df = df[df['Country/Region']== 'Spain']
datewise_spain= spain_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})

france_df = df[df['Country/Region']=='France']
datewise_france= france_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})

#acros america
us_df = df[df['Country/Region'] == 'US']
datewise_us= us_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})

mex_df = df[df['Country/Region'] == 'Mexico']
datewise_mex= mex_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})

canada_df = df[df['Country/Region']== 'Canada']
datewise_canada= canada_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})

aus_df = df[df['Country/Region'] == 'Australia']
datewise_aus= aus_df.groupby('ObservationDate').agg({'Confirmed':'sum', 'Recovered': 'sum', 'Deaths':'sum'})

#plot africa
plt.rcParams["figure.figsize"] = [16,9]

plt.plot(datewise_nig['Confirmed'])
plt.plot(datewise_SA['Confirmed'])
plt.plot(datewise_egypt['Confirmed'])
plt.plot(datewise_senegal['Confirmed'])
plt.plot(datewise_kenya['Confirmed'])

plt.legend(['Nigeria', 'South Africa', 'Egypt', 'Senegal', 'Kenya',])
plt.xlabel('days')
plt.ylabel('no of confirmed cases')
plt.title('growth of cases across africa')
plt.show()



plt.plot(datewise_nig['Recovered'])
plt.plot(datewise_SA['Recovered'] )
plt.plot(datewise_egypt['Recovered'])
plt.plot(datewise_senegal['Recovered'])
plt.plot(datewise_kenya['Recovered'])

plt.legend(['Nigeria', 'South Africa', 'Egypt', 'Senegal', 'Kenya',])
plt.xlabel('days')
plt.ylabel('no of Recoverd cases')
plt.title('growth of recovery across africa')
plt.show()


plt.plot(datewise_nig['Deaths'])
plt.plot(datewise_SA['Deaths'])
plt.plot(datewise_egypt['Deaths'])
plt.plot(datewise_senegal['Deaths'])
plt.plot(datewise_kenya['Deaths'])

plt.legend(['Nigeria', 'South Africa', 'Egypt', 'Senegal', 'Kenya',])
plt.xlabel('days')
plt.ylabel('no of death cases')
plt.title('death across africa')
plt.show()
#plot europe

plt.rcParams["figure.figsize"] = [16,9]
plt.plot(datewise_uk['Confirmed'])
plt.plot(datewise_spain['Confirmed'])
plt.plot(datewise_germany['Confirmed'])
plt.plot(datewise_france['Confirmed'])
plt.plot(datewise_italy['Confirmed'])
plt.legend(['UK', 'Spain', 'Germany', 'France', 'Italy'])
plt.xlabel('days')
plt.ylabel('# of confirmed cases')
plt.title('# europe confirmed cases')
plt.show()

plt.plot(datewise_uk['Recovered'])
plt.plot(datewise_spain['Recovered'])
plt.plot(datewise_germany['Recovered'])
plt.plot(datewise_france['Recovered'])
plt.plot(datewise_italy['Recovered'])
plt.legend(['UK', 'Spain', 'Germany', 'France', 'Italy'])
plt.xlabel('days')
plt.ylabel('# of recovered cases')
plt.title('# europe recovered cases')
plt.show()

plt.plot(datewise_uk['Deaths'])
plt.plot(datewise_spain['Deaths'])
plt.plot(datewise_germany['Deaths'])
plt.plot(datewise_france['Deaths'])
plt.plot(datewise_italy['Deaths'])
plt.legend(['UK', 'Spain', 'Germany', 'France', 'Italy'])
plt.xlabel('days')
plt.ylabel('# of death cases')
plt.title('# europe death cases')
plt.show()
#plot asia
plt.rcParams["figure.figsize"] = [16,9]
plt.plot(datewise_uae['Confirmed'])
plt.plot(datewise_saudi['Confirmed'])
plt.plot(datewise_iran['Confirmed'])
plt.plot(datewise_mainCH['Confirmed'])
plt.plot(datewise_india['Confirmed'])
plt.legend(['UAE', 'Saudi', 'iran', 'china', 'India'])
plt.title('# asia confimed')
plt.xlabel('days')
plt.ylabel('# of confirmed cases')
plt.show()

plt.plot(datewise_uae['Recovered'])
plt.plot(datewise_saudi['Recovered'])
plt.plot(datewise_iran['Recovered'])
plt.plot(datewise_mainCH['Recovered'])
plt.plot(datewise_india['Recovered'])
plt.legend(['UAE', 'Saudi', 'iran', 'chaina', 'India'])
plt.title('# of asia revovery')
plt.xlabel('days')
plt.ylabel('# of recoverd cases')
plt.show()

plt.plot(datewise_uae['Deaths'])
plt.plot(datewise_saudi['Deaths'])
plt.plot(datewise_iran['Deaths'])
plt.plot(datewise_mainCH['Deaths'])
plt.plot(datewise_india['Deaths'])
plt.legend(['UAE', 'Saudi', 'iran', 'chaina', 'India'], )
plt.title('# asia death cases')
plt.xlabel('days')
plt.ylabel('# of death cases')
plt.show()
#plot america
plt.rcParams["figure.figsize"] = [16,9]
plt.plot(datewise_us['Confirmed']/100)
plt.plot(datewise_canada['Confirmed'])
plt.plot(datewise_mex['Confirmed'])
plt.plot(datewise_aus['Confirmed'])
plt.legend(['US', 'Canada', 'Mexico', 'Austrailia'])
plt.title('# america confimed')
plt.xlabel('days')
plt.ylabel('# of confirmed cases')
plt.show()

plt.plot(datewise_us['Recovered'])
plt.plot(datewise_canada['Recovered'])
plt.plot(datewise_mex['Recovered'])
plt.plot(datewise_aus['Recovered'])
plt.legend(['US', 'Canada', 'Mexico', 'Austrailia'])
plt.title('# of america revovery')
plt.xlabel('days')
plt.ylabel('# of recoverd cases')
plt.show()

plt.plot(datewise_us['Deaths'])
plt.plot(datewise_canada['Deaths'])
plt.plot(datewise_mex['Deaths'])
plt.plot(datewise_aus['Deaths'])
plt.legend(['US', 'Canada', 'Mexico', 'Austrailia'])
plt.title('# america death cases')
plt.xlabel('days')
plt.ylabel('# of death cases')
plt.show()
