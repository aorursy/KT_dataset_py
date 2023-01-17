# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import os

plt.style.use("fivethirtyeight")# for pretty graphs



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

def configure_plotly_browser_state():

    import IPython

    display(IPython.core.display.HTML('''

              <script src= "/static/components/requirejs/require.js"></script>

              <script>

                requirejs.config({

                  paths:{

                    base: '/static/base',

                    plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',

                  },

                });

              </script> 

              '''))
corona_world = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

corona_States_india  = pd.read_csv('/kaggle/input/coronavirus-cases-in-india/Covid cases in India.csv')

corona_italy = pd.read_csv('/kaggle/input/covid19-in-italy/covid19_italy_region.csv')

corona_usa = pd.read_csv('/kaggle/input/covid19-in-usa/us_states_covid19_daily.csv')

bed = pd.read_csv('/kaggle/input/beds-per-1000-person-in-a-specific-country/DP_LIVE_22032020165806740.csv')

per_day_india = pd.read_excel('/kaggle/input/coronavirus-cases-in-india/per_day_cases.xlsx',sheet_name='India')
# from pandas_profiling import ProfileReport

# prof= ProfileReport(corona_world)

# prof
corona_States_india['Total_Cases'] = corona_States_india['Total Confirmed cases ( Foreign National )']+ corona_States_india['Total Confirmed cases (Indian National)']

corona_States_india['Active_Cases'] = corona_States_india['Total_Cases']- corona_States_india['Cured/Discharged/Migrated']-corona_States_india['Deaths']
corona_States_india.rename(columns={'Name of State / UT':'States'},inplace=True)
corona_States_india.sort_values(by='Active_Cases',inplace=True,ascending=False)
corona_States_india.reset_index(inplace=True)

corona_States_india.drop(['S. No.','index'],axis=1,inplace=True)
corona_States_india.head(2)
plt.figure(figsize=[10,10]);

sns.barplot(corona_States_india['Active_Cases'],corona_States_india['States'])

plt.title('State-wise Number of Active Cases',fontsize=15,loc='center')

plt.xticks(rotation=90);
corona_States_india.groupby(['States'])['Active_Cases'].sum().sort_values( ascending=False).to_frame()
plt.figure(figsize=(12,8))

sns.set(style='darkgrid',font_scale=2)



plt.bar(corona_States_india['States'],corona_States_india['Active_Cases'],color='blue')

plt.bar(corona_States_india['States'],corona_States_india['Deaths'],color='red')

plt.bar(corona_States_india['States'],corona_States_india['Cured/Discharged/Migrated'],color='green')



plt.legend(labels =['Active_Cases','Deaths','Cured'])

plt.title('State-Wise Cases in India',fontsize=25)

plt.xlabel('Indian States')

plt.ylabel('Number of cases')

plt.xticks(rotation = 90);
corona_world.head()
corona_world['ObservationDate']=pd.to_datetime(corona_world['ObservationDate'])

corona_world['Active_Cases'] = corona_world['Confirmed']-corona_world['Deaths']-corona_world['Recovered']
sns.set(style='darkgrid',font_scale=2)

plt.figure(figsize=(15,8))



corona_world.groupby(['ObservationDate'])['Active_Cases'].sum().plot(kind='line')

corona_world.groupby(['ObservationDate'])['Deaths'].sum().plot(kind='line')

corona_world.groupby(['ObservationDate'])['Recovered'].sum().plot(kind='line')



plt.legend(labels=['Active','death','recovered'],bbox_to_anchor=(1.1, 1.1))

plt.title('Recovered Vs Death Cases in World',fontsize=25)

plt.xlabel('Observation Date')

plt.ylabel('Number of cases')

plt.xticks(rotation =90);
corona_china = corona_world[corona_world['Country/Region']=='Mainland China']
corona_china.head()
sns.set(style='darkgrid',font_scale=2)

plt.figure(figsize=(15,8))



corona_china.groupby('ObservationDate')['Active_Cases'].sum().plot(kind='line')

corona_china.groupby('ObservationDate')['Deaths'].sum().plot(kind='line')

corona_china.groupby('ObservationDate')['Recovered'].sum().plot(kind='line')



plt.legend(labels=['Active','death','recovered'],bbox_to_anchor=(1.1, 1.1))

plt.title('Recovered Vs Death Cases in China',fontsize=25)

plt.xlabel('Observation Date')

plt.ylabel('Number of cases')

plt.xticks(rotation =90);
corona_usa['date'] = corona_usa['date'].apply(lambda x:pd.to_datetime(x,format='%Y%m%d', errors='ignore'))
corona_usa['positive']= corona_usa['positive'].fillna(0).astype(int)
corona_usa.head()
sns.set(style='darkgrid',font_scale=2)

plt.figure(figsize=(15,8))

corona_usa.groupby('date')['hospitalized'].sum().plot(kind='line')

corona_usa.groupby('date')['death'].sum().plot(kind='line')



plt.legend(labels=['Active','death'],bbox_to_anchor=(1.2, 1.1))

plt.title('Confirmed Vs Death Cases in USA',fontsize=15)

plt.xlabel('Observation Date')

plt.ylabel('Number of cases')

plt.xticks(rotation =90);
corona_italy['Date'] = corona_italy['Date'].apply(lambda x:pd.to_datetime(x).date())
corona_italy.head(2)
sns.set(style='darkgrid',font_scale=2)

plt.figure(figsize=(15,8))



corona_italy.groupby('Date')['CurrentPositiveCases'].sum().plot(kind='line')

corona_italy.groupby('Date')['Deaths'].sum().plot(kind='line')

corona_italy.groupby('Date')['Recovered'].sum().plot(kind='line')



plt.legend(labels=['Active','death','recovered'],bbox_to_anchor=(1.1, 1.1))

plt.title('Recovered Vs Death Cases in Italy',fontsize=25)

plt.xlabel('Observation Date')

plt.ylabel('Number of cases')

plt.xticks(rotation =90);
per_day_india.drop('Days after surpassing 100 cases',axis=1,inplace=True)
per_day_india.head(2)
sns.set(style='darkgrid',font_scale=2)

plt.figure(figsize=(15,8))





per_day_india.groupby('Date')['Active'].sum().plot(kind='line')

per_day_india.groupby('Date')['Deaths'].sum().plot(kind='line')

per_day_india.groupby('Date')['Recovered'].sum().plot(kind='line')



plt.title('Recovered Vs Death Cases in India',fontsize=25)

plt.legend(labels=['Active','death','recovered'],bbox_to_anchor=(1.1, 1.1))

plt.xlabel('Observation Date')

plt.ylabel('Number of cases')

plt.xticks(rotation =90);
bed.head()
bed=bed[(bed['LOCATION']=='ITA') | (bed['LOCATION']=='USA') | (bed['LOCATION']=='IND') | (bed['LOCATION']=='CHN')]
new_bed=bed[(bed['TIME']==2016)]
bb=new_bed.groupby(['LOCATION','TIME'])[['Value']].max().sort_values(by='Value')

bb=bb.reset_index()

bb
bed.drop(['INDICATOR',  'MEASURE', 'FREQUENCY'],axis=1,inplace=True)
sns.set(style='darkgrid',font_scale=2)

plt.figure(figsize=(15,8))



my_colors = 'rgbm'

b=plt.bar(bb['LOCATION'],bb['Value'],color=my_colors)

for rect in b:

    height = round(rect.get_height(),2)

    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%f' % height, ha='center', va='bottom')

plt.xlabel('Countries')

plt.ylabel('Number of Beds')

plt.title('Number of beds per 1000 persons in 2016',fontsize=25);
sns.set(style='darkgrid',font_scale=2)

plt.figure(figsize=(18,9))



corona_italy.groupby('Date')['CurrentPositiveCases'].sum().plot(kind='line')

corona_usa.groupby('date')['hospitalized'].sum().plot(kind='line',color='red')

corona_china.groupby('ObservationDate')['Active_Cases'].sum().plot(kind='line')

per_day_india.groupby('Date')['Active'].sum().plot(kind='line',color='yellow')



plt.legend(labels=['italy','usa','china','india'],bbox_to_anchor=(1.1, 1.1))

plt.title('Active Cases for the labeled countries',fontsize=25)

plt.xlabel('Observation Date')

plt.ylabel('Number of cases')

plt.xticks(rotation =90);
corona_world.head(2)
world_confirmed_cases = corona_world.groupby('ObservationDate')[['Confirmed']].sum()

world_confirmed_cases=world_confirmed_cases.reset_index()
def count_lakh(corona):

    x=[555]

    y=[corona['ObservationDate'][0]]

    m=1

    for i in range(len(corona)):

        k=corona['Confirmed'][i]

        if k>=100000*m:

            j=corona['ObservationDate'][i]

            x.append(k)

            y.append(j)

            m=m+1

    data = pd.DataFrame(list(zip(y,x)),columns =['Date', 'Count'])         

    return data
count_lakh=count_lakh(world_confirmed_cases)

count_lakh
days_diff=[0]

for i in range(len(count_lakh)-1):

    a=count_lakh['Date'][i+1]-count_lakh['Date'][i]

    days_diff.append(a)

count_lakh['days_took']=days_diff

count_lakh['days_took']=count_lakh['days_took'].astype(str).apply(lambda x:x.split(' ')[0])
count_lakh=count_lakh.groupby(['Date','days_took'])['Count'].sum()
count_lakh=count_lakh.reset_index()
count_lakh
sns.set(style='darkgrid',font_scale=2)

plt.figure(figsize=(15,8))



my_colors = 'rgbm'  #red, green, blue,etc



b=plt.bar(count_lakh['Date'],count_lakh['Count'],tick_label=count_lakh['days_took'],color=my_colors)



for rect in b:

    height = rect.get_height()

    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom',fontsize=15)



plt.xlabel('Difference in number of days')

plt.ylabel('Number of confirmed cases')

plt.title('Number of days it took for increase in every lakh(hundred thousand) cases',fontsize=25);