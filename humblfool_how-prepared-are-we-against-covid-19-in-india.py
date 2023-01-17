# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb

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



# Any results you write to the current directory are saved as output.
df_india = pd.read_csv('/kaggle/input/coronavirus-cases-in-india/Covid cases in India.csv')

dbd_india = pd.read_excel('/kaggle/input/coronavirus-cases-in-india/per_day_cases.xlsx',sheet_name='India')

dbd_italy = pd.read_excel('/kaggle/input/coronavirus-cases-in-india/per_day_cases.xlsx',sheet_name="Italy")

dbd_korea = pd.read_excel('/kaggle/input/coronavirus-cases-in-india/per_day_cases.xlsx',sheet_name="Korea")
df_india
df_india.describe()
df_india.drop(['S. No.'],axis=1,inplace=True)

df_india.rename(columns={"Name of State / UT": "States"},inplace=True)

dbd_india=dbd_india.fillna(0)

dbd_korea=dbd_korea.fillna(0)

dbd_italy=dbd_italy.fillna(0)

df_india['Total_cases'] = df_india['Total Confirmed cases (Indian National)']+df_india['Total Confirmed cases ( Foreign National )']

df_india['Active cases'] = df_india['Total_cases'] - (df_india['Cured/Discharged/Migrated'] + df_india['Deaths'])

#df_india.drop('total_cases',axis=1,inplace=True)
df_india
print(f'Total number of Confirmed COVID 2019 cases across India:', df_india['Total_cases'].sum())

print(f'Total number of Active COVID 2019 cases across India:', df_india['Total_cases'].sum())

print(f'Total number of Cured/Discharged/Migrated COVID 2019 cases across India:', df_india['Cured/Discharged/Migrated'].sum())

print(f'Total number of Deaths due to COVID 2019  across India:', df_india['Deaths'].sum())

print(f'Total number of States/UTs affected:', df_india['States'].count())
plt.figure(figsize=[9,9])

sb.set(style='darkgrid',font_scale=1.4)

sb.barplot(df_india['States'],df_india['Total_cases'])

plt.xticks(rotation=90)

plt.xlabel('Name of State')

plt.ylabel('Number of cases')

plt.title('Corona Pandemic')

plt.show()
sb.set(style='darkgrid',font_scale=1.2)

a = df_india.groupby(['States'])['Active cases'].sum().sort_values(ascending = False).to_frame()

a.style.background_gradient(cmap='Reds')

indian = df_india['Total Confirmed cases (Indian National)'].sum()

foreign = df_india['Total Confirmed cases ( Foreign National )'].sum()

dict ={"Indian":indian,"Foreigners":foreign}

plt.figure(figsize = (10,10))

plt.pie(dict.values(),labels=dict.keys(), autopct='%1.1f%%')

plt.axis('equal')

plt.show()


def autopct_format(values):

    def my_format(pct):

        total = sum(values)

        val = int(round(pct*total/100.0))

        return '{v:d}'.format(v=val)

    return my_format



values = autopct_format(dict.values())





plt.figure(figsize = (10,10))

plt.pie(dict.values(),labels=dict.keys(), autopct=values)

plt.axis('equal')

plt.show()
sb.set(style='darkgrid',font_scale=1.2)

ax1 = df_india.plot(x="States", y="Total Confirmed cases (Indian National)", kind="bar",color='green')

ax1 = df_india.plot(x="States", y="Total Confirmed cases ( Foreign National )", kind="bar", ax=ax1 , color="red")

plt.xlabel('Name of State')

plt.ylabel('Number of Total Cases Reported')

plt.title('Comparative distribution of Pandemic COVID-19 Cases reported from Indian and Foreign Nationals')

plt.show()
ax = df_india.plot(x="States", y="Total_cases", kind="bar",color='pink')

df_india.plot(x="States", y="Deaths", kind="bar", ax=ax, color="red")

df_india.plot(x="States", y="Cured/Discharged/Migrated", kind="bar", ax=ax, color="green")

plt.xlabel('Name of State')

plt.ylabel('Distribution of COVID-19 Cases')

plt.title('Comparative distribution of Pandemic COVID-19 Cases Reported In India')

plt.show()
df_india.sort_values(by=['States', 'Total_cases'], ascending = [True, False],inplace=True)

df_india.drop_duplicates(subset='States', keep="first",inplace=True)

df_india.sort_values(by='Total_cases', ascending = False,inplace=True)
plt.figure(figsize=[18,18])

sb.set(style='darkgrid',font_scale=1.5)

ax = sb.barplot(df_india['Active cases'].sort_values()[::-1],df_india.loc[df_india['Active cases'].sort_values()[::-1].index]['States'])

ax.set(xlabel='Total Number of Active Cases in India',ylabel='States')

plt.title('Total Active Cases of Corona Pandemic')

plt.show()
plt.figure(figsize=[20,18])

sb.set(style='darkgrid',font_scale=2)



plt.title("Rise of New cases reported of COVID-19 Pandemic in India")

ax = sb.lineplot(x="Date", y="New Cases", data=dbd_india,markers=True, dashes=False,label='Active')

ax.set(xlabel='Date',ylabel='Rise in New Cases')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=[20,18])

sb.set(style='darkgrid',font_scale=2)



plt.title("Distribution of Active Cases, Recovered Cases, Deaths from COVID-19 Pandemic in India")

ax = sb.lineplot(x="Date", y="Active", data=dbd_india,markers=True, dashes=False,label='Active')

ax = sb.lineplot(x="Date", y="Recovered", data=dbd_india,color='green',label='Recovered')

ax = sb.lineplot(x="Date", y="Deaths", data=dbd_india,color='red',label='Deaths')

ax.set(xlabel='Date',ylabel='Distribution of Cases')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=[19,14])

sb.set(style='darkgrid',font_scale=2)



plt.title("Comparative distribution of rise of total cases reported of COVID-19 from Italy, Korea , India")

ax1 = sb.lineplot(x="Date", y="Total Cases", data=dbd_india,markers=True, dashes=False,label='India')

ax1 = sb.lineplot(x="Date", y="Total Cases", data=dbd_korea,color='orange',label='korea')

ax1 = sb.lineplot(x="Date", y="Total Cases", data=dbd_italy,color='red',label='Italy')

ax1.set(xlabel='Date',ylabel='Number of Total Cases Reported')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=[22,14])

sb.set(style='darkgrid',font_scale=2)



plt.xticks(rotation=90)

plt.title("Comparative distribution of rise of New Cases reported of COVID-19 from Italy, Korea , India")

ax = sb.lineplot(x="Date", y="New Cases", data=dbd_india,markers=True, dashes=False,label='India')

ax = sb.lineplot(x="Date", y="New Cases", data=dbd_korea,color='orange',label='korea')

ax = sb.lineplot(x="Date", y="New Cases", data=dbd_italy,color='red',label='Italy')

ax.set(xlabel='Date',ylabel='Number of New Cases Reported')

plt.show()
plt.figure(figsize=[34,28])



plt.subplot(2,2,1)

ax1 = sb.lineplot(x="Date", y="Total Cases", data=dbd_india[dbd_india['Days after surpassing 100 cases']==0],markers=True, dashes=False,label='Rise in Total cases in India Before surpassing 100 cases Benchmark',color='green')

ax1.set(xlabel='Date',ylabel='Number of Total Cases Reported')

plt.xticks(rotation=90)



plt.subplot(2,2,2)

ax2= sb.lineplot(x="Date", y="Total Cases", data=dbd_india[dbd_india['Days after surpassing 100 cases']>0],markers=True, dashes=False,label='Rise in Total cases in India After surpassing 100 cases Benchmark',color='red')

ax2.set(xlabel='Date',ylabel='Number of Total Cases Reported')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=[34,28])





plt.subplot(2,2,1)

ax1 = sb.lineplot(x="Date", y="Total Cases", data=dbd_korea[dbd_korea['Days after surpassing 100 cases']==0],markers=True, dashes=False,label='Rise in Total cases in Korea Before surpassing 100 cases Benchmark',color='green')

ax1.set(xlabel='Date',ylabel='Number of Total Cases Reported ')

plt.xticks(rotation=90)



plt.subplot(2,2,2)

ax2= sb.lineplot(x="Date", y="Total Cases", data=dbd_korea[dbd_korea['Days after surpassing 100 cases']>0],markers=True, dashes=False,label='Rise in Total cases in korea After surpassing 100 cases Benchmark',color='red')

ax2.set(xlabel='Date',ylabel='Number of Total Cases Reported')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=[34,28])



plt.subplot(2,2,1)

ax1 = sb.lineplot(x="Date", y="Total Cases", data=dbd_italy[dbd_italy['Days after surpassing 100 cases']==0],markers=True, dashes=False,label='Rise in Total cases in Italy Before surpassing 100 cases Benchmark',color='green')

ax1.set(xlabel='Date',ylabel='Number of Total Cases Reported ')

plt.xticks(rotation=90)



plt.subplot(2,2,2)

ax2= sb.lineplot(x="Date", y="Total Cases", data=dbd_italy[dbd_italy['Days after surpassing 100 cases']>0],markers=True, dashes=False,label='Rise in Total cases in Italy After surpassing 100 cases Benchmark',color='red')

ax2.set(xlabel='Date',ylabel='Number of Total Cases Reported')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=[34,28])



plt.title("Distribution of Rise in Total cases of COVID-19 pandemic in India, Before and After surpassing 100 cases Benchmark'")

sb.set(style='darkgrid',font_scale=4)



ax= sb.lineplot(x="Date", y="Total Cases", data=dbd_india[dbd_india['Days after surpassing 100 cases']==0],markers=True, dashes=False,label='Rise in Total cases in india Before surpassing 100 cases Benchmark',color='green')

ax.set(xlabel='Date',ylabel='Number of Total Cases Reported ')

ax= sb.lineplot(x="Date", y="Total Cases", data=dbd_india[dbd_india['Days after surpassing 100 cases']>0],markers=True, dashes=False,label='Rise in Total cases in india After surpassing 100 cases Benchmark',color='red')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=[34,28])



plt.title("Distribution of Rise in Total cases of COVID-19 pandemic in Korea, Before and After surpassing 100 cases Benchmark'")

sb.set(style='darkgrid',font_scale=4)



ax = sb.lineplot(x="Date", y="Total Cases", data=dbd_korea[dbd_korea['Days after surpassing 100 cases']==0],markers=True, dashes=False,label='Rise in Total cases in korea Before surpassing 100 cases Benchmark',color='green')

ax.set(xlabel='Date',ylabel='Number of Total Cases Reported ')

ax= sb.lineplot(x="Date", y="Total Cases", data=dbd_korea[dbd_korea['Days after surpassing 100 cases']>0],markers=True, dashes=False,label='Rise in Total cases in korea After surpassing 100 cases Benchmark',color='red')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=[34,28])



plt.title("Distribution of Rise in Total cases of COVID-19 pandemic in italy, Before and After surpassing 100 cases Benchmark'")

sb.set(style='darkgrid',font_scale=4)



ax = sb.lineplot(x="Date", y="Total Cases", data=dbd_italy[dbd_italy['Days after surpassing 100 cases']==0],markers=True, dashes=False,label='Rise in Total cases in Italy Before surpassing 100 cases Benchmark',color='green')

ax= sb.lineplot(x="Date", y="Total Cases", data=dbd_italy[dbd_italy['Days after surpassing 100 cases']>0],markers=True, dashes=False,label='Rise in Total cases in Italy After surpassing 100 cases Benchmark',color='red')

ax.set(xlabel='Date',ylabel='Number of Total Cases Reported')

plt.xticks(rotation=90)

plt.show()