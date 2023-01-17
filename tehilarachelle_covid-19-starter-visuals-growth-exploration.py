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
import pandas as pd

COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
pd.options.display.max_columns=999

import matplotlib.pyplot as plt

%matplotlib inline
pop = pd.read_csv("../input/populationcountries/population.csv")
#list of unique countries/regions

country = covid_19_data['Country/Region'].unique()
covid_19_data.head()
#convert date to datetime format

covid_19_data['ObservationDate'] = pd.to_datetime(covid_19_data['ObservationDate']).dt.date
date = covid_19_data['ObservationDate'].unique()

print('days counted:', len(date),'\n' '1st date:',date.min(),'\n''last date:',date.max())
#sum country totals for last date in dataset.

df = pd.DataFrame()

confirmed_sum=[]

country_name=[]

for c in country:

    data=covid_19_data[(covid_19_data['Country/Region']==c) & (covid_19_data['ObservationDate'] == date.max())]

    country_name.append(c)

    confirmed_sum.append(data['Confirmed'].sum())

    

df['country_name']=country_name

df['confirmed_sum']=confirmed_sum   
print('Top 10 countries confirmed cases as of {}'.format(date.max()))

print(df.sort_values(by=['confirmed_sum'], ascending=False).head(10))
top_ten=df.sort_values(by=['confirmed_sum'], ascending=False).head(10)

top_ten_countries=list(top_ten.country_name.values)
#create a function to dreate df with confirmed sum, % change, and daynum (the number of days recording)

def country_growth(country):

    data = covid_19_data.copy()

    df = pd.DataFrame()

    conf_sum=[]

    date_sum=[]

#    pct_change=[]

    for d in range(0,len(date)):

        date_sum.append(date[d])

        conf_sum.append(data[(data['ObservationDate']==date[d]) &(data['Country/Region']==country)]['Confirmed'].sum())

    df['date']=date_sum

    df['confirmed_sum']=conf_sum

    df['pct_change']=round(df.confirmed_sum.pct_change()*100)

    df['daynum']=range(0,len(date))

    return df
us_growth_rate = country_growth('US')

len(us_growth_rate['date'])

us_growth_rate.head()
us_growth_rate.plot.scatter(x='daynum',y='pct_change')

plt.xticks(np.arange(0, len(date),step=3))

day_min = 30

day_max = us_growth_rate['daynum'].max()

sum_min = us_growth_rate.loc[us_growth_rate['daynum'] == day_min, 'confirmed_sum'].values[0]

sum_max = us_growth_rate.loc[us_growth_rate['daynum'] == day_max, 'confirmed_sum'].values[0]

pct_tot = round((sum_max/sum_min)-1)*100

plt.xlim(day_min, len(date))

plt.title('US % Change in Confirmed Cases by Day')

x=int(us_growth_rate['confirmed_sum'][len(date)-1])

plt.annotate('Total Confirmed: {}'.format(x),xy=(50,100),xytext=(32,100))

plt.annotate('% Increase since day 30: {0:.2f}%'.format(pct_tot),xy=(50,100),xytext=(32,80))

plt.show()
df_label=top_ten_countries



fig = plt.figure(figsize=(15,18))

for l in range(0,len(df_label)):

    df=country_growth(df_label[l])

    ax = fig.add_subplot(4,3,l+1)

    ax.scatter(df['daynum'], df['pct_change'])

    ax.set_title(df_label[l])

    ax.set_ylabel('% Change in Confirmed Cases')

    ax.set_xlim(0, len(date),4)
pop_country=pd.DataFrame(pop)

pop_country=pop_country.drop('Unnamed: 0', axis=1)

pop_country
# pop_country.loc[pop_country['name'] == 'France', 'population'].values[0]
#modify country names in the population data set to match names in covid dataset

pop_country=pop_country.replace(['China','United States','Korea, South','United Kingdom'],['Mainland China', 'US','South Korea','UK'])
pop_country
df_label=top_ten_countries



fig = plt.figure(figsize=(15,18))

for l in range(0,len(df_label)):

    df=country_growth(df_label[l])

    ax = fig.add_subplot(4,3,l+1)

    ax.scatter(df['daynum'], df['confirmed_sum'])

    x=int(df['confirmed_sum'][len(date)-1])

    d_max= df['date'].max()

    d_min= df['date'].min()

    pc=pop_country.loc[pop_country['name'] == df_label[l], 'population'].values[0]

    pct=(x/pc)*100

    ax.text(0.5, 0.55, 'Total Confirmed: {}'.format(x), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

    ax.text(0.5, 0.45, 'Total Population: {}'.format(pc), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

    ax.text(0.5, 0.35, '{0:.2f}% of population'.format(pct), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

    ax.text(0.5, 0.65, 'Latest Date: {}'.format(d_max), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

    ax.text(0.5, 0.75, 'Start Date: {}'.format(d_min), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

    

    ax.set_title(df_label[l])

    ax.set_ylabel('Confirmed Sum')

    ax.set_xlabel('Day Number')

    ax.set_xlim(0, len(date),4)

plt.subplots_adjust(wspace=0.4, hspace=0.3)

plt.savefig('top 10.pdf')
df_label=top_ten_countries



fig = plt.figure(figsize=(15,18))

for l in range(0,len(df_label)):

    df=country_growth(df_label[l])

    df_day30=df[df['daynum']>=30]

    ax = fig.add_subplot(4,3,l+1)

    ax.scatter(df_day30['daynum'], df_day30['confirmed_sum'],color='red')

    x=int(df_day30['confirmed_sum'][len(date)-1])

    d_max = df_day30['date'].max()

    d_min = df_day30['date'].min()

    pc=pop_country.loc[pop_country['name'] == df_label[l], 'population'].values[0]

    pct=(x/pc)*100

    ax.text(0.5, 0.55, 'Total Confirmed: {}'.format(x), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

    ax.text(0.5, 0.45, 'Total Population: {}'.format(pc), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

    ax.text(0.5, 0.35, '{0:.2f}% of population'.format(pct), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

    ax.text(0.5, 0.65, 'Latest Date: {}'.format(d_max), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

    ax.text(0.5, 0.75, 'Start Date: {}'.format(d_min), horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)

    

    ax.set_title(df_label[l])

    ax.set_ylabel('Confirmed Sum')

    ax.set_xlabel('Day Number')

    ax.set_xlim(30, len(date),4)

plt.subplots_adjust(wspace=0.4, hspace=0.3)

plt.savefig('top 10 last 30.pdf')