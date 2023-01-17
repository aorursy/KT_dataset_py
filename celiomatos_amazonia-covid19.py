# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import json, requests

import datetime

import folium as fl



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sns.set_style("whitegrid")

sns.set_context("poster")
page = 1

next_page = True

results = []

while(next_page):

    response = requests.get(f'https://brasil.io/api/dataset/covid19/caso/data/?page={page}&state=AM')

    if response.status_code != 200:

        print('Nao conectou')

        next_page = False

    else:    

        dados = json.loads(response.content)        

        results = results + (dados['results'])

        next_page = dados['next'] != None

        page = page + 1



amazonia_covid19 = pd.DataFrame(results)

print(amazonia_covid19.shape)

amazonia_covid19.head()
amazonia_covid19 = amazonia_covid19.query('state == "AM" & place_type == "city"')

print(amazonia_covid19['date'].min())

print(amazonia_covid19['date'].max())
df = amazonia_covid19.query('is_last & deaths > 0').sort_values(by='deaths', ascending=False)[['city','confirmed','deaths']]

df['rate'] = df['deaths'] / df['confirmed'] * 100

df['rate'] = df['rate'].apply(lambda x: '{:.2f}%'.format(x))

print('Total de munícipios afetados: {:d}'.format(df.shape[0]))

df
plt.figure(figsize=(18,15))

plt.yticks(size=8)

plt.xticks(size=12)

plot = sns.barplot(y='city', x='deaths', data=df, orient='h', order=df.sort_values(by='deaths', ascending=False)['city'])

plot.set_title('COVID19 IN STATE OF AMAZONAS', size=50)

plot.set_xlabel('deaths')

plot.set_ylabel('cities')



for p in plot.patches:

    x = p.get_x() + p.get_width() + 0.2

    y = p.get_y() + p.get_height()

    value = int(p.get_width())

    plot.text(x, y, value, ha='left', size=10)
df_without_capital = df.query("city != 'Manaus'")

plt.figure(figsize=(18,15))

plt.yticks(size=8)

plt.xticks(size=12)

plot = sns.barplot(y='city', x='deaths', data=df_without_capital, orient='h', order=df_without_capital.sort_values(by='deaths', ascending=False)['city'])

plot.set_title('COVID19 IN THE COUNTRY OF AMAZONAS', size=50)

plot.set_xlabel('deaths')

plot.set_ylabel('cities')



for p in plot.patches:

    x = p.get_x() + p.get_width() + 0.2

    y = p.get_y() + p.get_height()

    value = int(p.get_width())

    plot.text(x, y, value, ha='left', size=10)
df = amazonia_covid19.groupby('date')['date','confirmed', 'deaths'].sum().reset_index()

df['rate'] = df['deaths'] / df['confirmed'] * 100

df['rate'] = df['rate'].apply(lambda x: '{:.2f}%'.format(x))

df.tail()
plt.figure(figsize=(25,8))

plt.xticks(rotation=90, size=8)

plt.yticks(size=12)



plot = sns.lineplot(x='date', y='deaths', data=df, ci=None, marker='o')



plot = sns.lineplot(x='date', y='confirmed', data=df, ci=None, marker='>')



plot.set_title('COVID19 IN STATE OF AMAZONAS\n', size=50)

plot.set_xlabel('days')

plot.set_ylabel('cases')

plt.legend(['Deaths','Confirmed'])



# confirmed labels

for i in range(df.shape[0]):

    day=datetime.datetime.strptime(df['date'].iloc[i], '%Y-%m-%d').weekday()

    if i == 0 or day==0:

        plt.text(df['date'].iloc[i], df['confirmed'].iloc[i] + 200, df['confirmed'].iloc[i], size=12)

    elif (i+1) == df.shape[0]:

        plt.text(df['date'].iloc[i], df['confirmed'].iloc[i] + 200, df['confirmed'].iloc[i])



# deaths labels        

for i in range(df.shape[0]):

    day=datetime.datetime.strptime(df['date'].iloc[i], '%Y-%m-%d').weekday()

    if df['deaths'].iloc[i] > 0 and (day==0 or (i+1) == df.shape[0]):

        plt.text(df['date'].iloc[i], df['deaths'].iloc[i] + 100, df['deaths'].iloc[i], size=12)        
plt.figure(figsize=(25,8))

plt.xticks(rotation=90, size=8)

plt.yticks(size=12)



plot = sns.lineplot(x='date', y='deaths', data=df, ci=None, marker='o')



plot.set_title('COVID19 IN STATE OF AMAZONAS\n', size=50)

plot.set_xlabel('days')

plot.set_ylabel('cases')

plt.legend(['Deaths'])



# deaths labels        

for i in range(df.shape[0]):

    day=datetime.datetime.strptime(df['date'].iloc[i], '%Y-%m-%d').weekday()

    if df['deaths'].iloc[i] > 0 and (day==0 or (i+1) == df.shape[0]):

        plt.text(df['date'].iloc[i], df['deaths'].iloc[i] + 100, df['deaths'].iloc[i], size=12) 
cases_capital = amazonia_covid19.query('city == "Manaus" & is_last')[['confirmed','deaths']]

cases_country = amazonia_covid19.query('city != "Manaus" & is_last')[['confirmed','deaths']].sum()



df_country_x_capital = pd.DataFrame({'host':['capital','country'], 

                                     'confirmed':[cases_capital['confirmed'].iloc[0], cases_country[0]], 

                                     'deaths':[cases_capital['deaths'].iloc[0], cases_country[1]]})

df_country_x_capital.head()
plot = df_country_x_capital.plot(kind='bar', x='host', rot=0, figsize=(15,7), title='Deaths x Confirmed', fontsize=12)

for p in plot.patches:

    x = p.get_x() + p.get_width() / 2

    y = p.get_y() + p.get_height() + 100

    value = int(p.get_height())

    plot.text(x, y, value, ha='center', size=12)
df['deaths_daily'] = 0

for i in range(df.shape[0]):

    if i > 0: 

        df.loc[i, 'deaths_daily'] = df['deaths'].iloc[i] - df['deaths'].iloc[i-1]

        df.loc[i, 'confirmed_daily'] = df['confirmed'].iloc[i] - df['confirmed'].iloc[i-1]

    else:

        df.loc[i, 'deaths_daily'] = df['deaths'].iloc[i]

        df.loc[i, 'confirmed_daily'] = df['confirmed'].iloc[i]

df.tail()
plt.figure(figsize=(25,8))

plt.xticks(rotation=90, size=8)

plt.yticks(size=12)

plot = sns.barplot(x='date', y='deaths_daily', data=df)

plot.set_title('Deaths per Day')

plot.set_ylabel('deaths')



for p in plot.patches:

    x = p.get_x() + p.get_width() / 2

    y = p.get_y() + p.get_height()

    value = int(p.get_height())

    plot.text(x, y, value, ha='center', size=12)
plt.figure(figsize=(25,8))

plt.xticks(rotation=90, size=8)

plt.yticks(size=12)

plot = sns.barplot(x='date', y='confirmed_daily', data=df)

plot.set_title('Confirmed per Day')

plot.set_ylabel('confirmed')



for p in plot.patches:

    x = p.get_x() + p.get_width() / 2

    y = p.get_y() + p.get_height()

    value = int(p.get_height())

    plot.text(x, y, value, ha='center', size=12)
print('2020-07-19 deaths',amazonia_covid19.query("date == '2020-07-19'")['deaths'].sum())

print('2020-07-20 deaths',amazonia_covid19.query("date == '2020-07-20'")['deaths'].sum())
plt.figure(figsize=(25,8))

plt.xticks(rotation=90, size=8)

ax = sns.lineplot(x='date', y='deaths_daily', data=df)

ax.axes.xaxis.set_visible(False)