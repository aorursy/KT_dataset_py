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

data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
deaths.head()
deaths.reset_index()
deaths = deaths.rename(columns =  {'Country/Region' : 'country'})

confirmed = confirmed.rename(columns =  {'Country/Region' : 'country'})

recovered = recovered.rename(columns =  {'Country/Region' : 'country'})
deaths.head()
np.shape(deaths)
deaths.columns
deaths.groupby(['country', 'value']).max().sort_values('value', ascending = False).reset_index()
#deaths = deaths.reset_index()

deaths = pd.melt(deaths, id_vars='country', value_vars=['1/22/20', '1/23/20',

       '1/24/20', '1/25/20', '1/26/20', '1/27/20', '1/28/20', '1/29/20',

       '1/30/20', '1/31/20', '2/1/20', '2/2/20', '2/3/20', '2/4/20', '2/5/20',

       '2/6/20', '2/7/20', '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20',

       '2/13/20', '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20',

       '2/19/20', '2/20/20', '2/21/20', '2/22/20', '2/23/20', '2/24/20',

       '2/25/20', '2/26/20', '2/27/20', '2/28/20', '2/29/20', '3/1/20',

       '3/2/20', '3/3/20', '3/4/20', '3/5/20', '3/6/20', '3/7/20', '3/8/20',

       '3/9/20', '3/10/20', '3/11/20'])
confirmed = pd.melt(confirmed, id_vars='country', value_vars=['1/22/20', '1/23/20',

       '1/24/20', '1/25/20', '1/26/20', '1/27/20', '1/28/20', '1/29/20',

       '1/30/20', '1/31/20', '2/1/20', '2/2/20', '2/3/20', '2/4/20', '2/5/20',

       '2/6/20', '2/7/20', '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20',

       '2/13/20', '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20',

       '2/19/20', '2/20/20', '2/21/20', '2/22/20', '2/23/20', '2/24/20',

       '2/25/20', '2/26/20', '2/27/20', '2/28/20', '2/29/20', '3/1/20',

       '3/2/20', '3/3/20', '3/4/20', '3/5/20', '3/6/20', '3/7/20', '3/8/20',

       '3/9/20', '3/10/20', '3/11/20'])
recovered = pd.melt(recovered, id_vars='country', value_vars=['1/22/20', '1/23/20',

       '1/24/20', '1/25/20', '1/26/20', '1/27/20', '1/28/20', '1/29/20',

       '1/30/20', '1/31/20', '2/1/20', '2/2/20', '2/3/20', '2/4/20', '2/5/20',

       '2/6/20', '2/7/20', '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20',

       '2/13/20', '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20',

       '2/19/20', '2/20/20', '2/21/20', '2/22/20', '2/23/20', '2/24/20',

       '2/25/20', '2/26/20', '2/27/20', '2/28/20', '2/29/20', '3/1/20',

       '3/2/20', '3/3/20', '3/4/20', '3/5/20', '3/6/20', '3/7/20', '3/8/20',

       '3/9/20', '3/10/20', '3/11/20'])
deaths.head()
deaths.describe().T
type(deaths)
print(deaths.shape)

print('\n')

print(confirmed.shape)
deaths['variable'] = pd.to_datetime(deaths['variable'])

confirmed['variable'] = pd.to_datetime(confirmed['variable'])

recovered['variable'] = pd.to_datetime(recovered['variable'])
deaths['variable']
confirmed_china = confirmed[confirmed['country'] == 'China'].groupby('variable')[['country', 'value', 'variable']].sum().reset_index()

#confirmed_china.drop(49, inplace = True)
deaths_china = deaths[deaths['country'] == 'China'].groupby('variable')[['country', 'value', 'variable']].sum().reset_index()

#deaths_china.drop(49, inplace = True)
fig, ax = plt.subplots(1,1,figsize=(15,9))

sns.set_style('darkgrid')

plt.xticks(rotation=35)

sns.lineplot(data = deaths_china, x = 'variable', y ='value', linewidth = 4, color = 'darkred', label = 'Deaths')

sns.lineplot(data = confirmed_china, x = 'variable', y ='value', linewidth = 4, color = 'darkblue', label = 'Confirmed')

plt.title("China's Confirmed cases", size = 18)

plt.ylabel('Deaths', size= 13)

plt.xlabel('Date', size = 13)

plt.legend(loc="upper left")
confirmed_ger = confirmed[confirmed['country'] == 'Germany'].groupby('variable')[['country', 'value', 'variable']].sum().reset_index()

confirmed_ger.drop(49, inplace = True)
deaths_ger = deaths[deaths['country'] == 'Germany'].groupby('variable')[['country', 'value', 'variable']].sum().reset_index()

deaths_ger.drop(49, inplace = True)
fig, ax = plt.subplots(1,1,figsize=(15,9))

sns.set_style('darkgrid')

plt.xticks(rotation=35)

sns.lineplot(data = deaths_ger, x = 'variable', y ='value', linewidth = 4, color = 'darkred', label = 'Deaths')

sns.lineplot(data = confirmed_ger, x = 'variable', y ='value', linewidth = 4, color = 'darkblue', label = 'Confirmed')

plt.title("Germany", size = 18)

plt.ylabel('Deaths', size= 13)

plt.xlabel('Date', size = 13)

plt.legend(loc="upper left")
confirmed_ita = confirmed[confirmed['country'] == 'Italy'].groupby('variable')[['country', 'value', 'variable']].sum().reset_index()

confirmed_ita.drop(49, inplace = True)
deaths_ita = deaths[deaths['country'] == 'Italy'].groupby('variable')[['country', 'value', 'variable']].sum().reset_index()

deaths_ita.drop(49, inplace = True)
fig, ax = plt.subplots(1,1,figsize=(15,9))

sns.set_style('darkgrid')

plt.xticks(rotation=35)

sns.lineplot(data = deaths_ita, x = 'variable', y ='value', linewidth = 4, color = 'darkred', label = 'Deaths')

sns.lineplot(data = confirmed_ita, x = 'variable', y ='value', linewidth = 4, color = 'darkblue', label = 'Confirmed')

plt.title("Italy", size = 18)

plt.ylabel('Deaths', size= 13)

plt.xlabel('Date', size = 13)

plt.legend(loc="upper left")
five_ = confirmed[(confirmed['country'] == 'Italy') | (confirmed['country'] == 'Germany') | (confirmed['country'] == 'Netherlands')].groupby(['variable', 'country'])[['country', 'value', 'variable']].sum().reset_index()

#confirmed_ita.drop(49, inplace = True)
five_
plt.figure(figsize=(10,9))

sns.lineplot(data = five_, x = 'variable', y = 'value', hue = 'country', linewidth = 4)

plt.title('Confirmed Cases')

plt.xticks(rotation=55)
data.head()
data = data.rename(columns = {'Country/Region' : 'country', 'ObservationDate':'Date'})
data.shape
d = data[data['Date'] == max(data['Date'])]

d = d.groupby('country').sum().sort_values('Confirmed', ascending=False)

d = d[d['Deaths']>0]

d.drop('SNo', axis=1, inplace=True)

d.style.background_gradient(cmap='Reds')
import plotly.express as px
d = d.reset_index()


fig = px.treemap(d.sort_values(by='Confirmed', ascending=False).reset_index(drop=True), 

                 path=["country"], values="Confirmed", height=700,

                 title='Number of Confirmed Cases',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value'

fig.show()



fig = px.treemap(d.sort_values(by='Deaths', ascending=False).reset_index(drop=True), 

                 path=["country"], values="Deaths", height=700,

                 title='Number of Deaths reported',

                 color_discrete_sequence = px.colors.qualitative.Prism)

fig.data[0].textinfo = 'label+text+value'

fig.show()

data.head()
data[data['country'] == 'Mainland China'][['country', 'Confirmed', 'Deaths']]
alle = data.groupby('Date')['Confirmed'].max()

alle.reset_index().sort_values('Confirmed', ascending=  False)
comp = data.groupby('country')[['country', 'Confirmed', 'Deaths', 'Recovered']].max().sort_values('Confirmed', ascending = False).head(5)

comp
comp1 = pd.melt(comp, id_vars = 'country', value_vars = ['Confirmed', 'Deaths', 'Recovered'])

comp1
fig, ax = plt.subplots(1,1,figsize=(15,9))

sns.barplot(data = comp1, x = 'variable', y = 'value', hue='country')
data.head()
per_coun = data.groupby(['country', 'Date'])[['country', 'Date', 'Confirmed', 'Deaths']].max().reset_index()

per_coun
ger_ne = per_coun[(per_coun['country'] == 'Netherlands') | (per_coun['country'] == 'Germany')]

ger_ne
fig, ax = plt.subplots(1,1,figsize=(15,9))

plt.xticks(rotation=45)

sns.lineplot(data = ger_ne, x = 'Date', y = 'Confirmed', hue='country')
data.sort_values('Deaths', ascending = False).head(30)
! pip install calmap
import calmap 
data.head()


import pandas as pd

COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")