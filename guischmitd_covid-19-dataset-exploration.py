# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

%matplotlib inline



import plotly.express as px

import pycountry as pc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')
df.columns = ['state', 'country', 'lat', 'long', 'date', 'confirmed', 'death', 'recovered']

df
df.date = pd.to_datetime(df.date)
history = df.groupby('date')[['confirmed', 'death', 'recovered']].sum()

(history / history.shift(1)).plot()

plt.title('Geometric growth worldwide')

history.plot()

plt.title('Cases and outcomes worldwide')

history.plot()

plt.title('Cases and outcomes worldwide (Log plot)')

plt.yscale('log')
history
last = df.loc[df.date == df.date.max()]

last = last.groupby('country').sum().drop(['lat', 'long'], axis=1)

last['iso'] = last.index

def get_iso_code(country):

    try:

        return pc.countries.search_fuzzy(country)[0].alpha_3

    except:

        print(f'{country} not found')

        return np.NaN



last['country'] = last.index

last.iso = last.iso.map(get_iso_code)

cmap = [[0, "rgb(250, 250, 250)"],

        [0.0001, "rgb(250, 200, 200)"],

        [0.001, "rgb(250, 150, 150)"],

        [0.01, "rgb(250, 100, 100)"],

        [.1, "rgb(250, 50, 50)"],

        [1, "rgb(250, 0, 0)"]

      ]



last['log'] = np.log(last.confirmed)

fig = px.choropleth(last, locations="iso",

                    color="confirmed", # lifeExp is a column of gapminder

                    hover_name="country", # column to add to hover information

                    color_continuous_scale=cmap)

fig.show()
from sklearn.linear_model import LinearRegression



for country in ['Brazil', 'France', 'US', 'China']:

    data = df[df.country == country].groupby('date').sum().drop(['lat', 'long'], axis=1)

    fig = plt.figure()

    confirmed = np.log(data.loc[data.confirmed > 0].confirmed)

    

    confirmed.plot(style='x')

    plt.title(f'Growth rate of confirmed cases in {country}')

    

    lr = LinearRegression()

    X = pd.to_numeric(confirmed.index).values.reshape(-1, 1)

    lr.fit(X, confirmed.values.reshape(-1, 1))

    confirmed = pd.DataFrame(confirmed)

    confirmed['reg'] = lr.predict(X)

    confirmed['reg'].plot()

    

    growth = np.exp(lr.coef_ * (24*3600*1000000000))

    print(f'Growth rate in {country} = {growth[0][0]}')
def get_growth_rate(data):

    n = 3

    since_first_case = data.loc[data.confirmed > 0].copy()

    since_first_case['growth'] = np.NaN

    

    for i in range(since_first_case.shape[0] - n):

        lr = LinearRegression()

        subset = since_first_case.iloc[i:i+n]

        

        confirmed = np.log(subset.confirmed)

        X = pd.to_numeric(confirmed.index).values.reshape(-1, 1)

        y = confirmed.values.reshape(-1, 1)

        

        lr.fit(X, y)

        growth = np.exp(lr.coef_ * (24*3600*1000000000))

        since_first_case['growth'].iloc[i] = growth

        print(f'Growth rate = {growth[0][0]}')

        

    return since_first_case

    

data = df[df.country == 'Brazil'].groupby('date').sum().drop(['lat', 'long'], axis=1)

growth_data = get_growth_rate(data)

growth_data
country_dict = {'Brazil':'g', 'Italy':'b', 'China':'r', 'Spain':'orange'}

for country, style in country_dict.items():

    data = df[df.country == country].groupby('date').sum().drop(['lat', 'long'], axis=1)

    data.loc[data.confirmed > 0].confirmed.diff().plot(style=style)

    plt.yscale('log')

    plt.legend(country_dict)
it = df[df.country == 'Italy'].groupby('date').sum().drop(['lat', 'long'], axis=1)

it.confirmed.plot()

br = df[df.country == 'Brazil'].groupby('date').sum().drop(['lat', 'long'], axis=1)

br.confirmed.plot()

plt.yscale('log')