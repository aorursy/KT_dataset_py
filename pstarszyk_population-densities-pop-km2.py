import requests, pandas as pd, numpy as np

from bs4 import BeautifulSoup
train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

country_state = train[['Province_State', 'Country_Region']].drop_duplicates().reset_index(drop = True)

del train

country_state.head()
country_state[~country_state.Province_State.isna()].Country_Region.drop_duplicates()
countries = list(country_state.Country_Region.drop_duplicates())
url = 'https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population_density'

html_text = requests.get(url).text

soup = BeautifulSoup(html_text, 'html.parser').findAll('td')
cols = ['country', 'state', 'density'] # people / km2

df_WD = pd.DataFrame(columns = cols)
ix = 0

for ii in range(len(soup)):

    try:

        country = soup[ii].findAll('a')[0].text

    except:

        country = 'none'

    

    if country in countries:

        df_WD.loc[ix, 'country'] = country

        df_WD.loc[ix, 'density'] = soup[ii + 4].text[:-1]

        ix += 1



del soup

df_WD = df_WD.reset_index(drop = True)
df_WD.density = df_WD.density.str.replace(",","").astype(float)

df_WD.head()
states = list(country_state[country_state.Country_Region=='US'].Province_State)
url = 'https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States_by_population_density'

html_text = requests.get(url).text

soup = BeautifulSoup(html_text, 'html.parser').findAll('td')
cols = ['country', 'state', 'density'] # people / km2

df_US = pd.DataFrame(columns = cols)
ix = 0

for ii in range(len(soup)):

    try:

        state = soup[ii].findAll('a')[0].text

    except:

        state = 'none'

    

    if state in states:

        df_US.loc[ix, 'state'] = state

        df_US.loc[ix, 'density'] = soup[ii + 4].text[:-1]

        ix += 1



del soup

df_US.country = 'US'

df_US = df_US.reset_index(drop = True)

df_US = df_US.iloc[0:53]

df_US.density = df_US.density.astype(float)

df_US.head()
states = list(country_state[country_state.Country_Region=='Australia'].Province_State)
url = 'https://en.wikipedia.org/wiki/States_and_territories_of_Australia'

html_text = requests.get(url).text

soup = BeautifulSoup(html_text, 'html.parser').findAll('td')
cols = ['country', 'state', 'population', 'area']

df_AU = pd.DataFrame(columns = cols)
ix = 0

for ii in range(len(soup)):

    try:

        state = soup[ii].text

    except:

        state = 'none'

    

    if state in states:

        shift = (state in ['Australian Capital Territory', 'Northern Territory'])

        df_AU.loc[ix, 'state'] = state

        df_AU.loc[ix, 'population'] = soup[ii + 4 + shift].text

        df_AU.loc[ix, 'area'] = soup[ii + 5 + shift].text[:-1]

        ix += 1



del soup

df_AU.country = 'Australia'

df_AU = df_AU.reset_index(drop = True)
df_AU.population = df_AU.population.str.replace(",","").astype(float)

df_AU.area = df_AU.area.str.replace(",","").astype(float)

df_AU['density'] = df_AU.population / df_AU.area

df_AU = df_AU.drop(columns = ['population', 'area'])
df_AU.head()
states = list(country_state[country_state.Country_Region=='Canada'].Province_State)
url = 'https://en.wikipedia.org/wiki/Population_of_Canada_by_province_and_territory'

html_text = requests.get(url).text

soup = BeautifulSoup(html_text, 'html.parser').findAll('td')
cols = ['country', 'state', 'density']

df_CA = pd.DataFrame(columns = cols)
ix = 0

for ii in range(len(soup)):

    try:

        state = soup[ii].findAll('a')[0].text

    except:

        state = 'none'

    

    if state in states:

        df_CA.loc[ix, 'state'] = state

        df_CA.loc[ix, 'density'] = soup[ii + 5].text[:-1]

        ix += 1



del soup

df_CA.country = 'Canada'

df_CA = df_CA.reset_index(drop = True)
df_CA = df_CA.iloc[0:10]

df_CA.density = df_CA.density.str.replace(",","").astype(float)

df_CA.head()
states = list(country_state[country_state.Country_Region=='China'].Province_State)
url = 'https://en.wikipedia.org/wiki/Provinces_of_China'

html_text = requests.get(url).text

soup = BeautifulSoup(html_text, 'html.parser').findAll('td')
cols = ['country', 'state', 'density']

df_CH = pd.DataFrame(columns = cols)
suffix = ['Province', 'Municipality', 'Autonomous Region', 'Administrative Region']



ix = 0



for ii in range(len(soup)):

    try:

        state = soup[ii].findAll('a')[0].text

        if any(x in state for x in suffix):

            if 'Hong Kong' in state:

                state = 'Hong Kong'

            elif 'Inner Mongolia' in state:

                state = 'Inner Mongolia'

            else:

                state = state[0:state.index(' ')]

    except:

        state = 'none'

    

    if state in states:

        df_CH.loc[ix, 'state'] = state

        df_CH.loc[ix, 'density'] = soup[ii + 4].text[:-1]

        ix += 1



del soup

df_CH.country = 'China'

df_CH = df_CH.reset_index(drop = True)
df_CH = df_CH.iloc[2:35]

df_CH.density = df_CH.density.str.replace(",","").astype(float)

df_CH.head()
states = list(country_state[country_state.Country_Region=='France'].Province_State)
url = 'https://en.wikipedia.org/wiki/Overseas_France'

html_text = requests.get(url).text

soup = BeautifulSoup(html_text, 'html.parser').findAll('td')
cols = ['country', 'state', 'density']

df_FR = pd.DataFrame(columns = cols)
ix = 0

for ii in range(len(soup)):

    try:

        state = soup[ii].findAll('a')[0].text

    except:

        state = 'none'

    

    if state in states:

        df_FR.loc[ix, 'state'] = state

        df_FR.loc[ix, 'density'] = soup[ii + 4].text[:-1]

        ix += 1



del soup

df_FR.country = 'France'

df_FR = df_FR.reset_index(drop = True)
df_FR = df_FR.iloc[0:6]

df_FR.density = df_FR.density.str.replace(",","").astype(float)

df_FR
df = df_WD.append(df_US).append(df_AU).append(df_CA).append(df_CH).append(df_FR)
df = df.reset_index(drop = True)

df
df.to_csv('population_densities.csv')