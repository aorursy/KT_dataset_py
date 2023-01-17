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
covid_19_clean_complete = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')

country_wise_latest = pd.read_csv('/kaggle/input/corona-virus-report/country_wise_latest.csv')

worldometer_data =  pd.read_csv('/kaggle/input/corona-virus-report/worldometer_data.csv')

day_wise =  pd.read_csv('/kaggle/input/corona-virus-report/day_wise.csv')

usa_county_wise =  pd.read_csv('/kaggle/input/corona-virus-report/usa_county_wise.csv')

full_grouped =  pd.read_csv('/kaggle/input/corona-virus-report/full_grouped.csv')
covid_19_clean_complete.head()
covid_19_clean_complete[covid_19_clean_complete['Country/Region'] == 'US' ]
country_wise_latest.head()
country_wise_latest[country_wise_latest['Country/Region'] == 'US' ]
worldometer_data.head()
worldometer_data[worldometer_data['Country/Region'] == 'USA' ]
day_wise.head()
usa_county_wise.head()
full_grouped.head()
full_grouped[full_grouped['Country/Region'] == 'US' ]
full_grouped.loc[full_grouped['Country/Region'] == 'US', ['Country/Region']] = 'USA'
full_grouped[full_grouped['Country/Region'] == 'USA' ]
#full_grouped.set_index('Country/Region', inplace=True)

full_grouped
#worldometer_data.set_index('Country/Region', inplace=True)

worldometer_data
worldometer_data.loc[worldometer_data['Continent'] == 'North America', ['Continent']] = 'America'

worldometer_data.loc[worldometer_data['Continent'] == 'South America', ['Continent']] = 'America'
full_grouped.groupby(['WHO Region'], as_index = False)['Confirmed', 'Deaths', 'Recovered', 'Active'].sum()
full_grouped
worldometer_data
worldometer_data['TotalTests'] = worldometer_data['TotalTests'].fillna(0)

worldometer_data['Population'] = worldometer_data['Population'].fillna(0)

worldometer_data['TotalCases'] = worldometer_data['TotalCases'].fillna(0)

worldometer_data['TotalDeaths'] = worldometer_data['TotalDeaths'].fillna(0)

worldometer_data['TotalRecovered'] = worldometer_data['TotalRecovered'].fillna(0)
paises = worldometer_data[['Country/Region','Continent', 'Population', 'TotalTests', 'WHO Region', 'TotalCases', 'TotalDeaths', 'TotalRecovered']]
#paises.dropna(inplace=True)

paises
#paises['Population'] = paises['Population'].apply(np.int64)

#paises['TotalTests'] = paises['TotalTests'].apply(np.int64)

paises.loc[ : , ['Population']] = paises['Population'].apply(np.int64)

paises.loc[ : , ['TotalTests']] = paises['TotalTests'].apply(np.int64)

paises.loc[ : , ['TotalDeaths']] = paises['TotalDeaths'].apply(np.int64)

paises.loc[ : , ['TotalRecovered']] = paises['TotalRecovered'].apply(np.int64)
paises.groupby(['Continent'], as_index = False)['Population', 'TotalTests'].sum()
paises[paises['Country/Region'] == 'Brazil']
#full_grouped[full_grouped['Country/Region'] == 'Brazil']

full_grouped.loc[ : , ['Date']] = pd.to_datetime(full_grouped['Date'])

full_grouped.loc[ : , ['Deaths']] = full_grouped['Deaths'].apply(np.int64)

full_grouped.loc[ : , ['Active']] = full_grouped['Active'].apply(np.int64)

full_grouped.info()
full_grouped.groupby(['Country/Region'], as_index = False)['New cases', 'New deaths', 'New deaths'].sum()
paises.groupby(['Continent'], as_index = False)['Population', 'TotalTests'].sum()
casos = full_grouped[['Date', 'Country/Region', 'New cases', 'New deaths', 'New recovered', 'WHO Region', 'Confirmed', 'Deaths', 'Recovered']]
casos[casos['Date'] == '2020-06-10']
casos[casos['Country/Region'] == 'Brazil'].groupby(['Country/Region'], as_index = False)['New cases', 'New deaths'].sum()
paises.columns = ['Pais', 'Continente', 'Populacao', 'Testes', 'Regiao', 'CasosTotal', 'MortosTotal', 'RecuperadosTotal']

paises.info()
casos.columns = ['Data', 'Pais', 'Casos', 'Mortos', 'Recuperados', 'Regiao', 'ConfirmadosTotal', 'MortosTotal', 'RecuperadosTotal']

casos.info()
#paises.to_csv('Paises.csv', index=False)

#casos.to_csv('Casos.csv', index=False)
paises['porcentagemCasos'] = (paises['CasosTotal'] / paises['Populacao']) * 100

paises['porcentagemMortos'] = (paises['MortosTotal'] / paises['Populacao']) * 100

paises['porcentagemRecuperados'] = (paises['RecuperadosTotal'] / paises['Populacao']) * 100

paises['porcentagemMortosCasos'] = (paises['MortosTotal'] / paises['CasosTotal']) * 100

pd.DataFrame(paises.sort_values(by=['porcentagemCasos'], ascending=False).loc[paises['Populacao'] > 0 , :].head(30).values, columns=paises.columns)

#pd.DataFrame(paises.sort_values(by=['porcentagemMortosCasos'], ascending=False).loc[paises['Populacao'] > 0 , :].head(30).values, columns=paises.columns)
grafico = pd.DataFrame(paises.loc[: , ['Pais', 'CasosTotal', 'MortosTotal', 'RecuperadosTotal']].values, columns=['Pais', 'CasosTotal', 'MortosTotal', 'RecuperadosTotal'])

grafico.set_index('Pais', inplace=True)

grafico.sort_values(by=['CasosTotal'], ascending=False).head(10)
grafico.sort_values(by=['MortosTotal'], ascending=False).head(10)
grafico.sort_values(by=['RecuperadosTotal'], ascending=False).head(10)
# import matplotlib.pyplot as plt

# f, ax = plt.subplots(figsize=(15,15))

# plt.bar(graficoCasos['Pais'].values, graficoCasos['CasosTotal'].values, color='b')

# plt.bar(graficoCasos['Pais'].index, graficoCasos['MortosTotal'].values, color='r')

# plt.xticks(graficoCasos['Pais'].values)

# plt.yticks(graficoCasos['CasosTotal'].values)



# plt.show()