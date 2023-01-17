import pandas as pd

covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

totalDeaths = covid_19_data['Deaths'].sum()

totalRecovered = covid_19_data['Recovered'].sum()



print('Letalidade Global: ' + str(round((totalDeaths / (totalRecovered + totalDeaths)) * 100, 2)))
from decimal import Decimal



paises = covid_19_data['Country/Region'].unique()

casos = []

for pais in paises:

    totalConfirmed = covid_19_data[covid_19_data['Country/Region'] == pais]['Confirmed'].sum()

    totalRecovered = covid_19_data[covid_19_data['Country/Region'] == pais]['Recovered'].sum()

    totalDeaths = covid_19_data[covid_19_data['Country/Region'] == pais]['Deaths'].sum()

    caso = {'pais':pais, 'confirmed': totalConfirmed, 'recovered': totalRecovered, 'deaths':totalDeaths, 'letalidade':0}

    if totalRecovered > 0:

        caso['letalidade'] =  round((totalDeaths / (totalRecovered + totalDeaths) * 100), 2)

    casos.append(caso)

    



casos = sorted(casos, key = lambda i: i['confirmed'],reverse=True)

casos

[c for c in casos if c['pais'] in 'Brazil']

import plotly.express as px



df = pd.DataFrame(casos)



data= df[df['confirmed'] > 1000].sort_values('letalidade',ascending=False)[0:30]



fig = px.bar(data, x='pais', y='letalidade',

             hover_data=['pais','letalidade'], color='letalidade',

             labels={'pop':'Letalidade'}, height=400,title='Letalidade por pais')

fig.update_layout(template='none')

fig.show()