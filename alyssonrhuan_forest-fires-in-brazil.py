%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import folium
df = pd.read_excel('../input/forest-fires-in-brazil-adjusted/amazonfire.xlsx')
df.head()
queimadas_por_mes = df.groupby(['month']).sum()['number']
# Sorting by months chronologically



meses = df['month'].unique()

queimadas_por_mes = queimadas_por_mes.reindex(meses)

queimadas_por_mes
plt.rc('figure', figsize=(15,6))

plt.bar(meses, queimadas_por_mes)

plt.title('Total burnings reported in Brazil from 1998 to 2017 by months')

plt.xlabel('Months', fontsize=15)

plt.ylabel('Number of burns', fontsize=15)

plt.show()
queimadas_por_estado = df.groupby(['state']).sum()['number']
# Sort by quantity

queimadas_por_estado = queimadas_por_estado.sort_values(ascending=False)


plt.rc('figure', figsize=(15,6))

plt.bar(queimadas_por_estado.index, queimadas_por_estado)

plt.title('Total burnings reported in Brazil from 1998 to 2017 by States')

plt.xlabel('States', fontsize=15)

plt.ylabel('Number of burns', fontsize=15)

plt.xticks(fontsize=15, rotation=90)

plt.show()
queimadas_por_ano = df.groupby(['year']).sum()['number']

queimadas_por_ano.index
queimadas_por_ano_label = [str(x) for x in queimadas_por_ano.index]



plt.rc('figure', figsize=(15,6))

plt.plot(queimadas_por_ano_label, queimadas_por_ano)

plt.title('Variation of the number of burns over the years')

plt.xlabel('Years', fontsize=15)

plt.ylabel('Number of burns', fontsize=15)

plt.xticks(fontsize=10, rotation=0)

plt.grid(True)

plt.show()
# Calc average varible by tree
data_inicio = 1999

data_fim = 2016

queimadas_por_ano_media_variavel = []



i = data_inicio

while i < data_fim:

    ano_anterior = queimadas_por_ano[i - 1]

    ano_atual = queimadas_por_ano[i]

    ano_proximo = queimadas_por_ano[i + 1]

    media = (ano_anterior + ano_atual + ano_proximo) / 3

    

    queimadas_por_ano_media_variavel.append(media)

    i += 1

    

queimadas_por_ano_media_variavel = pd.Series(queimadas_por_ano_media_variavel)    
queimadas_por_ano_media_variavel_label = [str(x) for x in range(1999, 2016)]



plt.rc('figure', figsize=(15,6))

plt.plot(queimadas_por_ano_media_variavel_label, queimadas_por_ano_media_variavel)

plt.title('Variable average number of burns over the years')

plt.xlabel('Years', fontsize=15)

plt.ylabel('Number of burns', fontsize=15)

plt.xticks(fontsize=10, rotation=0)

plt.grid(True)

plt.show()
variacoes = [0]



for i in range(len(queimadas_por_ano_media_variavel)):

    if i > 0:

        variacao = ( (queimadas_por_ano_media_variavel[i] * 100 ) / queimadas_por_ano_media_variavel[i-1] ) - 100

        variacoes.append(variacao)
media_variacoes = 0

proxima_queimada = 0.0

queimadas_por_ano_media_variavel = queimadas_por_ano_media_variavel.tolist()



for tendencias in range(5):

    for i in range(len(variacoes)):

        media_variacoes += variacoes[i]



    media_variacoes = media_variacoes / len(variacoes)

    variacoes.append(media_variacoes)

    proxima_queimada = queimadas_por_ano_media_variavel[len(queimadas_por_ano_media_variavel) -1 ] * ( ( variacoes[len(variacoes) - 1] / 100 ) + 1 )

    queimadas_por_ano_media_variavel.append(proxima_queimada)
queimadas_por_ano_media_variavel = pd.Series(queimadas_por_ano_media_variavel)    
queimadas_por_ano_media_variavel_label = [str(x) for x in range(1999, 1999 + len(queimadas_por_ano_media_variavel))]



plt.rc('figure', figsize=(15,6))

plt.plot(queimadas_por_ano_media_variavel_label, queimadas_por_ano_media_variavel)

plt.title('Projection of burns for the next 5 years')

plt.xlabel('Years', fontsize=15)

plt.ylabel('Number of burns', fontsize=15)

plt.xticks(fontsize=10, rotation=0)

plt.grid(True)

plt.show()
brasil = folium.Map(

    location=[-16.1237611, -59.9219642],    # Coordenadas retiradas do Google Maps

    zoom_start=4

)
geo_data_brasil = '../input/geojson/Brasil.json'
len(df['state'].unique())

print(df['state'].unique())
dic_states = {

    'Mato Grosso': 'MT',

    'Paraiba': 'PB',

    'Sao Paulo': 'SP',

    'Rio de Janeiro': 'RJ',

    'Bahia': 'BA',

    'Piauí': 'PI',

    'Goias': 'GO',

    'Minas Gerais': 'MG',

    'Tocantins': 'TO',

    'Amazonas': 'AM',

    'Ceara': 'CE',

    'Maranhao': 'MA',

    'Pará': 'PA',

    'Pernambuco': 'PE',

    'Roraima': 'RR',

    'Santa Catarina': 'SC',

    'Amapa': 'AP',

    'Rondonia': 'RO',

    'Acre': 'AC',

    'Espirito Santo': 'ES',

    'Alagoas': 'AL',

    'Distrito Federal': 'DF',

    'Sergipe': 'SE',

    'Mato Grosso do Sul': 'MS',

    'Paraná': 'PR',

    'Rio Grande do Sul': 'RS',

    'Rio Grande do Norte': 'RN'

}

df_queimadas_por_estado = pd.DataFrame(columns = ['state', 'number'])
queimadas_por_estado = queimadas_por_estado.reset_index()

for index, row in queimadas_por_estado.iterrows():

    state = dic_states[row[0]]

    number = int(row[1])

    df_aux = pd.DataFrame({'state' : [state], 'number': [number]})

    df_queimadas_por_estado = pd.concat([df_queimadas_por_estado, df_aux])
df_queimadas_por_estado.reset_index(inplace = True, drop=True)

df_queimadas_por_estado
folium.Choropleth(

    geo_data = geo_data_brasil,

    name='choropleth',

    data = df_queimadas_por_estado,

    columns = ['state', 'number'],

    fill_color = 'OrRd',

    key_on = 'feature.properties.UF',

    fill_opacity = 0.7,

    line_opacity = 0.2,

    legend_name = 'Number of fires by State'

).add_to(brasil)



folium.LayerControl().add_to(brasil)

brasil