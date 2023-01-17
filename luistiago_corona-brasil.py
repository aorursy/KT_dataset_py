#importando o dataset



import pandas as pd

df = pd.read_csv("../input/corona-virus-brazil/brazil_covid19.csv")
db = df[df['date'] == df['date'].unique()[-1]]
print('Casos Total: ' + str(db['cases'].sum()))

print('Mortes Total: ' + str(db['deaths'].sum()))

print('Letalidade: ' + str(round((db['deaths'].sum() / db['cases'].sum()) * 100,2)))
estados = db['state'].unique()

totalizador = []



for estado in estados:

    totalCasos = db[db['state'] == estado]['cases'].sum()

    totalMortes = db[db['state'] == estado]['deaths'].sum()

    

    totalizador.append({

        'estado': estado,

        'casos': totalCasos,

        'mortes': totalMortes,

        'letalidade': round((totalMortes / totalCasos) * 100, 2)

    })

    

totalizador = sorted(totalizador, key = lambda i: i['casos'],reverse=True)



tabela = pd.DataFrame(totalizador)

tabela
import plotly.express as px



fig = px.bar(tabela, x='estado', y='casos',

             hover_data=['estado','casos'], color='casos',

             labels={'pop':'Casos'}, height=400,title='Casos por estados')

fig.update_layout(template='none')

fig.show()
fig = px.bar(tabela, x='estado', y='mortes',

             hover_data=['estado','mortes'], color='casos',

             labels={'pop':'Mortes'}, height=500,title='Mortes por estados')

fig.update_layout(template='none')

fig.show()
fig = px.bar(tabela, x='estado', y='letalidade',

             hover_data=['estado','letalidade'], color='letalidade',

             labels={'pop':'Letalidade'}, height=400,title='Letalidade por estados')

fig.update_layout(template='none')

fig.show()
brasil = []



for data in df['date'].unique(): 

    dfdata = df[df['date'] == data]

    brasil.append({

      'date': data, 

      'cases': dfdata['cases'].sum(),

      'deaths': dfdata['deaths'].sum()  

    })



dfbrasil = pd.DataFrame(brasil)

dfbrasil