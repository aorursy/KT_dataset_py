import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

base = pd.read_csv('/kaggle/input/ocomon.csv', sep=";")
dataframe1 = pd.DataFrame({

        'Técnico': np.array(base['Técnico'].value_counts().index),

        'Número de Atendimentos': np.array(base['Técnico'].value_counts())})  



fig1 = px.bar(dataframe1, y='Técnico', x='Número de Atendimentos', orientation='h')

fig1.update_layout(xaxis_tickangle=-45,

        title={

        'text': "Número de Atendimentos por técnico",

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

                 

fig1.show()
from datetime import datetime



def calculaDiferencaDatas(i,f):   

    formater = '%d/%m/%Y %H:%M'  

    dif = (datetime.strptime(f, formater) - datetime.strptime(i, formater)).total_seconds()

    return np.ceil(dif / 3600)





tecnicos = base['Técnico'].value_counts().index



base_datas = base.loc[base['Data final'] != '-'] 



medias = []



for ele in tecnicos:

    r = base_datas.loc[base_datas['Técnico'] == ele]

    ac = 0

    for d in r.index:

        ac += calculaDiferencaDatas(base_datas['Data inicial'][d], base_datas['Data final'][d])       

    medias.append(int(ac/len(r)))

    



dataframe2 = pd.DataFrame({

     'Técnico': tecnicos,

     'Tempo médio': medias

})





fig2 = px.bar(dataframe2, x='Tempo médio', y='Técnico', orientation='h')



fig2.update_layout(

     title={

        'text': "Tempo Médio de Solução de Chamados por Técnico(Em horas)",

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis = dict(

        title_standoff = 25,

        tickfont_size=10,

    ),

    yaxis = dict(

        title_standoff = 25))



fig2.show()

dataframe3 = pd.DataFrame({

     'Unidade': np.flip(np.array(base['Unidade'].value_counts()[:10].index)),

     'Atendimentos': np.flip(np.array(base['Técnico'].value_counts()[:10]))

})





fig3 = px.bar(dataframe3, x='Unidade', y='Atendimentos')



fig3.update_layout(

     title={

        'text': "10 Unidades com maior demanda",

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis = dict(

        title_standoff = 25,

        tickfont_size=10,

    ),

    yaxis = dict(

        title_standoff = 25))



fig3.show()
dataframe4 = pd.DataFrame({

    'Problema': np.flip(np.array(base['Problema'].value_counts()[:10].index)),

    'Recorrência': np.flip(np.array(base['Problema'].value_counts()[:10]))

})



fig4 = px.bar(dataframe4, x='Recorrência', y='Problema', orientation='h')



fig4.update_layout(

     title={

        'text': "10 Problemas mais recorrentes",

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis = dict(

        title_standoff = 25,

        tickfont_size=10,

    ),

    yaxis = dict(

        title_standoff = 25))



fig4.show()
dataframe5 = pd.DataFrame({

    'Campus': np.flip(np.array(base['Local'].value_counts()[:10].index)),

    'Atendimentos': np.flip(np.array(base['Local'].value_counts()[:10]))

})



fig5 = px.bar(dataframe5, x='Atendimentos', y='Campus', orientation='h')



fig5.update_layout(

     title={

        'text': "Atendimentos por Campi",

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis = dict(

        title_standoff = 25,

        tickfont_size=10,

    ),

    yaxis = dict(

        title_standoff = 25))



fig5.show()
dataframe6 = pd.DataFrame({

    'Área': np.flip(np.array(base['Área'].value_counts()[:10].index)),

    'Atendimentos': np.flip(np.array(base['Área'].value_counts()[:10]))

})



fig6 = px.bar(dataframe6, x='Atendimentos', y='Área', orientation='h')



fig6.update_layout(

     title={

        'text': "Atendimentos por Área",

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'},

    xaxis = dict(

        title_standoff = 25,

        tickfont_size=10,

    ),

    yaxis = dict(

        title_standoff = 25))



fig6.show()