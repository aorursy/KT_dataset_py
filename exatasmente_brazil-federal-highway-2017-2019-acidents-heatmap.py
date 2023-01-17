import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import pandas as pd

# acidentes2018_todas_causas_tipos = pd.read_csv("../input/acidentes2018_todas_causas_tipos.csv",encoding='latin',low_memory=False)

datatran2017 = pd.read_csv("../input/datatran2017.csv",encoding='latin',sep=";",error_bad_lines=False)

datatran2018 = pd.read_csv("../input/datatran2018.csv",encoding='latin',sep=";",error_bad_lines=False)

datatran2019 = pd.read_csv("../input/datatran2019.csv",encoding='latin',sep=";",error_bad_lines=False)



dados = [datatran2017,datatran2018,datatran2019]

dataset = 1
fase_dia={'(null)':None,

          'Amanhecer':'Dawn',

          'Anoitecer':'Nightfall',

          'Plena noite':'Full night',

          'Pleno dia':'Full day'}





dia_semana={'Segunda':'Monday',

            'Terça':'Tuesday',

            'Quarta':'Wednesday',

            'Quinta':'Thursday',

            'Sexta':'Friday',

            'Sábado':'Saturday',

            'Domingo':'Sunday'}

for dataset in range(len(dados)):

    dados[dataset]['data_inversa'] =  pd.to_datetime(pd.Series(dados[dataset]['data_inversa']), format="%Y-%m-%d")

    dados[dataset]['horario'] =  pd.to_datetime(pd.Series(dados[dataset]['horario']), format="%H:%M:%S")

    dados[dataset]['horario'] =  dados[dataset]['horario'].apply(lambda x: x.strftime('%H'))



    dados[dataset]['fase_dia'] = dados[dataset]['fase_dia'].map(fase_dia).astype(str)

    dados[dataset]['dia_semana'] = dados[dataset]['dia_semana'].map(dia_semana).astype(str)



g = {}

plot = []







for datset in range(len(dados)):

    g[dataset]= dados[dataset].groupby(['horario','data_inversa'],as_index=False).count().iloc[:,range(3)]

    g[dataset] = g[dataset].dropna()

    g[dataset].rename(columns={'id':'total'},inplace=True)

    g[dataset]['Month'] = g[dataset]['data_inversa'].apply(lambda x: x.strftime('%m %B'))

    plot.append(pd.pivot_table(g[dataset], values='total', index=['Month'] , columns=['horario'], aggfunc=np.sum))



fig = {}

for dataset in range(len(plot)):

    fig[dataset]=plt.figure()

    if dataset ==0:

        ano = "2017"

    elif dataset == 1:

        ano = "2018"

    else:

        ano = "2019"

    ax = fig[dataset].add_subplot(1, 1, 1)

    ax.set_title(label="Acidentes Por Horário e Mês : Ano "+ ano)

    ax = sns.heatmap(plot[dataset],linewidths=.2,cmap='inferno_r')    

    

plt.show()






