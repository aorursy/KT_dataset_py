import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from plotnine import *

import os
data = pd.read_csv('../input/enem-por-escola/MICRODADOS_ENEM_ESCOLA.csv', 

                   error_bad_lines = False, 

                   delimiter = ';', 

                   encoding = 'latin-1')
data.head()
data.tail()
data.info()
data.shape
data.isna().sum()
data.isna().sum().sum()
ax = sns.relplot(x = 'NU_ANO', 

                 y = 'NU_PARTICIPANTES_NEC_ESP', 

                 kind = 'line', 

                 data = data, 

                 aspect = 1.5)



ax.set(xlabel = 'Ano', ylabel = 'Qtde participantes c/ necessidade especial')

plt.show()
x = data['SG_UF_ESCOLA'].unique()

y = []
x
#data[data['SG_UF_ESCOLA'] == i]['NU_MEDIA_TOT'].sum()
for i in x:

    media = data[data['SG_UF_ESCOLA'] == i]['NU_MEDIA_TOT'].sum() / data[data['SG_UF_ESCOLA'] == i]['NU_MEDIA_TOT'].count()

    y.append(media)
dic = dict(uf = x, media_total = y)



df = pd.DataFrame(data = dic)
df.head(10)
(ggplot(df, aes(x = 'uf', y = 'media_total')) + 

    geom_col() +

    labs(title = 'Média das escolas por estado',

         x = 'Estado',

         y = 'Média das escolas'))
(ggplot(data, aes(x = 'PORTE_ESCOLA', fill = 'PORTE_ESCOLA')) + 

    geom_bar(stat = 'count') +

    labs(title = 'Distribuição de escolas por porte',

         fill = 'Portes',

         x = 'Porte das escolas',

         y = 'Quantidade de escolas'))
sns.relplot(x = 'NU_ANO', y = 'NU_MEDIA_TOT', kind = 'line', data = data, aspect = 1.5)
#ggplot(data, aes(x = 'NU_ANO', y = 'NU_TAXA_APROVACAO')) + geom_line() + theme_bw()
sns.relplot(x = 'NU_ANO', y = 'NU_TAXA_APROVACAO', kind = 'line', data = data, aspect = 1.5)
sns.relplot(x = 'NU_ANO', y = 'NU_TAXA_REPROVACAO', kind = 'line', data = data, aspect = 1.5)
sns.relplot(x = 'NU_ANO', y = 'NU_TAXA_ABANDONO', kind = 'line', data = data, aspect = 1.5)