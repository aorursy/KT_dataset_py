# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os, re

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
atlas = pd.read_csv('../input/hdi-brazil-idh-brasil/atlas.csv')

desc = pd.read_csv('../input/hdi-brazil-idh-brasil/desc.csv')

ibge = pd.read_csv('../input/ibgedataset/ibge_estados.csv', sep=';')

ibge = ibge.rename(columns={'uf':'sigla','codigo_uf':'uf'})



atlas = atlas.merge(ibge, on='uf', how='left')
for _,d in desc.iterrows():

    print('{}: {}'.format(d['SIGLA'], d['nome curto']))

atlas.uf.head()
espvida_media_uf = atlas[atlas.ano == 2010].groupby(['sigla']).espvida.mean()

espvida_media_uf = pd.DataFrame({'sigla':espvida_media_uf.index, 'espvida':espvida_media_uf.values})

sns.set(rc={'figure.figsize':(10,6)})

sns.barplot(x='sigla', y='espvida', data=espvida_media_uf.sort_values(['espvida']).reset_index(drop=True), color='darkblue').set(title='Gráfico de barras', 

                                                                                                                                 xlabel = 'Estado', 

                                                                                                                                 ylabel='Espectativa de vida ao nascer (anos)')
cidades_nordeste = atlas[atlas.uf.isin(list(range(21,30))) & atlas.ano.isin([2010])]

pobreza_analf = cidades_nordeste.groupby(['sigla'], as_index=False)['pind','t_analf25m'].mean()

pobreza_analf_melt = pd.melt(frame=pobreza_analf,id_vars='sigla',value_vars=['pind','t_analf25m'],var_name='tipo',value_name='taxa')

pobreza_analf_melt.loc[pobreza_analf_melt.tipo == 'pind', 'tipo'] = 'Percentual de extrema pobreza'

pobreza_analf_melt.loc[pobreza_analf_melt.tipo == 't_analf25m', 'tipo'] = 'Percentual de analfabetismo (25 ou + anos)'

sns.catplot(x="sigla", y="taxa", hue='tipo', 

            data=pobreza_analf_melt, 

            kind="bar", 

            height=6, aspect=11/6,

            palette="muted").set(title='Gráfico de barras agrupadas',xlabel = 'Estado',ylabel='Taxa (de 1 a 100%)')

evolucao_idh = atlas[atlas.uf.isin(list(range(21,30)))]

evolucao_idh.groupby(['sigla','ano'], as_index = False).idhm.mean()

sns.relplot(data=evolucao_idh, x='ano', y='idhm', hue='sigla', kind='line', height=8, aspect=11/8).set(title='Gráfico linhas - Evolução do IDH',xlabel = 'Estado',ylabel='Taxa (de 1 a 100%)')
#idhm_r

cidades_nordeste = atlas[atlas.uf.isin(list(range(21,30))) & atlas.ano.isin([2010])]

sns.set(rc={'figure.figsize':(16,12)})

sns.scatterplot(x="espvida", y="e_anosestudo", data=cidades_nordeste, hue='sigla').set(xlabel = 'Espectativa de vida ao nascer', ylabel = 'Anos de estudo')
campos_piramide_populacional = desc.loc[lambda x: (x['SIGLA'].str.contains('mulh') | 

                    x['SIGLA'].str.contains('hom')) &

         ~x['SIGLA'].str.contains('to'), 'SIGLA']

campos_piramide_populacional = campos_piramide_populacional.to_frame()

sexo_faixa = [('m' if x[0] == 'h' else 'f', g[0] if g[0] is not None else g[1]) for (x, g) in [(x, re.compile(".*[mhrs](\d{1,2}a*\d{1,2})*").search(x.strip()).groups()) for x in campos_piramide_populacional.SIGLA]]

sexo, faixa = zip(*sexo_faixa)

campos_piramide_populacional['sexo'] = list(sexo)

campos_piramide_populacional['faixa_etaria'] = list(faixa)

campos_piramide_populacional = campos_piramide_populacional.rename(columns={'SIGLA':'tipo'})
cidades_nordeste_piramide = cidades_nordeste[['sigla'] + list(campos_piramide_populacional.tipo)].groupby('sigla', as_index=False).sum()

cidades_nordeste_piramide = cidades_nordeste_piramide.melt(id_vars='sigla', value_vars=list(campos_piramide_populacional.tipo), var_name='tipo', value_name='quantidade')

cidades_nordeste_piramide = cidades_nordeste_piramide.merge(campos_piramide_populacional)

cidades_nordeste_piramide.faixa_etaria = cidades_nordeste_piramide['faixa_etaria'].astype('str')

cidades_nordeste_piramide.quantidade = pd.to_numeric(cidades_nordeste_piramide.quantidade)

cidades_nordeste_piramide.head()



cidades_nordeste_piramide.loc[cidades_nordeste_piramide.sexo == 'f', 'quantidade'] = cidades_nordeste_piramide.loc[cidades_nordeste_piramide.sexo == 'f'].quantidade * -1

#cidades_nordeste_violin.loc[cidades_nordeste_violin.sexo == 'f', 'quantidade']
def piramide(x,y,z,**kwargs):

    df = kwargs.pop("data")

    order_of_bars =['80', '75a79', '70a74', '65a69', '60a64', '55a59', '50a54',

       '45a49', '40a44', '35a39', '30a34', '25a29', '20a24', '15a19',

       '10a14', '5a9', '0a4']

    colors = [plt.cm.Spectral(i/float(len(df[z].unique())-1)) for i in range(len(df[z].unique()))]

    g = None

    for c, group in zip(colors, df[z].unique()):

        g = sns.barplot(x=x, y=y, data=df.loc[df[z]==group, :], order=order_of_bars, color=c, label=group)

    g.set(xlabel = "população", ylabel = "faixa etária")

    return g



g = sns.FacetGrid(cidades_nordeste_piramide, col="sigla", col_wrap=3, height=8)

g = g.map_dataframe(piramide, 'quantidade', 'faixa_etaria', 'sexo')

g.add_legend()



#sns.barplot(x="quantidade", y="faixa_etaria", hue='sexo', data=cidades_nordeste_violin, orient='h')

#g = sns.FacetGrid(cidades_nordeste_violin, row="sexo", col="sigla", margin_titles=True)

#g.map(sns.barplot, "faixa_etaria", "quantidade")



evolucao_espvida = atlas[atlas.uf.isin(list(range(21,30)))]

#evolucao_espvida = evolucao_espvida[evolucao_espvida.uf == 24]

evolucao_espvida.groupby(['sigla','ano'], as_index = False).t_agua.mean()

sns.catplot(data=evolucao_espvida, x='sigla', y='t_agua', kind='violin', height=8, aspect=11/8).set(title='Violino - Água Encanada',xlabel = 'Estado',ylabel='% de Água Encanada nos Municípios do Estado (1 até 100%)')