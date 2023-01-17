import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/international-football-results-from-1872-to-2017/results.csv')

df.head()
df.shape
df.head()
df.tail()
df.info()
df.rename(columns={'date':'Data', 

                   'home_team':'Time da Casa',

                   'away_team':'Time Visitantes',

                   'home_score':'Gols Casa',

                   'away_score':'Gols Visitantes',

                   'tournament':'Competição',

                   'city':'Cidade',

                   'country':'País',

                   'neutral':'Local Neutro'                                                  

                  }, inplace=True)



df.head()
df = df[df['Competição'] != 'Friendly']

df.head()
df['Local Neutro'] = np.where(df['Local Neutro'] == False, 'Não', 'Sim')

df.head()
df['Total de Gols'] = df['Gols Visitantes'] + df['Gols Casa']

df.head()
resultado = []



for index, result in df.iterrows():

    if result['Gols Casa'] > result['Gols Visitantes']:

        resultado.append('Casa')



    elif result['Gols Casa'] < result['Gols Visitantes']:

        resultado.append('Visitantes')

    

    else:

        resultado.append('Empate')



df['Vitória'] = resultado



df.head()
df['Ano'] = pd.DatetimeIndex(df['Data']).year

df.head()
f, ax = plt.subplots(figsize=(6,4))

sns.heatmap(df[{'Gols Casa', 'Gols Visitantes', 'Ano'}].corr(), annot=True, fmt='.2f', linecolor='black', ax=ax, lw=.7)

f.tight_layout()
pd.value_counts(df['Vitória']).plot.bar(color='#4682B4')

plt.xticks(rotation=0)

plt.show()
df.sort_values(['Gols Casa'], ascending=[False]).head(10).style.hide_index()
df.sort_values(['Gols Visitantes'], ascending=[False]).head(10).style.hide_index()
df[{'Gols Casa', 'Time da Casa', 'Ano'}].groupby(['Time da Casa']).agg({'Gols Casa':'sum', 'Ano':'count'}).reset_index().rename(columns={'Ano':'Qtd de Jogos'}).nlargest(5, 'Gols Casa').style.hide_index()
df[{'Gols Visitantes', 'Time Visitantes', 'Ano'}].groupby(['Time Visitantes']).agg({'Gols Visitantes':'sum', 'Ano':'count'}).reset_index().rename(columns={'Ano':'Qtd de Jogos'}).nlargest(5, 'Gols Visitantes').style.hide_index()
df[{'Gols Casa', 'Competição', 'Ano'}].groupby(['Competição']).agg({'Gols Casa':'sum', 'Ano':'count'}).reset_index().rename(columns={'Ano':'Qtd de Jogos'}).nlargest(5, 'Gols Casa').style.hide_index()
df[{'Gols Visitantes', 'Competição', 'Ano'}].groupby(['Competição']).agg({'Gols Visitantes':'sum', 'Ano':'count'}).reset_index().rename(columns={'Ano':'Qtd de Jogos'}).nlargest(5, 'Gols Visitantes').style.hide_index()
df[{'Gols Casa', 'Time da Casa', 'Ano', 'Data'}].groupby(['Time da Casa', 'Ano']).agg({'Gols Casa':'sum', 'Data':'count'}).reset_index().rename(columns={'Data':'Qtd de Jogos'}).nlargest(5, 'Gols Casa').style.hide_index()
df[{'Gols Visitantes', 'Time Visitantes', 'Ano', 'Data'}].groupby(['Time Visitantes', 'Ano']).agg({'Gols Visitantes':'sum', 'Data':'count'}).reset_index().rename(columns={'Data':'Qtd de Jogos'}).nlargest(5, 'Gols Visitantes').style.hide_index()
df_atual = df[df['Ano'] >= 2010]

df_atual.head()
# Dicionário das colunas numéricas

dicionario = {'Gols Casa':1, 'Gols Visitantes':2, 'Total de Gols':3}



# Printo os boxplots

plt.figure(figsize=(15,6))



for variavel,i in dicionario.items():

    plt.subplot(1,3,i)

    plt.boxplot(df_atual[variavel], whis=1.5)

    plt.title(variavel)

    

    # tornar em inteiros os valores apresentados no eixo y

    yint = []

    for vl in range(12):

        if (df_atual[variavel].max() + 2 > (vl * 2)):

            yint.append(int(vl * 2))

    plt.yticks(yint)



plt.show()
pd.value_counts(df_atual['Vitória']).plot.bar(color='#4682B4')

plt.xticks(rotation=0)

plt.show()
plt.plot('Ano',

         'Total de Gols',

         data=df_atual[{'Total de Gols', 'Ano'}].groupby(['Ano']).agg({'Total de Gols':'sum'}).reset_index(), # retorno df com o total de gols por ano

         color='skyblue')

plt.grid(True)

plt.show()
# separo as partidas onde houve vitória

df_vitorias_vis = df_atual[df_atual['Vitória'] == 'Visitantes'][{'Vitória', 'Time Visitantes'}].groupby(['Time Visitantes']).agg({'Vitória':'count'}).reset_index().rename(columns={'Time Visitantes':'Time', 'Vitória':'Vitórias'})

df_vitorias_casa = df_atual[df_atual['Vitória'] == 'Casa'][{'Vitória', 'Time da Casa'}].groupby(['Time da Casa']).agg({'Vitória':'count'}).reset_index().rename(columns={'Time da Casa':'Time', 'Vitória':'Vitórias'})



# uno os dataframes

# uso outer, pois existem times sem registro em casa, assim como times sem registro fora de casa

df_vitorias = pd.merge(df_vitorias_vis, df_vitorias_casa, on = 'Time', how='outer', suffixes=('_vis', '_casa'))



# preencho os valores nulos com 0

df_vitorias['Vitórias_vis'] = df_vitorias['Vitórias_vis'].fillna(0)

df_vitorias['Vitórias_casa'] = df_vitorias['Vitórias_casa'].fillna(0)



# somo as vitórias

df_vitorias['Total Vitórias'] = df_vitorias['Vitórias_vis'] + df_vitorias['Vitórias_casa']



# deixo somente o top 5 e reordeno o dataframe para printar em ordem crescente

df_top5_vitorias = df_vitorias.nlargest(5, 'Total Vitórias').sort_values('Total Vitórias')



plt.figure(figsize=(6,7))

ax = sns.barplot(

    data=df_top5_vitorias,

    x='Time', 

    y='Total Vitórias',

    palette="Blues_d")



# printo os valores acima das barras

for i, v in enumerate(df_top5_vitorias['Total Vitórias'].iteritems()):        

    ax.text(i - 0.05 ,v[1], int(v[1]), color='black', va ='bottom', rotation=0)

    

plt.tight_layout()

plt.show()
# separo as partidas onde houve vitória

df_gols_vis = df_atual[{'Gols Visitantes', 'Time Visitantes', 'Ano'}].groupby(['Time Visitantes']).agg({'Gols Visitantes':'sum', 'Ano':'count'}).reset_index().rename(columns={'Time Visitantes':'Time', 'Gols Visitantes':'Qtd de Gols', 'Ano':'Qtd de Jogos'})

df_gols_casa = df_atual[{'Gols Casa', 'Time da Casa', 'Ano'}].groupby(['Time da Casa']).agg({'Gols Casa':'sum', 'Ano':'count'}).reset_index().rename(columns={'Time da Casa':'Time', 'Gols Casa':'Qtd de Gols', 'Ano':'Qtd de Jogos'})



# uno os dataframes

# uso outer, pois existem times sem registro em casa, assim como times sem registro fora de casa

df_gols = pd.merge(df_gols_vis, df_gols_casa, on = 'Time', how='outer', suffixes=('_vis', '_casa'))



# preencho os valores nulos com 0

df_gols['Qtd de Gols_vis'] = df_gols['Qtd de Gols_vis'].fillna(0)

df_gols['Qtd de Jogos_vis'] = df_gols['Qtd de Jogos_vis'].fillna(0)

df_gols['Qtd de Gols_casa'] = df_gols['Qtd de Gols_casa'].fillna(0)

df_gols['Qtd de Jogos_casa'] = df_gols['Qtd de Jogos_casa'].fillna(0)



# somo as quantidades de gols

df_gols['Total de Gols'] = df_gols['Qtd de Gols_vis'] + df_gols['Qtd de Gols_casa']



# deixo somente o top 5 e reordeno o dataframe para printar em ordem crescente

df_top5_gols = df_gols.nlargest(5, 'Total de Gols').sort_values('Total de Gols')



plt.figure(figsize=(6,8))

ax = sns.barplot(

    data=df_top5_gols,

    x='Time', 

    y='Total de Gols',

    palette="Blues_d")



# printo os valores acima das barras

for i, v in enumerate(df_top5_gols['Total de Gols'].iteritems()):        

    ax.text(i - 0.1 ,v[1], int(v[1]), color='black', va ='bottom', rotation=0)

    

plt.tight_layout()

plt.show()
# Aproveito os dataframes anteriores e os uno

df_resumo = pd.merge(df_vitorias, df_gols, on = 'Time', how='outer')



# trato os nulos, existem times sem vitórias

df_resumo['Vitórias_vis'] = df_resumo['Vitórias_vis'].fillna(0)

df_resumo['Vitórias_casa'] = df_resumo['Vitórias_casa'].fillna(0)

df_resumo['Total Vitórias'] = df_resumo['Total Vitórias'].fillna(0)



# Renomeio algumas colunas

df_resumo.rename(columns={

    'Vitórias_vis':'Vitórias Fora', 

    'Vitórias_casa':'Vitórias Casa',

    'Qtd de Gols_vis':'Qtd de Gols Fora',

    'Qtd de Gols_casa':'Qtd de Gols Casa',

    'Qtd de Jogos_vis':'Qtd de Jogos Fora',

    'Qtd de Jogos_casa':'Qtd de Jogos Casa'

}, inplace=True)



# Adiciono as colunas "% de Vitória Casa", "% de Vitória Fora", "% Total de Vitórias" e "Qtd Total de Jogos"

df_resumo['% de Vitórias Casa'] =  np.around(df_resumo['Vitórias Casa'] / df_resumo['Qtd de Jogos Casa'], decimals=2)

df_resumo['% de Vitórias Fora'] =  np.around(df_resumo['Vitórias Fora'] / df_resumo['Qtd de Jogos Fora'], decimals=2)

df_resumo['% Total de Vitórias'] =  np.around((df_resumo['Vitórias Casa'] + df_resumo['Vitórias Fora']) / (df_resumo['Qtd de Jogos Casa'] + df_resumo['Qtd de Jogos Fora']), decimals=2)

df_resumo['Qtd Total de Jogos'] = df_resumo['Qtd de Jogos Fora'] + df_resumo['Qtd de Jogos Casa']



df_resumo.head()
plt.figure(figsize=(7,7))



sns.scatterplot(data=df_resumo,

                x='Total de Gols', 

                y='Total Vitórias'

               )
df_top10_porc_vitorias = df_resumo[df_resumo['Qtd Total de Jogos'] > df_resumo['Qtd Total de Jogos'].mean()][{'Time', '% Total de Vitórias', 'Qtd Total de Jogos'}].nlargest(10, '% Total de Vitórias')



# deixo somente o top 5 e reordeno o dataframe para printar em ordem crescente

df_top10_porc_vitorias = df_top10_porc_vitorias.sort_values('% Total de Vitórias')



plt.figure(figsize=(12,8))

ax = sns.barplot(

    data=df_top10_porc_vitorias,

    x='Time', 

    y='% Total de Vitórias',

    palette="Blues_d")



# printo os valores acima das barras

for i, v in enumerate(df_top10_porc_vitorias['% Total de Vitórias'].iteritems()):

    ax.text(i ,v[1], v[1], color='black', va='bottom', rotation=45)     

    

plt.tight_layout()

plt.show()
brasil = df_atual[(df_atual['Time da Casa'] == 'Brazil') | (df_atual['Time Visitantes'] == 'Brazil')].copy()

brasil.head()
brasil[brasil['Time da Casa'] == 'Brazil'].nlargest(5, 'Gols Casa')
brasil[brasil['Time Visitantes'] == 'Brazil'].nlargest(5, 'Gols Visitantes')
plt.figure(figsize=(7,7))

sns.swarmplot(y='Gols Casa', x='Local Neutro', data=brasil[brasil['Time da Casa'] == 'Brazil'])
brasil_resumo_competicao = brasil[brasil['Time da Casa'] == 'Brazil'][{'Time da Casa', 'Gols Casa', 'Competição', 'Ano', 'Vitória'}].rename(columns={'Time da Casa':'Time', 'Gols Casa':'Gols'}).append(brasil[brasil['Time Visitantes'] == 'Brazil'][{'Time Visitantes', 'Gols Visitantes', 'Competição', 'Ano', 'Vitória'}].rename(columns={'Time Visitantes':'Time', 'Gols Visitantes':'Gols'}), sort=False)

brasil_resumo_competicao.head()
f, ax = plt.subplots(figsize=(9,5))



plot = brasil_resumo_competicao[{'Ano', 'Gols'}].groupby(['Ano']).mean()[{'Gols'}].plot(ax=ax)

plot.legend(loc='upper left', ncol=1)



plt.show()



f.tight_layout()
brasil_gols_competicao = brasil_resumo_competicao.groupby(['Competição']).agg({'Gols':'sum', 'Time':'count'}).rename(columns={'Time':'Qtd de Jogos'}).sort_values('Gols').reset_index()



plt.figure(figsize=(10,4))

ax = sns.barplot(

    data=brasil_gols_competicao,

    y='Competição', 

    x='Gols',

    palette='Blues_d')



for index, row in brasil_gols_competicao.iterrows():    

    ax.text(2, index, 'Qtd de Jogos: ' + str(row['Qtd de Jogos']), color='white', va='bottom', rotation=0) 

    ax.text(row['Gols'] + 0.2, index, row['Gols'], color='black', va='bottom', rotation=0) 

    

plt.tight_layout()

plt.show()
plt.figure(figsize=(12,6))



plt.subplot(1,2,1)

pd.value_counts(brasil[(brasil['Time da Casa'] == 'Brazil')]['Vitória']).plot.bar(color='#4682B4')

plt.title('Resultados dos jogos em casa')

plt.xticks(rotation=0)



plt.subplot(1,2,2)

pd.value_counts(brasil[(brasil['Time Visitantes'] == 'Brazil')]['Vitória']).plot.bar(color='#708090')

plt.title('Resultados dos jogos fora de casa')

plt.xticks(rotation=0)



plt.show()
# junto as vitórias, visitante + casa

df_brasil_vitorias = brasil[(brasil['Time da Casa'] == 'Brazil') & (brasil['Vitória'] == 'Casa')][{'Competição', 'Ano'}].append(brasil[(brasil['Time Visitantes'] == 'Brazil') & (brasil['Vitória'] == 'Visitantes')][{'Competição', 'Ano'}])

df_brasil_derrotas = brasil[(brasil['Time da Casa'] == 'Brazil') & (brasil['Vitória'] == 'Visitantes')][{'Competição', 'Ano'}].append(brasil[(brasil['Time Visitantes'] == 'Brazil') & (brasil['Vitória'] == 'Casa')][{'Competição', 'Ano'}])

df_brasil_empates = brasil[(brasil['Vitória'] == 'Empate')][{'Competição', 'Ano'}]



# agrupo por competição

df_brasil_vitorias_competicao = df_brasil_vitorias[{'Competição', 'Ano'}].groupby(['Competição']).agg({'Ano':'count'}).rename(columns={'Ano':'Qtd de Vitórias'}).sort_values('Competição').reset_index()



plt.figure(figsize=(10,4))

ax = sns.barplot(

    data=df_brasil_vitorias_competicao,

    y='Competição', 

    x='Qtd de Vitórias',

    palette='Blues_d')



for index, row in df_brasil_vitorias_competicao.iterrows():    

    ax.text(row['Qtd de Vitórias'] + 0.05, index, row['Qtd de Vitórias'], color='black', va='bottom', rotation=0)

    ax.text(1, index, 'Qtd de Jogos: ' + str(df_brasil_vitorias[df_brasil_vitorias['Competição'] == row['Competição']].count()[0] + df_brasil_empates[df_brasil_empates['Competição'] == row['Competição']].count()[0] + df_brasil_derrotas[df_brasil_derrotas['Competição'] == row['Competição']].count()[0]), color='white', va='bottom', rotation=0) 

    

plt.tight_layout()

plt.show()