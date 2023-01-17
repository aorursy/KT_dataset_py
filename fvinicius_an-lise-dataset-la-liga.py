# Manipulação de Dataframes

import pandas as pd 



# Funções de algebra linear

import numpy as np



# Biblioteca para laços

import itertools



# Visulização de Gráficos

from matplotlib.ticker import MaxNLocator

import matplotlib.pyplot as plt

import seaborn as sns

# Carrega arquivo 'csv' em um dataframe Pandas

df = pd.read_csv('/kaggle/input/la-liga-dataset/LaLiga_dataset.csv')
# Sorteando 5 amostras aleatórias para conhecimento do formato do dataset

df.sample(5)
# Informações sobre os atributos (quantidade de valores, tipo do atributo, etc)

df.info()
# Visulizando algumas estatisticas básicas do conjunto de dados

df.describe().round(2)
# Cria uma nova figura

fig, ax1 = plt.subplots(1, 1, figsize=(16, 10))



# Primeiro Plot: Quantidade de Clubes por Temporada

sns.lineplot(x=df.season.unique(), y=df.groupby(by='season').size().values, color='blue', linewidth=3, label='Qtde de Clubes', ax=ax1)

ax1.set_ylabel('Quantidade de Clubes', fontsize=20) # Label do Eixo Y (esq)

ax1.set_xlabel('Temporada', fontsize=20) # Label do Eixo X

ax1.set_xticklabels(labels=ax1.get_xticklabels(), rotation=90); # Comando para rotactionar os ticks do eixo X

ax1.legend(loc=1, bbox_to_anchor=(1, 1), fontsize=20) # Legenda do primeiro gráfico

ax1.grid(alpha=0.3) # Habilita grid com transparência



# Segundo Plot: Quantidade de Jogos por Temporada

ax2 = ax1.twinx() # Cria um segundo eixo espelhado

sns.lineplot(x='season', y='total_matches', color='red', linewidth=3, data=df, label='Qtde de Jogos', ax=ax2)

ax2.set_ylabel('Quantidade de Jogos', fontsize=20) # Label do Eixo Y (dir)

ax2.legend(loc=1, bbox_to_anchor=(1, 0.92), fontsize=20) # Legenda do segundo gráfico



fig.tight_layout() # Comando para ajustar o gráfico automaticamente à área disponível
clubes_campeoes = df.loc[df.groupby('season')['points'].idxmax()].club.value_counts().index

qtde_titulos = df.loc[df.groupby('season')['points'].idxmax()].club.value_counts().values
# Cria uma nova figura

fig, ax = plt.subplots(1, 1, figsize=(12, 8))



# Grafico em Barras Quantidade de Titulos vs Clube

sns.barplot(x=qtde_titulos, y=clubes_campeoes, ax=ax)

ax.set_ylabel('Clube', fontsize=18) # Label Eixo Y

ax.set_xlabel('Quantidade de Títulos', fontsize=18) # Label Eixo X

ax.xaxis.set_major_locator(MaxNLocator(nbins=25, integer=True)) # Forçando exibição de inteiros no Eixo X 

ax.tick_params(axis='both', which='major', labelsize=16) # Aumentando o tamanho dos ticks nos eixos

ax.grid(True, alpha=0.2) # Habilita grid



fig.tight_layout() # Comando para ajustar o gráfico automaticamente à área disponível
# Cria uma nova figura

fig, ax = plt.subplots(1, 1, figsize=(12, 12))



# Grafico Pizza: Clubes Campeões e Proporções

wedges, texts, autotexts = ax.pie(qtde_titulos, autopct='%.0f%%', wedgeprops=dict(width=0.5), textprops={'fontsize': 18}, startangle=-30)



# Configurações das Anotações

bbox_props = dict(boxstyle="round, pad=0.4", fc="white", ec="black", lw=1)

kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")



# Insere as anotações automaticamente no gráfico

for i, p in enumerate(wedges):

    ang = (p.theta2 - p.theta1)/2. + p.theta1

    connectionstyle = "angle,angleA=0,angleB={}".format(ang)

    kw["arrowprops"].update({"connectionstyle": connectionstyle})

    y = np.sin(np.deg2rad(ang))

    x = np.cos(np.deg2rad(ang))

    ax.annotate(clubes_campeoes[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y), fontsize=14, **kw)



# Titulo do gráfico

ax.set_title("Clubes Campeões La Liga", fontsize=20)



fig.tight_layout() # Comando para ajustar o gráfico automaticamente à área disponível
# Cria um novo dataset contendo o menor, o maior e a media de gols marcados por temporada

#gols_marcados = df.groupby('season')['goals_scored'].agg({'Low Value':'min','High Value':'max','Mean':'mean'})

gols_marcados = df.groupby('season')['goals_scored'].agg(['min', 'max','mean'])

gols_marcados.reset_index(inplace=True)



# Cria um novo dataset contendo o menor, o maior e a media de gols sofridos por temporada

#gols_sofridos = df.groupby('season')['goals_conceded'].agg({'Low Value':'min','High Value':'max','Mean':'mean'})

gols_sofridos = df.groupby('season')['goals_conceded'].agg(['min', 'max','mean'])

gols_sofridos.reset_index(inplace=True)



# Cria uma nova figura

fig, ax = plt.subplots(1, 2, figsize=(20, 8), sharey=True)



# Primeiro Plot: Gols Marcados

sns.lineplot(x='season', y='mean', c='red', data=gols_marcados, ax=ax[0], label='Media')

ax[0].fill_between(x='season',y1='min', y2='max', data=gols_marcados)

ax[0].set_title('Análise de gols marcados por temporada', fontsize=18)

ax[0].set_ylabel('Gols', fontsize=20) # Label do Eixo Y (esq)

ax[0].set_xlabel('Temporada', fontsize=20) # Label do Eixo X

ax[0].set_xticklabels(labels=ax1.get_xticklabels(), rotation=90); # Comando para rotactionar os ticks do eixo X

ax[0].tick_params(axis='y', which='major', labelsize=16) # Aumentando o tamanho dos ticks no eixo Y

ax[0].tick_params(axis='y', which='major', labelsize=14) # Aumentando o tamanho dos ticks no eixo X

ax[0].legend(loc=0, fontsize=14) # Legenda

ax[0].grid(alpha=0.3) # Habilita grid com transparência



# Segundo Plot: Gols Sofridos

sns.lineplot(x='season', y='mean', c='red', data=gols_sofridos, ax=ax[1], label='Media')

ax[1].fill_between(x='season',y1='min', y2='max', data=gols_sofridos)

ax[1].set_title('Análise de gols sofridos por temporada', fontsize=18)

ax[1].set_ylabel('Gols', fontsize=20) # Label do Eixo Y (esq)

ax[1].set_xlabel('Temporada', fontsize=20) # Label do Eixo X

ax[1].set_xticklabels(labels=ax1.get_xticklabels(), rotation=90); # Comando para rotactionar os ticks do eixo X

ax[1].tick_params(axis='y', which='major', labelsize=16) # Aumentando o tamanho dos ticks no eixo Y

ax[1].tick_params(axis='y', which='major', labelsize=14) # Aumentando o tamanho dos ticks no eixo X

ax[1].legend(loc=0, fontsize=14) # Legenda

ax[1].grid(alpha=0.3) # Habilita grid com transparência



fig.tight_layout() # Comando para ajustar o gráfico automaticamente à área disponível
gols_marcados = df.loc[df.groupby('season')['points'].idxmax()]['goals_scored']

gols_sofridos = df.loc[df.groupby('season')['points'].idxmax()]['goals_conceded']

saldo_gols = gols_marcados - gols_sofridos

temporadas = df.season.unique()



# Cria uma nova figura

fig, ax = plt.subplots(1, 1, figsize=(16, 10))



# Plot de linhas

sns.lineplot(x=temporadas, y=gols_marcados, color='blue', label='Marcados', ax=ax)

sns.lineplot(x=temporadas, y=gols_sofridos, color='red', label='Sofridos', ax=ax)

sns.lineplot(x=temporadas, y=saldo_gols, color='black', label='Saldo', ax=ax)

ax.set_title('Análise dos gols do clube campeão em cada temporada', fontsize=16)

ax.set_ylabel('Gols', fontsize=20) # Label do Eixo Y (esq)

ax.set_xlabel('Temporada', fontsize=20) # Label do Eixo X

ax.set_xticklabels(labels=ax1.get_xticklabels(), rotation=90); # Comando para rotactionar os ticks do eixo X

ax.tick_params(axis='y', which='major', labelsize=16) # Aumentando o tamanho dos ticks no eixo Y

ax.tick_params(axis='y', which='major', labelsize=14) # Aumentando o tamanho dos ticks no eixo X

ax.legend(loc=0, fontsize=20) # Legenda

ax.grid(alpha=0.3) # Habilita grid com transparência



fig.tight_layout() # Comando para ajustar o gráfico automaticamente à área disponível
cols = ['matches_won', 'matches_lost', 'matches_drawn', 'goals_scored', 'goals_conceded', 'points']

labels = ['Vitorias', 'Derrotas', 'Empates', 'Gols Marcados', 'Gols Sofridos', 'Pontos']

times = ['Barcelona', 'Real Madrid']



# Criando um novo dataframe contedo informações apenas de Barcelona e Real Madrid

df2 = df.groupby('club')[cols].agg(['sum'])

df2.reset_index(inplace=True)

df2 = df2[df2['club'].isin(times)]



# Cria uma nova figura, 2x3 graficos

fig, ax = plt.subplots(2, 3, figsize=(20, 10))

axs = ax.flatten() # Matriz para vetor



# Varre todos os atributos presentes em 'cols'

for i,j in itertools.zip_longest(cols, range(len(cols))):

    atributo = df2[i].values.ravel() 

    sns.barplot(x=atributo, y=times, ax=axs[j], label=labels[j], palette=['darkred','royalblue'], linewidth=2, edgecolor="k"*2)

    

    axs[j].set_title(labels[j], fontsize=18) # Titulo

    axs[j].tick_params(axis='y', which='major', labelsize=16) # Aumentando os ticks

    

    # Escreve o valor de cada caracteristica dentro de cada barra

    for k,l in enumerate(atributo):

        axs[j].text(.7, k, l, fontsize=20)



# Comando para ajustar o gráfico automaticamente à área disponível

fig.tight_layout() 