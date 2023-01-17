# Importando bibliotecas necessárias

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

from matplotlib.gridspec import GridSpec
# Definindo dados para geração de gráfico violino

df_blackfriday = pd.read_csv('../input/dataviz-facens-20182-ex3/BlackFriday.csv')



# Definição dos elementos do gráfico

sns.set(rc={"axes.facecolor":"#e6e6e6",

            'axes.labelsize':20,

            'figure.figsize':(15, 8),

            'xtick.labelsize':14,

            'ytick.labelsize':14})

ax = sns.violinplot(data=df_blackfriday, x='Age', y='Purchase', order=sorted(list(df_blackfriday['Age'].value_counts().index)))

plt.title('Consumo por faixa etária', fontsize=20)

plt.ylabel('Valor Gasto ($)')

plt.xlabel('Faixa etária')

ax.annotate('Fonte: Loja Varejo - Hosted by Analytics Vidhya',

            xy=(0.3, 0), xytext=(0, 0),

            xycoords=('axes fraction', 'figure fraction'),

            textcoords='offset points',

            size=13, ha='right', va='bottom')
# Definindo dados para geração de gráfico de barras

df_produtos = df_blackfriday['Product_ID'].value_counts().head(8).index



# Definição dos elementos do gráfico

sns.set(rc={"axes.facecolor":"#e6e6e6",

            'axes.labelsize':20,

            'figure.figsize':(15, 8),

            'xtick.labelsize':14,

            'ytick.labelsize':14})

ax = sns.countplot(x = 'Product_ID', data = df_blackfriday, order = df_produtos)

plt.title('Produtos mais comprados',fontsize= 20)

plt.xlabel('ID do produto')

plt.ylabel('Quantidade comprada')

ax.annotate('Fonte: Loja Varejo - Hosted by Analytics Vidhya',

            xy=(0.3, 0), xytext=(0, 0),

            xycoords=('axes fraction', 'figure fraction'),

            textcoords='offset points',

            size=13, ha='right', va='bottom')
# Definindo dados para geração de gráfico violino

list_ocupacoes = list(df_blackfriday['Occupation'].value_counts().index)[:5]



# Idade 00-17 - Filtro ocupações mais frequentes

df_00_17 = df_blackfriday[df_blackfriday['Age'] == "0-17"]

df_00_17 = df_00_17[df_00_17['Occupation'].isin(list_ocupacoes)]



# Idade 18-25 - Filtro ocupações mais frequentes

df_18_25 = df_blackfriday[df_blackfriday['Age'] == "18-25"]

df_18_25 = df_18_25[df_18_25['Occupation'].isin(list_ocupacoes)]



# Idade 26-35 - Filtro ocupações mais frequentes

df_26_35 = df_blackfriday[df_blackfriday['Age'] == "26-35"]

df_26_35 = df_26_35[df_26_35['Occupation'].isin(list_ocupacoes)]



# Idade 36-45 - Filtro ocupações mais frequentes

df_36_45 = df_blackfriday[df_blackfriday['Age'] == "36-45"]

df_36_45 = df_36_45[df_36_45['Occupation'].isin(list_ocupacoes)]



# Idade 46-50 - Filtro ocupações mais frequentes

df_46_50 = df_blackfriday[df_blackfriday['Age'] == "46-50"]

df_46_50 = df_46_50[df_46_50['Occupation'].isin(list_ocupacoes)]



# Idade 51-55 - Filtro ocupações mais frequentes

df_51_55 = df_blackfriday[df_blackfriday['Age'] == "51-55"]

df_51_55 = df_51_55[df_51_55['Occupation'].isin(list_ocupacoes)]



# Idade maior que 55 - Filtro ocupações mais frequentes

df_56_99 = df_blackfriday[df_blackfriday['Age'] == "55+"]

df_56_99 = df_56_99[df_56_99['Occupation'].isin(list_ocupacoes)]
# Definição dos elementos do gráfico

def mostra_grid(grid, df_dados, list_ocupacoes, titulo, rodape):

    sns.violinplot(data=df_dados, x='Occupation', y='Purchase', order=sorted(list_ocupacoes))

    sns.set(rc={"axes.facecolor":"#e6e6e6",

                'axes.labelsize':20,

                'figure.figsize':(50, 40),

                'xtick.labelsize':14,

                'ytick.labelsize':14})

    plt.title(titulo, fontsize=20)

    plt.ylabel('Valor Gasto ($)')

    if rodape == False:

        plt.xlabel('')

    else:

        plt.xlabel('Ocupação')

    return ''



plt.clf()

fig = plt.figure()

gs = GridSpec(nrows=7, ncols=1, figure=fig, left=0.05, right=0.48, wspace=0.05)

mostra_grid(fig.add_subplot(gs[0, :]), df_00_17, list_ocupacoes, 'Consumo 5 Ocupações mais frequentes e Faixa Etária até 17 anos', False)

mostra_grid(fig.add_subplot(gs[1, :]), df_18_25, list_ocupacoes, 'Consumo 5 Ocupações mais frequentes e Faixa Etária entre 18 e 25 anos', False)

mostra_grid(fig.add_subplot(gs[2, :]), df_26_35, list_ocupacoes, 'Consumo 5 Ocupações mais frequentes e Faixa Etária entre 26 e 35 anos', False)

mostra_grid(fig.add_subplot(gs[3, :]), df_36_45, list_ocupacoes, 'Consumo 5 Ocupações mais frequentes e Faixa Etária entre 36 e 45 anos', False)

mostra_grid(fig.add_subplot(gs[4, :]), df_46_50, list_ocupacoes, 'Consumo 5 Ocupações mais frequentes e Faixa Etária entre 46 e 50 anos', False)

mostra_grid(fig.add_subplot(gs[5, :]), df_51_55, list_ocupacoes, 'Consumo 5 Ocupações mais frequentes e Faixa Etária entre 51 e 55 anos', False)

mostra_grid(fig.add_subplot(gs[6, :]), df_56_99, list_ocupacoes, 'Consumo 5 Ocupações mais frequentes e Faixa Etária maior que 55 anos', True)

plt.annotate('Fonte: Loja Varejo - Hosted by Analytics Vidhya',

            xy=(0.3, 0), xytext=(0, 0),

            xycoords=('axes fraction', 'figure fraction'),

            textcoords='offset points',

            size=13, ha='right', va='bottom')

plt.show()
# Definindo dados para geração de gráfico de barras

df_maior_9000   = df_blackfriday[df_blackfriday['Purchase'] > 9000]



df_maior_9000_0 = df_maior_9000[df_maior_9000['Marital_Status'] == 0]

df_maior_9000_0 = df_maior_9000_0[['Occupation']]

df_maior_9000_0 = df_maior_9000_0.groupby(['Occupation']).size().reset_index()



df_maior_9000_1 = df_maior_9000[df_maior_9000['Marital_Status'] == 1]

df_maior_9000_1 = df_maior_9000_1[['Occupation']]

df_maior_9000_1 = df_maior_9000_1.groupby(['Occupation']).size().reset_index()



df_ocupacoes  = list(df_blackfriday['Occupation'].value_counts().index)

df_ocupacoes  = sorted(df_ocupacoes)



# Definição dos elementos do gráfico

bars1 = list(df_maior_9000_0[0])

bars2 = list(df_maior_9000_1[0])



barWidth = 0.25

r1    = df_ocupacoes

r2    = [x + barWidth for x in df_ocupacoes]



bars     = np.add(list(df_maior_9000_0[0]), list(df_maior_9000_1[0])).tolist()

plt.clf()

plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Estado Civil 0')

plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Estado Civil 1')



plt.title('Relação entre Ocupação x Estado Civil para compras acima de $ 9000 ', fontsize=50)

plt.xlabel('Ocupação', fontsize=40)

plt.ylabel('Quantidade de Compras', fontsize=40)

plt.yticks(fontsize=30)

plt.xticks(np.arange(np.array(r1).min(), np.array(r1).max()+1, 1), fontsize=30)

plt.legend(fontsize=50)

plt.annotate('Fonte: Loja Varejo - Hosted by Analytics Vidhya',

            xy=(0.3, 0), xytext=(0, 0),

            xycoords=('axes fraction', 'figure fraction'),

            textcoords='offset points',

            size=30, ha='right', va='bottom')

plt.show()