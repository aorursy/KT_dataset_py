# Bibliotecas utilizadas

import pandas as pd 

from operator import itemgetter

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
# Leitura do dataset Black Friday]

df_blackfriday = pd.read_csv('../input/dataviz-facens-20182-ex3/BlackFriday.csv') 

print(df_blackfriday.info())

print(df_blackfriday.describe())
## Rotinas gerais



# Cálculo da frequência de valores e identificação dos valores não preenchidos

def frequencia_coluna(df, coluna, qtd_linhas):    

    list_value = []    

    list_freq_absoluta = df[coluna].value_counts() 

    qtd_tot_valores = 0

    for item in list_freq_absoluta.iteritems():

        list_value.append([str(item[0]), item[1]])

        qtd_tot_valores = qtd_tot_valores + item[1] 

    qtd_nao_inf = qtd_linhas - qtd_tot_valores

    list_value.append(['Não disponível', qtd_nao_inf])

    list_value = sorted(list_value,key=itemgetter(0)) 

    df = pd.DataFrame(list_value)

    df = df.rename(columns={0:'valor', 1:'freq'}) 

    return df



# Identificação dos valores não preenchidos

def valores_zerados(df, coluna):

    list_value = []    

    list_freq_absoluta = df_blackfriday[coluna].value_counts() 

    qtd_tot_valores = 0

    for item in list_freq_absoluta.iteritems():

        qtd_tot_valores = qtd_tot_valores + item[1] 

    list_value.append(['Disponível', qtd_tot_valores])

    qtd_nao_inf = qtd_linhas - qtd_tot_valores

    list_value.append(['Não disponível', qtd_nao_inf])

    list_value = sorted(list_value,key=itemgetter(0)) 

    df = pd.DataFrame(list_value)

    df = df.rename(columns={0:'valor', 1:'freq'}) 

    return df



# Mostra fonte dos dados

def mostra_fonte(plt):

    plt.annotate('Fonte: Loja Varejo - Hosted by Analytics Vidhya',

                xy=(0.3, 0), xytext=(0, 0),

                xycoords=('axes fraction', 'figure fraction'),

                textcoords='offset points',

                size=13, ha='right', va='bottom')  

    

# Apresentação de gráfico de barras

def mostra_bargraph(df, coluna_x, coluna_y, titulo, descricao_x, descricao_y):

    df = df.sort_values(by=[coluna_x])

    plt.figure(figsize=(15,6.5))

    graph = sns.barplot(x=coluna_x, y=coluna_y, data=df)

    for p in graph.patches:

        graph.annotate(format(p.get_height(), '.0f'), 

                       (p.get_x() + p.get_width() / 2., p.get_height()), 

                       ha = 'center',

                       va = 'center', 

                       xytext = (0, 10), 

                       textcoords = 'offset points') 

    plt.title(titulo, fontsize=16)

    plt.xlabel(descricao_x, fontsize=14)

    plt.ylabel(descricao_y, fontsize=14)

    mostra_fonte(plt)

    plt.show()    

    

# Apresentação de gráfico de dispersão

def mostra_boxplot(df, coluna_x, coluna_y, titulo, descricao_x, descricao_y):

    df = df.sort_values(by=[coluna_x])

    plt.figure(figsize=(15,6.5))    

    sns.set_style("whitegrid")

    sns.boxplot(x=coluna_x, y=coluna_y, data=df) #, order=list(sorted_nb.index))

    plt.title(titulo, fontsize=16)

    plt.xlabel(descricao_x, fontsize=14)

    plt.ylabel(descricao_y, fontsize=14)

    mostra_fonte(plt)

    plt.show()

    

def mostra_heatmap(df, titulo):

    plt.figure(figsize=(25,7))    

    sns.heatmap(df, annot=True, annot_kws={"size": 12})

    plt.title(titulo, fontsize=16)

    mostra_fonte(plt)

    plt.show()

    

def mostra_scatter(df, coluna_x, coluna_y, coluna_s, titulo, descricao_x, descricao_y):

    plt.figure(figsize=(25,7)) 

    df2 = df.groupby([coluna_x,coluna_y]).count()

    for column in df2:

        if column != coluna_x and column != coluna_y and column != coluna_s:

            df2.drop(column, axis=1, inplace=True)

    df2 = df2.reset_index()

    plt.scatter(x=df2[coluna_x], y=df2[coluna_y], s=df2[coluna_s]/25)

    plt.title(titulo, fontsize=16)

    plt.xlabel(descricao_x, fontsize=14)

    plt.ylabel(descricao_y, fontsize=14)  

    mostra_fonte(plt)

    plt.show() 

    

# Classificação das colunas

def classificacao(codigo): 

    if codigo == 'QD':

        return 'Quantitativa Descritiva'

    elif codigo == 'QC':

        return 'Quantitativa Contínua'

    elif codigo == 'QN':

        return 'Qualitativa Nominal'

    elif codigo == 'QO':

        return 'Qualitativa Ordinal'

    

# Cálculo das frequências absolutas e relativas

def classificacao_coluna(coluna, classif, qtd_linhas):

    classif_ret = classificacao(classif)



    list_value = []    

    if qtd_linhas != 0:                                               # Calcular frequências absolutas e relativas

        qtd_tot_valores = 0

        list_freq_absoluta = df_blackfriday[coluna].value_counts() 

        for item in list_freq_absoluta.iteritems():

            list_value.append([item[0], item[1], '{:.3%}'.format(float(item[1] / qtd_linhas))])

            qtd_tot_valores = qtd_tot_valores + item[1] 

        if qtd_tot_valores != qtd_linhas:                             # se diferente, existem dados não informados

            qtd_nao_inf = qtd_linhas - qtd_tot_valores

            list_value.append(['N/A', qtd_nao_inf, '{:.3%}'.format(qtd_nao_inf / qtd_linhas)])

    else:                                                             # Não calcular frequências

        list_value.append([0, 0, ''])

    return [coluna, classif_ret, list_value]
# Descrição do Dataset

# estrutura saída: lista(nome coluna, classificação, lista(valor, frequência absoluta, frequência relativa))

qtd_linhas = len(df_blackfriday)                                  

colunas = [classificacao_coluna('User_ID',                   'QN', qtd_linhas),              # QN - Qualitativa Nominal: característica não numérica sem ordem entre os valores             

           classificacao_coluna('Product_ID',                'QN', qtd_linhas),              # QO - Qualitativa Ordinal: característica não numérica com ordem entre os valores

           classificacao_coluna('Gender',                    'QN', qtd_linhas),              # QD - Quantitativa Discreta: conjunto finito ou enumerável de números,

           classificacao_coluna('Age',                       'QO', 0         ),              #      e que resultam de uma contagem 

           classificacao_coluna('Occupation',                'QN', qtd_linhas),              # QC - Quantitativa Contínua: valor contido num intervalo de números reais

           classificacao_coluna('City_Category',             'QN', qtd_linhas),               

           classificacao_coluna('Stay_In_Current_City_Years','QO', 0         ),

           classificacao_coluna('Marital_Status',            'QN', qtd_linhas),

           classificacao_coluna('Product_Category_1',        'QN', qtd_linhas),

           classificacao_coluna('Product_Category_2',        'QN', qtd_linhas),

           classificacao_coluna('Product_Category_3',        'QN', qtd_linhas),

           classificacao_coluna('Purchase',                  'QC', 0         )]

colunas
# Colunas consideradas para análise:

# Gender / Age / Stay_In_Current_City_Years / Marital_Status / Purchase

    

# Cálculo das frequências de cada coluna verificando se alguma não foi preenchida

qtd_linhas = len(df_blackfriday)

df_Gender                          = frequencia_coluna(df_blackfriday, 'Gender',                     qtd_linhas)

df_Age                             = frequencia_coluna(df_blackfriday, 'Age',                        qtd_linhas)

df_Stay_In_Current_City_Yearsprint = frequencia_coluna(df_blackfriday, 'Stay_In_Current_City_Years', qtd_linhas)

df_Marital_Status                  = frequencia_coluna(df_blackfriday, 'Marital_Status',             qtd_linhas)



# Verificando se todos os valores de compras estão preenchidos

df_Purchase                        = valores_zerados(df_blackfriday,   'Purchase')



# Apresentação do gráfico de barras com os dados encontrados

print('Número total de linhas do dataset origem:', qtd_linhas)

mostra_bargraph(df_Gender,                          'valor', 'freq', 'Qtd. Vendas por Gênero',                      'Gênero',            'Número Vendas')

mostra_bargraph(df_Age,                             'valor', 'freq', 'Qtd. Vendas por Faixa de Idade',              'Faixa Idade',       'Número Vendas')

mostra_bargraph(df_Stay_In_Current_City_Yearsprint, 'valor', 'freq', 'Qtd. Vendas por Tempo permanência na cidade', 'Tempo permanência', 'Número Vendas')

mostra_bargraph(df_Marital_Status,                  'valor', 'freq', 'Qtd. Vendas por Estado Civil',                'Estado Civil',      'Número Vendas')

mostra_bargraph(df_Purchase,                        'valor', 'freq', 'Valores gastos em cada compra',               'Informação',        'Número Vendas')
# Cálculo do ticket médio

df_ticket_medio_idade    = pd.DataFrame(df_blackfriday.groupby("Age")["Purchase"].sum()                        / df_blackfriday.groupby("Age")["Purchase"].count()).reset_index()

df_ticket_medio_genero   = pd.DataFrame(df_blackfriday.groupby("Gender")["Purchase"].sum()                     / df_blackfriday.groupby("Gender")["Purchase"].count()).reset_index()

df_ticket_medio_tempo    = pd.DataFrame(df_blackfriday.groupby("Stay_In_Current_City_Years")["Purchase"].sum() / df_blackfriday.groupby("Stay_In_Current_City_Years")["Purchase"].count()).reset_index()

df_ticket_medio_estcivil = pd.DataFrame(df_blackfriday.groupby("Marital_Status")["Purchase"].sum()             / df_blackfriday.groupby("Marital_Status")["Purchase"].count()).reset_index()



# Apresentação do gráfico de barras com os dados encontrados

mostra_bargraph(df_ticket_medio_idade,    'Age',                        'Purchase', 'Vr.Ticket Médio / Faixa Etária',    'Faixa Etária',    '$ Ticket Médio')

mostra_bargraph(df_ticket_medio_genero,   'Gender',                     'Purchase', 'Vr.Ticket Médio / Gênero',          'Gênero',          '$ Ticket Médio')

mostra_bargraph(df_ticket_medio_tempo,    'Stay_In_Current_City_Years', 'Purchase', 'Vr.Ticket Médio / Tempo na cidade', 'Tempo na cidade', '$ Ticket Médio')

mostra_bargraph(df_ticket_medio_estcivil, 'Marital_Status',             'Purchase', 'Vr.Ticket Médio / Estado Civil',    'Estado Civil',    '$ Ticket Médio')
# Gráfico de caixa (boxplot)

mostra_boxplot(df_blackfriday, 'Gender',                     'Purchase', 'Distribuição de valores gastos x gênero',          'Gênero',          '$ Valor Gasto')

mostra_boxplot(df_blackfriday, 'Age',                        'Purchase', 'Distribuição de valores gastos x faixa etária',    'Faixa Etária',    '$ Valor Gasto')

mostra_boxplot(df_blackfriday, 'Stay_In_Current_City_Years', 'Purchase', 'Distribuição de valores gastos x tempo na cidade', 'Tempo na Cidade', '$ Valor Gasto')

mostra_boxplot(df_blackfriday, 'Marital_Status',             'Purchase', 'Distribuição de valores gastos x estado civil',    'Estado Cibil',    '$ Valor Gasto')
# Mapa de Calor (Heatmap)

mostra_heatmap(df_blackfriday.corr(), 'Cálculo de correlação entre as colunas do dataset')

# Gráfico de bolha (Bubble Plot)

mostra_scatter(df_blackfriday, 'Product_Category_1',        'Age',            'Purchase', 'Volume de Vendas por Idade x Categoria de Produtos','Categoria de Produtos', 'Faixa Etária')

mostra_scatter(df_blackfriday, 'Gender',                    'Age',            'Purchase', 'Volume de Vendas por Idade x Gênero',               'Gênero',                'Faixa Etária')

mostra_scatter(df_blackfriday, 'Gender',                    'Marital_Status', 'Purchase', 'Volume de Vendas por Estado Civil x Gênero',        'Gênero',                'Estado Civil')

mostra_scatter(df_blackfriday, 'Stay_In_Current_City_Years','Marital_Status', 'Purchase', 'Volume de Vendas por Estado Civil x Tempo Cidade',  'Tempo na Cidade',       'Estado Civil')