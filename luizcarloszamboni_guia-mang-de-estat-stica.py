# todas as libs usadas neste notebook

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import math
# exibe todos os arquivos e diretórios dentro de /kaggle/input

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# TODO - passar isso para um csv

questionario_df = df = pd.DataFrame({

    'nome': ["luy", "A", "B", "C", "D", "E", "F", "G", "H", "I"],

    "opinião":   ["muito divertida", "divertida", "mais ou menos", "um pouco chata", "divertida", "muito chata", "muito divertida", "divertida", "mais ou menos", "mais ou menos"],

    "sexo": ["feminino", "feminino", "masculino", "masculino", "feminino", "masculino", "feminino", "feminino", "masculino", "feminino"],

    "idade": [ 17, 17, 18, 22, 25,20 ,16, 17, 18 , 21],

    "revistas por mês": [ 2, 1,5, 7,4,3,1,2,0,3]

})



questionario_df.head()
# TODO - passar para um csv

restaurantes_df = pd.DataFrame([

        700, 850, 600, 650, 980,

        750, 500, 890, 880, 700, 

        890, 720, 680, 650, 790, 

        670, 680, 900, 880, 720, 

        850, 700, 780, 850, 750, # primeiros 25

        780, 590, 650, 580, 750,

        800, 550, 750, 700, 600, 

        800, 800, 880, 790, 790,

        780, 600, 670, 680, 650, 

        890, 930, 650, 777, 700  # até o 50

    ], 

    columns=["preço"],

)



def split_classes(v):

    if v >= 500 and v < 600:

        return "1p"

    elif v >= 600 and v < 700:

        return "2p"

    elif v >= 700 and v < 800:

        return "3p"

    elif v >= 800 and v < 900:

        return "4p"

    elif v >= 900 and v < 1000:

        return "5p"

    

# separação por classes

restaurantes_df["classe"] = restaurantes_df["preço"].map(split_classes)

restaurantes_df
classes_series = restaurantes_df.groupby('classe')['classe'].count()

classes_means_series = restaurantes_df.groupby('classe').mean()



# o retorno é um tipo Pandas.Series

# print(type(classes_series))



def frequencia_absoluta(classe):

    return classes_series[classe]



def media_da_classe(classe):

    return round(

        classes_means_series['preço'][classe]

    )





classes_means_series['preço']
classes_df = pd.DataFrame({ 'frequencia':  classes_series })



classes_df['frequencia relativa'] = classes_df["frequencia"].map(lambda v: v/classes_series.sum() * 100)

# o campo classe é indice deste dataframe, portanto o processo é um pouco diferente

classes_df['media da classe'] = classes_df.index.to_series().map(media_da_classe)



classes_df
absolute_axis = classes_df.plot(kind = 'bar', y = 'frequencia', x = 'media da classe' )

absolute_axis.set_ylabel('frequência absoluta')



relative_axis = classes_df.plot(kind= 'bar', y = 'frequencia relativa', x = 'media da classe' )

relative_axis.set_ylabel('frequência em %')
# frequencia de cada grupo



restaurantes_df["frequencia"] = restaurantes_df['classe'].map(frequencia_absoluta)



restaurantes_df.head(10)
# TODO - passar isso para um csv

notas_df = pd.DataFrame({

    'nome': ["luy",  "Yumi", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"],

    "inglês": [ 90, 81, 73, 97, 85, 60, 74, 64, 72, 67, 87, 78, 85, 96, 77, 100, 92, 86],

    "japonês": [ 71, 90, 79, 70, 67, 66, 60, 83, 57, 85, 93, 89, 78, 74, 65, 78, 53, 80], 

    "história": [ 73, 61, 14, 41, 49, 87, 69, 65, 36, 7, 53, 100, 57, 45, 56, 34, 37, 70 ],

    "biologia": [ 59, 73, 47 , 38, 63, 56, 15, 53, 80, 50, 41, 62, 44, 26, 91, 35, 53, 68], 

})

notas_df
axis = notas_df["biologia"].plot.hist(bins=15)

axis.set_ylabel('frequência')

axis
# esta função serve apenas para "traduzir"

def desvio_padrao(series):

    return round(series.std(), 2)



# esta função serve apenas para "traduzir"

def media(series):

    return round(series.mean(), 2)



def valor_padrao(valor, column_name):

    return round((valor - media(notas_df[column_name])) / desvio_padrao(notas_df[column_name]), 2)



def valor_do_desvio(valor, column_name): 

    return valor_padrao(valor, column_name) * 10 + 50
# TODO - passar isso para um csv

materias = pd.DataFrame({

    "materias": [ "inglês", "japonês", "história", "biologia"],

})



materias["médias"] = materias["materias"].map(lambda v: media(notas_df[v]))

materias["desvio padrao"] = materias["materias"].map(lambda v: desvio_padrao(notas_df[v]))



materias


# TODO - passar isso para um csv

notas_valor_padrao = pd.DataFrame({

    "nome": notas_df["nome"],

    "inglês valor padrão":   list(notas_df["inglês"].map(lambda value: valor_padrao(value, "inglês" ))),

    "japonês valor padrão":  list(notas_df["japonês"].map(lambda value: valor_padrao(value, "japonês" ))),

    "história valor padrão": list(notas_df["história"].map(lambda value: valor_padrao(value, "história" ))),

    "biologia valor padrão": list(notas_df["biologia"].map(lambda value: valor_padrao(value, "biologia" ))),

})



notas_valor_padrao
# TODO - passar isso para um csv

notas_valor_do_desvio = pd.DataFrame({

    "nome": notas_df["nome"],

    "inglês valor do desvio":   list(notas_df["inglês"].map(lambda value: valor_do_desvio(value, "inglês" ))),

    "japonês valor do desvio":  list(notas_df["japonês"].map(lambda value: valor_do_desvio(value, "japonês" ))),

    "história valor do desvio": list(notas_df["história"].map(lambda value: valor_do_desvio(value, "história" ))),

    "biologia valor do desvio": list(notas_df["história"].map(lambda value: valor_do_desvio(value, "história" ))),

})



notas_valor_do_desvio
# TODO - passar isso para um csv

gastos = pd.DataFrame({

    "nome": list(map(lambda v: "Sra. {}".format(v), ["A", "B", "C", "D", "E" , "F", "G", "H", "I", "J"] )),

    "gasto com maquiagem": [3000, 5000, 12000, 2000, 7000, 15000, 5000, 6000, 8000, 10000 ],

    "gasto com roupas":  [7000, 8000, 25000, 5000, 12000, 30000, 10000, 15000, 20000, 18000],

})



gastos
# correlação entre cada tipo de dados



# numéricos e numéricos

def numeric_numeric_correlation(series_x, series_y):

    return series_x.mean() + series_y.mean()



numeric_numeric_correlation(gastos["gasto com maquiagem"], gastos["gasto com roupas"])

correlation_data = pd.DataFrame({

    "nome": gastos["nome"],

    "x - xm": list(map(lambda v: v - gastos["gasto com maquiagem"].mean(),gastos["gasto com maquiagem"])),

    "y - ym": list(map(lambda v: v - gastos["gasto com roupas"].mean(),gastos["gasto com roupas"]))

})



correlation_data['(x - xm)²'] = [ x**2 for x in correlation_data['x - xm'] ]

correlation_data['(y - ym)²'] = [ y**2 for y in correlation_data['y - ym'] ]

correlation_data['(x - xm)(y - ym)'] =  [ x*y for x,y in zip(correlation_data['x - xm'] , correlation_data['y - ym']) ]



Sxy = correlation_data['(x - xm)(y - ym)'].sum()

Sxx = correlation_data['(x - xm)²'].sum()

Syy = correlation_data['(y - ym)²'].sum()

print("correlation is:", Sxy/(math.sqrt(Sxx * Syy)))
gastos.plot.scatter(y='gasto com maquiagem', x="gasto com roupas", )
marcas_dict = [

    "Theremes" ,

    "Channelior",

    "Bureperry"

]





idade_e_grife = pd.DataFrame({

    "Entrevistada": [ "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O" ],

    "Idade": [ 27, 33, 16, 29, 32, 23, 25, 28, 22, 18, 26, 26, 15, 29, 26 ],

    "Marca": [ 

        0, 1, 2, 2, 1, 0, 1, 

        0, 2, 2, 1, 0, 2, 1, 2

    ]

})

idade_e_grife
# usando mathplot para ter mais flexibilidade

plt.plot(idade_e_grife["Marca"], idade_e_grife["Idade"], 'ro')

plt.xticks(idade_e_grife["Marca"], marcas_dict, rotation='vertical')



plt.margins(0.2)

plt.subplots_adjust(bottom=0.15)

plt.show()
def taxa_de_correlacao_numerico_categorico(serie):

    return sum([ (valor - serie.mean())**2 for valor in serie ])

    

idades_theremes = idade_e_grife[idade_e_grife['Marca'] == marcas_dict.index('Theremes')]['Idade']

idades_channelior = idade_e_grife[idade_e_grife['Marca'] == marcas_dict.index('Channelior')]['Idade']

idades_bureperry = idade_e_grife[idade_e_grife['Marca'] == marcas_dict.index('Bureperry')]['Idade']





tt = taxa_de_correlacao_numerico_categorico(idades_theremes)

tc = taxa_de_correlacao_numerico_categorico(idades_channelior)

tb = taxa_de_correlacao_numerico_categorico(idades_bureperry)
variacao_intraclasse = tt + tc + tb

variacao_intraclasse
todas_as_idades = idade_e_grife['Idade']



variacao_interclasse = sum([

    (len(idades_theremes) * (idades_theremes.mean() - todas_as_idades.mean())**2),

    (len(idades_channelior) * (idades_channelior.mean() - todas_as_idades.mean())**2),

    (len(idades_bureperry) * (idades_bureperry.mean() - todas_as_idades.mean())**2)

])



variacao_interclasse
taxa_de_correlacao = variacao_interclasse/(variacao_intraclasse + variacao_interclasse)

taxa_de_correlacao
# etapa 1

cross_table_gender_invite = pd.DataFrame({

        'Telefone': [ 34, 38 ],

        'Email': [ 61, 40 ],

        'Pessoalmente': [ 53, 74  ],

    },

        index = ["feminino", "masculino"], 

)

cross_table_gender_invite
cross_table_somatorio = sum([sum(cross_table_gender_invite['Pessoalmente']),

    sum(cross_table_gender_invite['Email']),

    sum(cross_table_gender_invite['Telefone'])])





def frequencia_esperada(nome_linha, nome_coluna): 

    return sum(cross_table_gender_invite.loc[nome_linha]) * sum(cross_table_gender_invite[nome_coluna]) / cross_table_somatorio
frequencia_esperada('feminino', 'Pessoalmente')

# etapa 2

cross_table_gender_invite_frequencia_esperada = pd.DataFrame({

        'Telefone': [ frequencia_esperada('feminino', 'Telefone'), frequencia_esperada('masculino', 'Telefone')  ],

        'Email': [ frequencia_esperada('feminino', 'Email'), frequencia_esperada('masculino', 'Email') ],

        'Pessoalmente': [ frequencia_esperada('feminino', 'Pessoalmente'), frequencia_esperada('masculino', 'Pessoalmente')  ],

    },

        index = ["feminino", "masculino"], 

)

cross_table_gender_invite_frequencia_esperada
# etapa 3

def etapa_3(nome_linha, nome_coluna):

    return (cross_table_gender_invite.loc[nome_linha, nome_coluna] - frequencia_esperada(nome_linha, nome_coluna))**2 / frequencia_esperada(nome_linha, nome_coluna)



teste_qui_quadrado = sum([ 

      etapa_3('feminino', 'Telefone'), etapa_3('feminino', 'Email'), etapa_3('feminino', 'Pessoalmente'), 

      etapa_3('masculino', 'Telefone'),etapa_3('masculino', 'Email'), etapa_3('masculino', 'Pessoalmente'),

    ])



teste_qui_quadrado
# etapa 5

numero_de_linhas = 2

numero_de_colunas = 3

coeficiente_de_cramer = math.sqrt(

    (teste_qui_quadrado/cross_table_somatorio * (min(numero_de_linhas, numero_de_colunas) - 1))# )

)



coeficiente_de_cramer