import pandas as pd

resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
# Imports

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import math as math
df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/anv.csv', delimiter=',')

df.head()
df[[

    'aeronave_operador_categoria',

    'aeronave_pmd_categoria', 

    'aeronave_modelo',  

    'aeronave_tipo_veiculo', 

    'aeronave_assentos', 

    'aeronave_ano_fabricacao',

    'total_fatalidades',

]].head()
classificacao = [["aeronave_operador_categoria", "Qualitativa Nominal"],

            ["aeronave_pmd_categoria","Qualitativa Nominal"],

            ["aeronave_modelo","Qualitativa Nominal"],

            ["aeronave_tipo_veiculo","Qualitativa Nominal"],

            ["aeronave_motor_tipo","Qualitativa Nominal"], 

            ["aeronave_ano_fabricacao","Quantitativa Discreta"], 

            ["total_fatalidades","Quantitativa Discreta"]]



classificacao = pd.DataFrame(classificacao, columns=["Variavel", "Classificação"])

classificacao
op_categoria = pd.DataFrame(df["aeronave_operador_categoria"].value_counts())

op_categoria
pmd_cat = pd.DataFrame(df["aeronave_pmd_categoria"].value_counts())

pmd_cat
modelo = pd.DataFrame(df["aeronave_modelo"].value_counts())

modelo
tipo_veiculo = pd.DataFrame(df["aeronave_tipo_veiculo"].value_counts())

tipo_veiculo
motor_tipo = pd.DataFrame(df["aeronave_motor_tipo"].value_counts())

motor_tipo
df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/anv.csv', delimiter=',')

df.head(1)
ax = df["aeronave_operador_categoria"].value_counts().plot(kind='barh', figsize=(15,7),

                                        color="steelblue", fontsize=13);

ax.set_alpha(0.8)

ax.set_title("Distribuição de Operadores de voo por categoria.", fontsize=18)

ax.set_xticks([])

# create a list to collect the plt.patches data

totals = []



# find the values and append to list

for i in ax.patches:

    totals.append(i.get_width())



# set individual bar lables using above list

total = sum(totals)



# set individual bar lables using above list

for i in ax.patches:

    # get_width pulls left or right; get_y pushes up or down

    ax.text(i.get_width()+.3, i.get_y()+.38, \

            str(round((i.get_width()), 2)), fontsize=15,

color='dimgrey')



# invert for largest on top 

ax.invert_yaxis()
labels = df["aeronave_pmd_categoria"].unique()

pmd_cat = df["aeronave_pmd_categoria"].value_counts()





x = np.arange(len(labels))  # the label locations

width = 0.7  # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(x, pmd_cat, width,)





# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Quantidade de ocorrências')

ax.set_xlabel('Categoria de peso maximo de decolagem')

ax.set_title('Ocorrências X Categoria de peso maximo de decolagem.')

ax.set_xticks(x)

ax.set_yticks([])

ax.set_xticklabels(labels)

ax.legend()





def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')





autolabel(rects1)





fig.tight_layout()

plt.margins(0.1)

plt.show()
modelo = pd.DataFrame(df["aeronave_modelo"].value_counts())

top5_modelos = modelo.head(5)







ax = top5_modelos.plot(kind='barh', figsize=(15,7),

                                        color="teal", fontsize=13);

ax.set_alpha(0.8)

ax.set_title("Top 5 modelos que mais registraram ocorrências.", fontsize=18)

ax.set_xticks([])

# create a list to collect the plt.patches data

totals = []



# find the values and append to list

for i in ax.patches:

    totals.append(i.get_width())



# set individual bar lables using above list

total = sum(totals)



# set individual bar lables using above list

for i in ax.patches:

    # get_width pulls left or right; get_y pushes up or down

    ax.text(i.get_width()+.3, i.get_y()+.38, \

            str(round((i.get_width()), 2)), fontsize=15,

color='dimgrey')



# invert for largest on top 

ax.invert_yaxis()
tipo_veiculo = df['aeronave_tipo_veiculo'].value_counts().reset_index().head()



with plt.style.context('default'):

    plt.figure(figsize=(10, 5))

    plt.pie(x=tipo_veiculo['aeronave_tipo_veiculo'])

    plt.title('Tipo de aeronave')

    plt.legend(labels=tipo_veiculo['index'], bbox_to_anchor=(1, 1))

    

    plt.show()
motor_tipo = df["aeronave_motor_tipo"].value_counts().reset_index().head(10)



plt.figure(figsize=(15, 8))

plt.bar(motor_tipo['index'], motor_tipo["aeronave_motor_tipo"], color='darkslategray')

plt.title('Tipo do Motor')

plt.xlabel('Motor')

plt.ylabel('Quantidade')

plt.show()
#função para excluir os registros com ano de fabricação 0

def clean_df(df):

    df.dropna(inplace=True)

    return df[(df[['aeronave_ano_fabricacao']] != 0).all(axis=1)]



df_ano = clean_df(df)



ax= df_ano["aeronave_ano_fabricacao"].hist(bins=40, figsize=(12,10))

ax.set_xlabel('Ano')

ax.set_ylabel('Quantidade de aeronaves')

ax.set_title('Quantidade de aeronaves por ano de fabricação.',fontsize = 20)



total_fatalidade = df['total_fatalidades'].sum()

numero_registros = df['codigo_ocorrencia'].count()



data = {

    'Label' : ['Ocorrencias','Fatalidades'],

'Total': [numero_registros,total_fatalidade]

}

#Criando o DataFrame

tab_fatalidades = pd.DataFrame(data)



total = tab_fatalidades["Total"]

labels = tab_fatalidades['Label'].unique()





x = np.arange(len(labels))  # the label locations

width = 0.7  # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(x, total, width,)







# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Quantidade')

ax.set_title('Total de ocorrências X total de Fatalidades.',fontsize=18)

ax.set_xticks(x)

ax.set_yticks([])

ax.set_xticklabels(labels)





def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')





autolabel(rects1)







fig.tight_layout()

plt.margins(0.1)

plt.show()
