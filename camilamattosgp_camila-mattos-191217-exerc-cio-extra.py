import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/dataviz-facens-20182-ex3/BlackFriday.csv', delimiter=',')

df.dataframeName = 'BlackFriday.csv'





#Substituindo os rótulos das colunas para melhor entendimento

colunas = {

    'User_ID': 'IdUsuario',

    'Product_ID': 'IdProduto',

    'Gender':'Genero',

    'Age':'Idade',

    'Occupation':'Ocupacao',

    'City_Category':'Categoria_Cidade',

    'Stay_In_Current_City_Years':'Anos_Permanencia_Cidade',

    'Marital_Status':'Estado_Civil',

    'Product_Category_1':'Categoria_Produto_1',

    'Product_Category_2':'Categoria_Produto_2',

    'Product_Category_3':'Categoria_Produto_3',

    'Purchase':'Valor_Compra'

}

df = df.rename(columns = colunas)

df.head()
classificacao = [["IdUsuario", "Qualitativa Nominal"],

            ["IdProduto","Qualitativa Nominal"],

            ["Genero","Qualitativa Nominal"],

            ["Idade","Qualitativa Ordinal"],

            ["Ocupacao","Qualitativa Nominal"], 

            ["Categoria_Cidade","Qualitativa Nominal"], 

            ["Estado_Civil","Qualitativa Nominal"],

            ["Categoria_Produto_1","Qualitativa Nominal"],

            ["Categoria_Produto_2","Qualitativa Nominal"],

            ["Categoria_Produto_3","Qualitativa Nominal"],

            ["Anos_Permanencia_Cidade","Quantitativa Discreta"],

            ["Valor_Compra","Quantitativa Discreta"]]



classificacao = pd.DataFrame(classificacao, columns=["Variavel", "Classificação"])

classificacao
pd.DataFrame(df["IdUsuario"].value_counts())
pd.DataFrame(df["IdProduto"].value_counts())
pd.DataFrame(df["Ocupacao"].value_counts())
pd.DataFrame(df["Categoria_Cidade"].value_counts())
pd.DataFrame(df["Estado_Civil"].value_counts())      
pd.DataFrame(df["Categoria_Produto_1"].value_counts())
pd.DataFrame(df["Categoria_Produto_2"].value_counts())
pd.DataFrame(df["Categoria_Produto_3"].value_counts())
genero = df["Genero"].value_counts().reset_index()



plt.figure(figsize=(10, 5))

plt.bar(genero['index'], genero["Genero"])

plt.title('Quantidade X Gênero')

plt.xlabel('Gênero')

plt.ylabel('Quantidade')

plt.show()

fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(aspect="equal"))



# Some data

labels = df["Genero"].unique()

qtde = df["Genero"].value_counts()



def func(pct, allvals):

    absolute = int(pct/100.*np.sum(allvals))

    return "{:.1f}%\n({:d})".format(pct, absolute)





wedges, texts, autotexts = ax.pie(qtde, autopct=lambda pct: func(pct, qtde),

                                  textprops=dict(color="w"))



ax.legend(wedges, labels,

          title="Gênero",

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1)

         )



plt.setp(autotexts, size=12, weight="bold")



ax.set_title("Distribuição por Gênero",fontsize=18)

plt.show()
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(aspect="equal"))



# Some data

labels = df["Estado_Civil"].unique()

qtde = df["Estado_Civil"].value_counts()



def func(pct, allvals):

    absolute = int(pct/100.*np.sum(allvals))

    return "{:.1f}%\n({:d})".format(pct, absolute)





wedges, texts, autotexts = ax.pie(qtde, autopct=lambda pct: func(pct, qtde),

                                  textprops=dict(color="w"))



ax.legend(wedges, labels,

          title="Estado Civil",

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1)

         )



plt.setp(autotexts, size=12, weight="bold")



ax.set_title("Distribuição por Estado Civil",fontsize=18)

plt.show
Idade_order = list(df['Idade'].unique())

Idade_order.sort()

sns.set()



fig = plt.figure(figsize=(14,16))

ax1 = fig.add_subplot(211)





ax1.set_title('Valor gasto por produto x Idade', fontsize=20)

sns.violinplot(x='Idade', y='Valor_Compra', data=df, order=Idade_order, ax=ax1)

ax1.set_xlabel('Idade', fontsize=12)

ax1.set_ylabel('Valor do produto', fontsize=12)

plt.show()


soma = df["Categoria_Cidade"].value_counts().sum()

ax = df["Categoria_Cidade"].value_counts().plot(kind='barh', figsize=(18,7),

                                        color="steelblue", fontsize=12);

ax.set_alpha(0.5)

ax.set_title("Distribuição das categorias de cidades.", fontsize=18)

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

            str("{}%".format(round((i.get_width()/soma *100), 2))), fontsize=12,

color='dimgrey')



# invert for largest on top 

ax.invert_yaxis()