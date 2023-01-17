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
# representando no grafico de violino

data1 = df['Idade'].sort_values(ascending=True)

data2 = df['Valor_Compra']

fig = plt.figure(figsize=(8,8))

ax = sns.violinplot(x=data1, y=data2, palette="muted", split=True).set_title('Valor gasto por Idade')

n = 10 

produtos_comprados = pd.DataFrame(df["IdProduto"].value_counts())

top_produtos = produtos_comprados.head(n)



ax = top_produtos.plot(kind='barh', figsize=(15,10), fontsize=13);

ax.set_alpha(0.8)

ax.set_title("Top 10 produtos mais comprados.", fontsize=18)

ax.set_xticks([])

ax.legend('')

ax.set_xlabel('Quantidade de itens comprados')

plt.margins(0.1)

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
# representando no grafico de violino

top_ocupacao = df[df['Ocupacao'].isin(df['Ocupacao'].value_counts().head(5).index)]

plt.figure(figsize=(20,10))

sns.catplot(x="Ocupacao",

            y="Valor_Compra", 

            kind="boxen",

            hue="Idade",

            data=top_ocupacao.sort_values("Ocupacao"),

            height=5, # make the plot 5 units high

            aspect=3);
data1 =  df[df['Valor_Compra'] > 9000]

sns.catplot(x="Ocupacao", 

            y="Valor_Compra", 

            kind="boxen",

            hue="Estado_Civil",

            data=data1,

            height=4, # make the plot 5 units high

            aspect=3

           );