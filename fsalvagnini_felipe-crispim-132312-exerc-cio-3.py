# Importa as libs necessárias

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns
# Carrega o data frame

# Recebe o range de idades 

df_bf = pd.read_csv('../input/BlackFriday.csv')

age_ranges = sorted(list(df_bf['Age'].value_counts().index))
sns.set(rc = {'figure.figsize' : (10, 5)})

sns.violinplot(x = df_bf['Age'], y = df_bf['Purchase'], order = age_ranges)

# Coloca informações no gŕafico para facilitar a compreenssão

plt.title('Consumo por faixa etária')

plt.xlabel('Faixa etária')

plt.ylabel('Valor gasto')
# Agrupa por usuários

users_grouped = df_bf.groupby(['User_ID', 'Age']).sum().reset_index()
sns.set(rc = {'figure.figsize' : (10, 5)})

sns.violinplot(x = users_grouped['Age'], y = users_grouped['Purchase'], order = age_ranges)

# Coloca informações no gŕafico para facilitar a compreenssão

plt.title('Consumo de clientes por faixa etária ')

plt.xlabel('Faixa etária')

plt.ylabel('Valor gasto')
# Verifica quais são os produtos que geram mais valor durante as black frydays

high_value_itens = list(df_bf['Product_ID'].value_counts().head(8).index)

high_value_itens = [i for i in reversed(high_value_itens)]
sns.set(rc = {'figure.figsize' : (10, 5)})

sns.countplot(x = 'Product_ID', data = df_bf, order = high_value_itens)

plt.title('Produtos que geram mais retorno')

plt.xlabel('ID do produto')

plt.ylabel('Quantidade de itens comprados')
# Verifica quais são os IDs das ocupações que ocorrem com mais frequência

most_freq_occupations = list(df_bf['Occupation'].value_counts().index)[0:5]
sns.set(rc = {'figure.figsize' : (10, 5)})

df_17_occupations = df_bf.query('Age == "0-17"')

df_17_occupations = df_17_occupations[df_17_occupations['Occupation'].isin(most_freq_occupations)]

sns.violinplot(x = df_17_occupations['Occupation'], y = df_17_occupations['Purchase'])

plt.title('Consumo das 5 ocupações mais frequentes (Até 17 Anos)')

plt.xlabel('ID da Ocupação')

plt.ylabel('Valor gasto')
sns.set(rc = {'figure.figsize' : (10, 5)})

df_25_occupations = df_bf.query('Age == "18-25"')

df_25_occupations = df_25_occupations[df_25_occupations['Occupation'].isin(most_freq_occupations)]

sns.violinplot(x = df_25_occupations['Occupation'], y = df_25_occupations['Purchase'])

plt.title('Consumo das 5 ocupações mais frequentes (18 à 25 Anos)')

plt.xlabel('ID da Ocupação')

plt.ylabel('Valor gasto')
sns.set(rc = {'figure.figsize' : (10, 5)})

df_35_occupations = df_bf.query('Age == "26-35"')

df_35_occupations = df_35_occupations[df_35_occupations['Occupation'].isin(most_freq_occupations)]

sns.violinplot(x = df_35_occupations['Occupation'], y = df_35_occupations['Purchase'])

plt.title('Consumo das 5 ocupações mais frequentes (26 à 35 Anos)')

plt.xlabel('ID da Ocupação')

plt.ylabel('Valor gasto')
sns.set(rc = {'figure.figsize' : (10, 5)})

df_45_occupations = df_bf.query('Age == "36-45"')

df_45_occupations = df_45_occupations[df_45_occupations['Occupation'].isin(most_freq_occupations)]

sns.violinplot(x = df_45_occupations['Occupation'], y = df_45_occupations['Purchase'])

plt.title('Consumo das 5 ocupações mais frequentes (36 à 45 Anos)')

plt.xlabel('ID da Ocupação')

plt.ylabel('Valor gasto')
sns.set(rc = {'figure.figsize' : (10, 5)})

df_50_occupations = df_bf.query('Age == "46-50"')

df_50_occupations = df_50_occupations[df_50_occupations['Occupation'].isin(most_freq_occupations)]

sns.violinplot(x = df_50_occupations['Occupation'], y = df_50_occupations['Purchase'])

plt.title('Consumo das 5 ocupações mais frequentes (46 à 50 Anos)')

plt.xlabel('ID da Ocupação')

plt.ylabel('Valor gasto')
sns.set(rc = {'figure.figsize' : (10, 5)})

df_55_occupations = df_bf.query('Age == "51-55"')

df_55_occupations = df_55_occupations[df_55_occupations['Occupation'].isin(most_freq_occupations)]

sns.violinplot(x = df_55_occupations['Occupation'], y = df_55_occupations['Purchase'])

plt.title('Consumo das 5 ocupações mais frequentes (51 à 55 Anos)')

plt.xlabel('ID da Ocupação')

plt.ylabel('Valor gasto')
sns.set(rc = {'figure.figsize' : (10, 5)})

df_56_occupations = df_bf.query('Age == "55+"')

df_56_occupations = df_56_occupations[df_56_occupations['Occupation'].isin(most_freq_occupations)]

sns.violinplot(x = df_56_occupations['Occupation'], y = df_56_occupations['Purchase'])

plt.title('Consumo das 5 ocupações mais frequentes (Mais que 55 Anos)')

plt.xlabel('ID da Ocupação')

plt.ylabel('Valor gasto')
# Seleciona todas as compras com mais de 9k gastos

df_purchases_9k = df_bf.query('Purchase > 9000')

all_occupations = list(df_purchases_9k['Occupation'].value_counts().index)
plt.figure(figsize=(10,5))

plt.scatter(df_purchases_9k['Occupation'], df_purchases_9k['Marital_Status'], s = df_purchases_9k['Occupation'] * 33, color = 'green')

plt.yticks([0, 1])

plt.xticks(all_occupations)

plt.title("Consumo por ocupação e estado civil (Maior que 9000)")

plt.ylabel("Estado Civil")

plt.xlabel("Ocupação")

plt.show()