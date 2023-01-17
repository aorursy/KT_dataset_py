import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv('../input/dataviz-facens-20182-ex3/BlackFriday.csv')



idades = sorted(list(df['Age'].value_counts().index))



sns.set(rc = {'figure.figsize' : (15, 10)})

sns.violinplot(x = df['Age'], y = df['Purchase'], order = idades)

plt.title('Consumo por faixa etária')

plt.xlabel('Faixa etária')

plt.ylabel('Valor gasto')

products = df.groupby(by='Product_ID')

labels_1 = products.groups.keys()

values_1 = products.sum().sort_values(by='Purchase', ascending=False).tail(9)

values_1.reset_index(inplace=True)



sns.set(style="whitegrid")



f, ax = plt.subplots(figsize=(35, 10))



sns.set_color_codes("pastel")

sns.barplot(x="Purchase", y="Product_ID", data=values_1, color="b")
lista = df['Occupation'].value_counts()[0:5].index

data = df[df['Occupation'].isin(lista)]

sns.boxenplot(x='Occupation', y='Purchase', hue='Age', data=data)
data = df[df['Purchase']>9000]

sns.boxenplot(x='Occupation', y='Purchase', hue='Marital_Status', data=data)

# sns.catplot(x='Occupation', y='Purchase', hue='Marital_Status', data=data)