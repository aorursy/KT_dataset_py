import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

df = pd.read_csv('../input/dataviz-facens-20182-ex3/BlackFriday.csv', delimiter=',')
ages = df.sort_values(by="Age")
ages = ages.dropna(subset=['Age', 'Purchase'])
dims = (15, 10)
fig, ax = plt.subplots(figsize=dims)
sns.set_style("whitegrid")
v = sns.violinplot(x='Age', y='Purchase', data= ages, palette="RdBu_r")
v.set_title('Valor gasto por faixa de idade')

pd = df[df['Product_Category_1'] > 8]
pd = pd['Product_Category_1'].astype(str)
pd.value_counts()

pd = pd.value_counts()

figure(num=None, figsize=(15, 10), dpi=80, edgecolor='green')
plt.bar(pd.keys(), pd, color='lightgreen',  edgecolor='b')
plt.title('Produtos Mais Comprados Superiores a Categoria 8')
plt.show()

occ = df['Occupation'].astype(str)
occ = occ.dropna()
filt_occ = occ.value_counts().head(5).keys()
filt_occ = filt_occ.astype('int64')
new_df = df[df['Occupation'].isin(filt_occ)]

new_df = new_df.sort_values(by=["Age","Occupation"])

dims = (20, 10)
fig, ax = plt.subplots(figsize=dims)
sns.set_style("whitegrid")
v = sns.boxplot(x='Age', y='Purchase', data= new_df, hue='Occupation', palette="RdBu_r")
v.set_title('Valor Gasto por Faixa de Idade Pelas 5 Ocupações Mais Comuns')

over9k = df[df['Purchase'] > 9000]
over9k = over9k.dropna(subset=['Occupation', 'Marital_Status','Purchase'])
over9k['Marital'] =  np.where(over9k['Marital_Status']== 1,'Not Single','Single')



s = sns.catplot(x='Marital', y='Purchase', col='Occupation', data= over9k, kind='violin', palette="RdBu_r", col_wrap=7, aspect=.6)
plt.subplots_adjust(top=0.9)
s.fig.suptitle('Relação de Gasto entre Ocupação e Estado Civil para Valores acima de 9000')

