import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import rc

import numpy as np

import pandas as pd



df = pd.read_csv('../input/BlackFriday.csv', delimiter=',')

df.head()
sns.set_style('whitegrid')

 

sns.violinplot(x='Age', y='Purchase', cut=0, scale="count", data=df.sort_values(by=['Age']))
g = sns.catplot(x="Purchase", y="Gender", col="Age",

                data=df.sort_values(by=['Age']), col_wrap=3,

                orient="h", height=2, aspect=3, palette="Set3",

                kind="violin", dodge=True, bw=.2)
df_target = df.groupby('Product_ID')['Product_ID'].count().reset_index(name='count').sort_values(['count'], ascending=False).head(10).reset_index(drop=True)



sns.set(style="whitegrid")

ax = sns.barplot(x="Product_ID", y="count", data=df_target)



ax.set_xlabel('Produtos')

ax.set_ylabel('Total vendido')



for item in ax.get_xticklabels():

    item.set_rotation(90)

    

for i in range(len(df_target['Product_ID'])):

    plt.text(x = i - 0.3 , y = df_target.loc[i,'count'] + 20 , s = df_target.loc[i,'count'], size = 8, color='Blue')



plt.show()
occupation_order = list(df['Occupation'].value_counts().head(5).index)



df_target = df[df['Occupation'].isin(occupation_order)].sort_values(by='Age')



plt.figure(figsize=(20,10))

g = sns.boxplot(x="Occupation", y="Purchase", hue="Age", data=df_target)



plt.title('Valores gastos por faixa etária associados às 5 ocupações mais frequentes\n', fontsize=16)

plt.xlabel('Ocupação')

plt.ylabel('Valor gasto')

plt.legend(loc=1, title='Idade')

plt.ylim(0, 35000)



plt.show() 
df_target = df[df['Purchase'] > 9000].groupby(['Marital_Status', 'Occupation'])['Purchase'].count().reset_index(name='count').reset_index(drop=True)



g = sns.catplot(x="Marital_Status", y="count", col="Occupation", col_wrap=9, data=df_target, kind="bar", height=3, aspect=.6)

(g.set_axis_labels("", "Estado Civil")

    .despine(left=True))  