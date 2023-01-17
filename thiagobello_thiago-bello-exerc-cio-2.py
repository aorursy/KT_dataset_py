import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



bf = pd.read_csv('../input/dataviz-facens-20182-ex3/BlackFriday.csv', delimiter=',')

bf = pd.DataFrame(bf)

bf
idade = list(bf['Age'].unique())

idade.sort()



group_by_user = bf.groupby(['User_ID', 'Age']).sum().reset_index()
sns.set()



fig = plt.figure(figsize=(20,16))

ax1 = fig.add_subplot(211)



ax1.set_title('Valor gasto Produto x Idade', fontsize=20)

sns.violinplot(x='Age', y='Purchase', data=bf, order=idade, ax=ax1)

ax1.set_xlabel('Idade', fontsize=12)

ax1.set_ylabel('Valor do Produto', fontsize=12)

pd.DataFrame(bf["Product_ID"].value_counts())

plt.figure(figsize=(20, 6))

bf["Product_ID"].value_counts().head(5).plot(kind='bar', title='Produtos mais comprados')
valor = bf[bf['Purchase'] > 9000]

plt.figure(figsize=(16, 6))

sns.catplot(x='Marital_Status', y='Purchase', hue='Marital_Status',  margin_titles=True,

            kind="strip", col='Occupation', data=valor, aspect=.4, col_wrap=7)
import pandas as pd

BlackFriday = pd.read_csv("../input/BlackFriday.csv")