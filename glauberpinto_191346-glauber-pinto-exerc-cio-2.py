import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

from matplotlib.gridspec import GridSpec
# Gráfico

blackfriday = pd.read_csv('../input/dataviz-facens-20182-ex3/BlackFriday.csv')



ax = sns.violinplot(data=blackfriday, x='Age', y='Purchase', order=sorted(list(blackfriday['Age'].value_counts().index)))

plt.title('Consumo por faixa', fontsize=20)

plt.ylabel('Valor Gasto')

plt.xlabel('Faixa')







# Dados dos produtos

produtos = blackfriday['Product_ID'].value_counts().head(8).index



# Gráfico



sns.set(style="whitegrid", font_scale=1.5)

ax = sns.countplot(x = 'Product_ID', data = blackfriday, order = produtos)

plt.title('Produtos com mais compra',fontsize= 20)

plt.xlabel('Produto')

plt.ylabel('Qtd. Compra')

ocupacao = blackfriday['Occupation'].value_counts().head(5)



df = blackfriday[(blackfriday['Occupation'] == ocupacoes.index[0]) | 

   (blackfriday['Occupation'] == ocupacao.index[1]) |

   (blackfriday['Occupation'] == ocupacao.index[2]) |

   (blackfriday['Occupation'] == ocupacao.index[3]) |

   (blackfriday['Occupation'] == ocupacao.index[4]) ].groupby(['Occupation', 'Age'])['Purchase'].sum()/100000000



plt = df.plot.bar(color = 'green');

plt.plot()
sns.catplot(x = 'Marital_Status',

            y = 'Purchase',

            hue = 'Marital_Status',

            margin_titles = True,

            kind = 'point',

            col = 'Occupation',

            data = blackfriday[blackfriday['Purchase'] > 9000],

            aspect = .9,

            col_wrap = 7)