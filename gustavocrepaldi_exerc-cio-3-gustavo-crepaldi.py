import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
dfBlackFriday = pd.read_csv('../input/BlackFriday.csv', delimiter=',')
dfBlackFriday.head()
# Analisando dados estatisticos do dataset

dfBlackFriday.describe()
# verificando se existe campos com nulos

dfBlackFriday.isnull().sum()
# representando no grafico de violino

sns.violinplot(dfBlackFriday["Age"].sort_values(),dfBlackFriday['Purchase'],data=dfBlackFriday)





plt.show()
# contando quantos foram comprados de cada produto

dfBlackFriday['Product_ID'].value_counts()
# Pegando os 10 maiores produtos comprados

produtosmaisComprados = dfBlackFriday["Product_ID"].value_counts().head(10)



produtosmaisComprados.plot(kind='bar', title='10 Produtos mais comprados')

plt.xlabel('Produtos')

plt.ylabel('Quantidade')

# Usando o grafico de violino para demonstrar a relação entre Ocupação e estado civil

dfBlackFridayCons = dfBlackFriday.query('Purchase > 9000')

sns.violinplot(dfBlackFridayCons['Marital_Status'],dfBlackFridayCons['Occupation'],data=dfBlackFridayCons)