import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Regex

import re

# Visualização de dados

import matplotlib.pyplot as plt

# Manipulação de dados

import numpy as np

import pandas as pd

pd.set_option("max_columns", None)

pd.set_option("max_rows", 200)
df = pd.read_csv('../input/pesquisa-data-hackers-2019/datahackers-survey-2019-anonymous-responses.csv')



# Os nomes da colunas possuem padrão ('PX', 'nome'). Para melhorar a usabilidade no código, vou deixar os nomes das colunas em PX_nome.

pattern = "\'(.*?)\'"

df.columns = ['_'.join(re.findall(pattern,x)) for x in df.columns]



df.head()
df.info(verbose=True)
print(f'Dimensão do dataset: {df.shape}')
colunas_com_dados_nan = df.columns[ df.isna().sum()>0 ]



plt.figure(figsize=(10,8))

plt.style.use('ggplot')

plt.title('% de dados nulos', fontsize=16)

(df[colunas_com_dados_nan].isna().sum() / len(df)).sort_values(ascending=True).plot(kind='barh')