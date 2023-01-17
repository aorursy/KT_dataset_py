# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importando os dados

df = pd.read_csv('/kaggle/input/wine-reviews/winemag-data-130k-v2.csv')



# Verificando tamanho (linhas, colunas)

df.shape
# Verificando tipos e quantidades

df.info()
# dando uma olhada nos dados

df.head()
# Slice - corte no dataframe



# Pegar todos os dados de uma coluna

df['variety']
# Quais as variedades mais comuns?

df['variety'].value_counts()
# Quais as variedades mais comuns?

df['variety'].value_counts().head()
# Quais as variedades mais comuns? Em gráfico!

df['variety'].value_counts().head().sort_values(ascending=True).plot.barh()
# Slice de duas (ou mais) colunas, precisamos passar uma lista

df[['country', 'province']]
# Slice, seecionando linhas específicas

# Vinhos dos US

df[df['country'] == 'US']
# Slice, seecionando linhas específicas

# Vinhos dos US, do tipo Zinfandel

df[(df['country'] == 'US') & (df['variety'] == 'Zinfandel')]
# Slice, seecionando linhas específicas

# Qual a maior pontuação de vinhos dos US, do tipo Zinfandel?

df[(df['country'] == 'US') & (df['variety'] == 'Zinfandel')]['points'].max()
# Slice, selecionando linhas específicas

# Qual o vinho dos US, do tipo Zinfandel, com maior pontuação?

df_us_zin = df[(df['country'] == 'US') & (df['variety'] == 'Zinfandel')]



df_us_zin[df_us_zin['points'] == df_us_zin['points'].max()]
# Qual o vinho dos US, do tipo Zinfandel, com maior pontuação?

df_us_zin.nlargest(1, 'points')
# Os 5 vinhos dos US, do tipo Zinfandel, com maior pontuação?

df_us_zin.nlargest(5, 'points')
# Os 5 vinhos dos US, do tipo Zinfandel, com menor pontuação?

df_us_zin.nsmallest(5, 'points')