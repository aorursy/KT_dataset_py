# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Lendo a base de dados

suicide = pd.read_csv('/kaggle/input/who-suicide-statistics/who_suicide_statistics.csv')
suicide.shape
suicide.head()
suicide.info()
# Tratando "missing values"

suicide[suicide['population'].isnull()]
# Excluindo as linhas sem valores para população

suicide.dropna(subset=['population'],inplace=True)
suicide.info()
# Excluindo as linhas sem valores para número de suicídios

suicide.dropna(subset=['suicides_no'],inplace=True)
suicide.info()
suicide.head()
# Verificando estatísticas descritivas

suicide.describe()
# Criando um dataframe com os dados de 2016

year_2016 = suicide[suicide['year'] == 2016]
year_2016.head(20)
# Agrupando os dados por país

gb_country_2016 = year_2016.groupby('country').sum()
# Criando uma coluna calculada para taxa de suícidio por 100 mil habitantes

gb_country_2016['suicides_per_100k'] = gb_country_2016['suicides_no']/(gb_country_2016['population']/100000)
# Ordenando os dados pela taxa de suicídio em ordem decrescente

gb_country_2016.sort_values(by='suicides_per_100k', ascending=False)
# Ordenando os dados pela taxa de suicídio em ordem crescente

gb_country_2016.sort_values(by='suicides_per_100k', ascending=True)
# Criando um dataframe com os dados de 2015

year_2015 = suicide[suicide['year'] == 2015]
year_2015.head(20)
# Agrupando os dados por país

gb_country_2015 = year_2015.groupby('country').sum()
# Criando uma coluna calculada para taxa de suícidio por 100 mil habitantes

gb_country_2015['suicides_per_100k'] = gb_country_2015['suicides_no']/(gb_country_2015['population']/100000)
# Ordenando os dados pela taxa de suicídio em ordem decrescente

gb_country_2015.head(60).sort_values(by='suicides_per_100k', ascending=False)
# Ordenando os dados pela taxa de suicídio em ordem crescente

gb_country_2015.head(60).sort_values(by='suicides_per_100k', ascending=True)
import seaborn as sns



# 'years' na faixa etária parece desnecessário e '05-14' será melhor que '5-14 '

suicide.loc[:, 'age'] = suicide['age'].str.replace(' years','')

suicide.loc[suicide['age'] == '5-14', 'age'] = '05-14'



# cálculo da taxa de suicídio por 100 mil habitantes

suicide['suicides_per_100k'] = suicide['suicides_no'] / (suicide['population'] / 100000)



# visualização

pd.concat([suicide[:2], suicide[10000:10002], suicide[-2:]])
sns.set()



cd = (suicide.loc[(suicide['country'].isin(['Denmark','Brazil','United States of America'])) 

             & (suicide['year'] == 2015), ['country','sex','age','suicides_per_100k']]

      .sort_values(['sex','age']))



sns.catplot(x='age', hue='sex', col='country', y='suicides_per_100k'

            , data=cd, kind='bar', col_wrap=3)