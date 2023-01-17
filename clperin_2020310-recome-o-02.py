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
# Carregando o arquivo

df = pd.read_csv('/kaggle/input/insurance/insurance.csv')
df.shape
df.info()
df.head()
df['charges'].plot.hist(bins=50)
df['bmi'].plot.hist(bins=50)
import seaborn as sns

# Relacionando fumantes e não fumantes com gastos

sns.stripplot(x='smoker', y='charges', data=df, linewidth=1)


# Relacionando idade mínima e máxima com gastos

df['age'].min(), df['age'].max()
# Criando categorias por idade

def age_to_cat(age):

    if (age >=18) & (age <=35):

        return 'Adulto'

    elif (age >35) & (age <=55):

        return 'Senior'

    else:

        return 'Idoso'



df['age_cat'] = df['age'].apply(age_to_cat)



df.head()
df.sample()
# relação idade gasto

sns.stripplot(x='age_cat', y='charges', data=df, linewidth=1)
sns.stripplot(x='smoker', y='charges', data=df, linewidth=1, hue='age_cat')
sns.stripplot(x='age_cat', y='charges', data=df, linewidth=1, hue='smoker')
# Para casa

#

# = Determinar os clusters

#  ----bmi

# ----charges

# Usar método do cumprimento de cotovelo do coronavírus
