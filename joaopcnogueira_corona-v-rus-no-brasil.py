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

pd.set_option('display.max_rows', None)
df = pd.read_csv("/kaggle/input/corona-virus-brazil/brazil_covid19.csv")
df.head(200)
melt_df = (

    df

    .groupby('date')

    .agg(suspects = ('suspects', 'sum'), 

         refuses  = ('refuses', 'sum'), 

         cases    = ('cases', 'sum'), 

         deaths   = ('deaths', 'sum'))

    .reset_index()

    .query('date >= "2020-02-20"')

    .melt(id_vars='date', value_vars=['suspects', 'refuses', 'cases', 'deaths'], var_name='tipo', value_name='qtde')

    .assign(lag = lambda df: df.groupby('tipo')['qtde'].shift())

    .assign(taxa_de_crescimento = lambda df: (df['qtde'] / df['lag'] - 1))

    .drop('lag', axis=1)

)
melt_df.head(200)
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')



fig, ax = plt.subplots(figsize=(20, 6))



sns.lineplot(data=melt_df, x='date', y='qtde', hue='tipo', ax=ax);

ax.set_xlabel('');

ax.set_ylabel('');

plt.xticks(rotation=45);

ax.set_title('Quantidade de Casos por Dia', fontsize=15);
fig, ax = plt.subplots(figsize=(20, 6))



sns.lineplot(data=melt_df.dropna(), x='date', y='taxa_de_crescimento', hue='tipo', ax=ax);

ax.set_xlabel('');

ax.set_ylabel('');

plt.xticks(rotation=45);

ax.set_title('Taxa de Crescimento', fontsize=15);
907/893