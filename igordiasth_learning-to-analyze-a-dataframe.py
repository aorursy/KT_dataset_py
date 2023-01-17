# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')
# Reading the dataset file

df = pd.read_csv("../input/vgsales.csv")
# Displaying the first 5 rows of the DataFrame

df.head()
# Summary of information in all columns

df.describe()
# Data type in each column

df.dtypes
# Number of rows and columns in the DataFrame

df.shape
# Renaming the columns

df.columns = ['Ranking', 'Nome', 'Plataforma', 'Ano', 'Gênero',

                     'Editora', 'Vendas América do Norte', 'Vendas EUA',

                     'Vendas Japão', 'Outras vendas', 'Vendas global']
# Displaying the first 5 lines of the file

df.head()
# Checking lines where there is no release year set

df[df['Ano'].isnull()].head()
# How many games have been released for each platform

df['Plataforma'].value_counts()
# Generate a graph two ways

titulos_lancados = df['Plataforma'].value_counts()

titulos_lancados.plot()



df['Plataforma'].value_counts().plot()



plt.show()
# Creating a Chart Using Only One Line of Code

df['Plataforma'].value_counts().head(50).plot(kind='bar', figsize=(11,5), grid=False, rot=0, color='blue')



# Embellishing the chart. Below, we define a title

plt.title('The 50 video games with the most titles released')

plt.xlabel('Videogame') # Naming the X axis, where is the name of the video games

plt.ylabel('Number of games released') # Naming the Y-axis, where is the number of games

plt.show() # Show the chart
# 50 best selling games ever

top_10_vendidos = df[['Nome', 'Vendas global']].head(10).set_index('Nome').sort_values('Vendas global', ascending=True)

top_10_vendidos.plot(kind='barh', grid=False, color='darkred', legend=False)



plt.title('The 50 best selling games in the world')

plt.xlabel('Total sales (in millions of dollars)')

plt.show()
crosstab_vg = pd.crosstab(df['Plataforma'], df['Gênero'])

crosstab_vg.head()
crosstab_vg['Total'] = crosstab_vg.sum(axis=1)

crosstab_vg.head()
top10_platforms = crosstab_vg[crosstab_vg['Total']>1000].sort_values('Total', ascending=False)

top10_final = top10_platforms.append(pd.DataFrame(top10_platforms.sum(), columns=['Total']).T, ignore_index=False)



sns.set(font_scale=1)

plt.figure(figsize=(18,9))

sns.heatmap(top10_final, annot=True, vmax=top10_final.loc[:'PS',:'Strategy'].values.max(), vmin=top10_final.loc[:,:'Strategy'].values.min(), fmt='d')

plt.xlabel('Genre')

plt.ylabel('Console')

plt.title('Number of titles by genre and console')

plt.show()