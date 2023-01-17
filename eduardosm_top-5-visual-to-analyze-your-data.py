# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import Data

df = pd.read_csv("../input/mpg-ggplot2/mpg_ggplot2.csv")

df_counts = df.groupby(['hwy', 'cty']).size().reset_index(name='counts')



# Draw Stripplot

fig, ax = plt.subplots(figsize=(16,10), dpi= 80)    

sns.stripplot(df_counts.cty, df_counts.hwy, size=df_counts.counts*2, ax=ax)



# Decorations

plt.title('Counts Plot - Quanto maior a correlação maior será o circulo', fontsize=22)

plt.show()
df = pd.read_csv("../input/mpg-ggplot2/mpg_ggplot2.csv")



# Draw Plot

plt.figure(figsize=(13,10), dpi= 80)

sns.violinplot(x='class', y='hwy', data=df, scale='width', inner='quartile')



# Decoration

plt.title('Violin Plot Quilometragem pelo categoria do veículo', fontsize=22)

plt.show()
# Prepare data

x_var = 'manufacturer'

groupby_var = 'class'

df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)

vals = [df[x_var].values.tolist() for i, df in df_agg]



# Draw

plt.figure(figsize=(16,9), dpi= 80)

colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]

n, bins, patches = plt.hist(vals, df[x_var].unique().__len__(), stacked=True, density=False, color=colors[:len(vals)])



# Decoration

plt.legend({group:col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})

plt.title(f"Histograma empilhado ${x_var}$ por ${groupby_var}$", fontsize=22)

plt.xlabel(x_var)

plt.ylabel("Frequency")

plt.ylim(0, 40)

plt.xticks(ticks=bins, labels=np.unique(df[x_var]).tolist(), rotation=90, horizontalalignment='left')

plt.show()
# Draw Plot

plt.figure(figsize=(13,10), dpi= 80)

sns.boxplot(x='class', y='hwy', data=df, notch=False)



# Add N Obs inside boxplot (optional)

def add_n_obs(df,group_col,y):

    medians_dict = {grp[0]:grp[1][y].median() for grp in df.groupby(group_col)}

    xticklabels = [x.get_text() for x in plt.gca().get_xticklabels()]

    n_obs = df.groupby(group_col)[y].size().values

    for (x, xticklabel), n_ob in zip(enumerate(xticklabels), n_obs):

        plt.text(x, medians_dict[xticklabel]*1.01, "#obs : "+str(n_ob), horizontalalignment='center', fontdict={'size':14}, color='white')



add_n_obs(df,group_col='class',y='hwy')    



# Decoration

plt.title('Box Plot of Highway Mileage by Vehicle Class', fontsize=22)

plt.ylim(10, 40)

plt.show()
df = pd.read_csv("../input/mtcars/mtcars.csv")



# Plot

plt.figure(figsize=(12,10), dpi= 80)

sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap="YlGnBu", center=0, annot=True)



# Decorations

plt.title('Correlação das variáveis de mtcars', fontsize=22)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()