import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
d_2019 = pd.read_csv('../input/world-happiness/2019.csv')

df_2019 = pd.DataFrame(d_2019)
d_2019.loc[d_2019['Score'].idxmax()]
d_2019.loc[d_2019['Score'].idxmin()]
ss = df_2019['Social support']

rank = df_2019['Score']
denominator = rank.dot(rank) - rank.mean() * rank.sum()
m = (rank.dot(ss) - ss.mean() * rank.sum()) / denominator

b = (ss.mean() * rank.dot(rank) - rank.mean() * rank.dot(ss)) / denominator 
y_pred = m*rank + b
# scatter plot

plt.scatter(rank,ss)

plt.plot(rank,y_pred,'r')

np.corrcoef(rank,ss)
# correlation matrix

df_2019[['Score','GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices',

                   'Generosity','Perceptions of corruption']].corr()
# correlation matrix

f = plt.figure(figsize=(15, 12))

plt.matshow(df_2019.corr(), fignum=f.number)

plt.xticks(range(df_2019.shape[1]), df_2019.columns, fontsize=14, rotation=45)

plt.yticks(range(df_2019.shape[1]), df_2019.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
# filter by US

df_2019.set_index('Country or region', inplace=True)

df_2019.loc['United States']
top_inf = df_2019[['GDP per capita','Social support','Healthy life expectancy','Freedom to make life choices',

                   'Generosity','Perceptions of corruption']].idxmax(axis=1)
# pie chart

labels = 'Social Support', 'GDP per Capita', 'Generosity'

my_colors = ['lightblue','lightsteelblue','silver']

my_explode = (0.1, 0, 0)

plt.pie(top_inf.value_counts(), labels=labels,autopct='%1.1f%%', startangle=15, shadow = True, colors=my_colors, explode=my_explode)

plt.title('Contributor to Happiness')

plt.axis('equal')

plt.show()