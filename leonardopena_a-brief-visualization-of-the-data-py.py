

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud



fpath = '../input/top50spotify2019/top50.csv'

data = pd.read_csv(fpath, encoding='ISO-8859-1').drop(['Unnamed: 0'], axis =1)

data.head()
print(data.shape)
print(data.dtypes)

data['cut_pop'] = 0

for i in range(len(data)):

    data['cut_pop'].iloc[i] = pd.cut(data.Popularity, bins=5).iloc[i]



plopop = sns.countplot(y= data.cut_pop, data=data)

plopop.set(xlabel = "count", ylabel = "Popularity" )

plt.show()
# Create horizontal bars : Energy

data['cut_energy'] = 0

for i in range(len(data)):

    data['cut_energy'].iloc[i] = pd.cut(data.Energy, bins=10).iloc[i]



plopop = sns.countplot(y= data.cut_energy, data=data)

plopop.set(xlabel = "count", ylabel = "Energy" )

plt.show()
# Create horizontal bars : Length

data['cut_Length.'] = 0

for i in range(len(data)):

    data['cut_Length.'].iloc[i] = pd.cut(data['Length.'], bins=10).iloc[i]



plopop = sns.countplot(y= data['cut_Length.'], data=data)

plopop.set(xlabel = "count", ylabel = "Length." )

plt.show()
# Create the wordcloud object

wordcloud = WordCloud(width=700, height=700, margin=0).generate(str(data.Genre))



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()

data_cor = data.drop(['Track.Name','Artist.Name','Genre','cut_pop', 'cut_energy','cut_Length.'], axis=1)

sns.pairplot(data_cor, kind="reg")

plt.show()