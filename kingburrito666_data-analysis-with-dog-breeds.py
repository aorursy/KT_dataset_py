import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import networkx as nx

import glob

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Thanks to the commenter for this; still a student

data = pd.DataFrame()

for f in glob.glob('../input/*.csv'): # all files in the directory that matchs the criteria.

    data = pd.concat([data,pd.read_csv(f)])
# Lets make sure we have all the correct columns

data.columns
dog_breeds = pd.DataFrame(data.groupby('Breed').size().sort_values(ascending=False).rename('Amount').reset_index())

print(len(dog_breeds), ' dog breeds')

dog_breeds.head()
# What are the most popular dog names?

from wordcloud import WordCloud 

# Making the data frame 'DogName' into a pandas series for wordcloud

dog_names = pd.Series(data['DogName'].tolist()).astype(str)

print(len(dog_names), ' different dog names')

# Building a wordcloud!

cloud = WordCloud(

                 width=900,

                 height=800,

                 min_font_size=2,

                 background_color='white',

                 colormap='plasma'

                 ).generate(' '.join(dog_names))



plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
# Comparing Breed to Color

# First making a dataFrame with just breed and color

dog_colors = pd.DataFrame(data.groupby('Color').size().sort_values(ascending=False).rename('Amount').reset_index())

f, ax = plt.subplots(figsize=(6, 15))



sns.barplot(x='Amount', y='Color', data=dog_colors.head(20))

plt.show();
# Breed per zipcode area

breed_zip = pd.concat([data['Breed'], data['OwnerZip']], axis=1)

f, ax = plt.subplots(figsize=(5, 15))

# Boxplot

sns.boxplot(x='OwnerZip', y='Breed', data=breed_zip.head(200))

plt.xticks(rotation=10)

plt.show();
g= nx.Graph()

g = nx.from_pandas_dataframe(data.head(1000),source='Breed',target='DogName')

print (nx.info(g))
plt.figure(figsize=(80,80))

pos=nx.spring_layout(g, scale=1.5, k=0.6)

nx.draw_networkx(g,pos,node_size=100, node_color='black', font_color='white')

plt.show()