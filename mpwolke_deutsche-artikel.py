#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTCrqa0R8bMPOUhQWeNXNkuHpYVovmW_Bka0Pbpdmnxd-ov-FHg',width=400,height=400)
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
df = pd.read_csv('../input/10k-german-news-articles/Articles.csv', encoding='ISO-8859-2')

df.head()
df.dtypes
sns.distplot(df["ID_Article"].apply(lambda x: x**4))

plt.show()
df["ID_Article"].plot.box()

plt.show()
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df.Title)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
df1 = pd.read_csv('../input/10k-german-news-articles/Posts.csv', encoding='ISO-8859-2')

df1.head()
df1.dtypes
sns.pairplot(df1, x_vars=['ID_Article'], y_vars='PositiveVotes', markers="+", size=4)

plt.show()
sns.pairplot(df1, x_vars=['ID_Article'], y_vars='NegativeVotes', markers="+", size=4)

plt.show()
df1corr=df1.corr()

df1corr
sns.heatmap(df1corr,annot=True,cmap='pink')

plt.show()
fig, axes = plt.subplots(1, 1, figsize=(12, 4))

sns.boxplot(x='ID_Article', y='PositiveVotes', data=df1, showfliers=False);
sns.countplot(df1["NegativeVotes"])

plt.xticks(rotation=90)

plt.show()
fig=sns.lmplot(x="ID_Article", y="PositiveVotes",data=df1)
#word cloud

from wordcloud import WordCloud, ImageColorGenerator

text = " ".join(str(each) for each in df1.Body)

# Create and generate a word cloud image:

wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)

plt.figure(figsize=(10,6))

plt.figure(figsize=(15,10))

# Display the generated image:

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.show()
#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSGpe_-yZDnNbNSBNqCzv4fI9T1b7uhLX1zXejCdzWW-QZn9tak',width=400,height=400)