# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
top1000 = pd.read_csv("../input/Top_1_000_Songs_To_Hear_Before_You_Die.csv")
missing_values = top1000.isnull().sum()

print(missing_values)
top1000.head()
top1000.describe()
song_types = top1000["THEME"].value_counts(normalize = True).plot(kind='barh')
sns.set(style="white")

ax = sns.countplot(y="THEME", data=top1000, palette="Reds", order = top1000['THEME'].value_counts(ascending = True).index

                )

ax.set_xlabel('totalCount')

ax.set_ylabel('THEMES')

sns.despine(left=True, bottom=True, right=True)

ax.xaxis.label.set_color('black')

ax.yaxis.label.set_color('black')

plt.show()
sns.set(style="white")

ax = sns.countplot(y="ARTIST", data=top1000, palette="Blues_d", order = top1000['ARTIST'].value_counts(ascending = False).iloc[:15].index

                )

ax.set_xlabel('totalCount')

ax.set_ylabel('ARTIST')

sns.despine(left=True, bottom=True, right=True)

ax.xaxis.label.set_color('black')

ax.yaxis.label.set_color('black')

plt.show()
ax = sns.countplot(y="YEAR", data=top1000, palette="Greens_r", order = top1000['YEAR'].value_counts(ascending = False).iloc[:10].index

                )

ax.set_xlabel('totalCount')

ax.set_ylabel('YEAR')

sns.despine(left=True, bottom=True, right=True)

ax.xaxis.label.set_color('black')

ax.yaxis.label.set_color('black')

plt.show()
top1000[top1000["YEAR"] == 1968]
decades = []

for i in range(0,11):

    new_d = 1910 + 10*i

    decades.append(new_d)

top_decades = pd.cut(top1000["YEAR"], decades).value_counts()

top_decades
plt.hist(top1000["YEAR"], bins=decades, edgecolor="k")

plt.xticks(decades)

plt.show()
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

title_values = top1000["TITLE"].str.strip().str.replace("['´’]","").values



wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'white',

    stopwords = STOPWORDS).generate(str(title_values))

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'white',

    edgecolor = 'white')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()