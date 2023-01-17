# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df=pd.read_csv('../input/winemag-data_first150k.csv')
df.head(5)

# Any results you write to the current directory are saved as output.
import seaborn as sns
sns.heatmap(df.isnull())
import seaborn as sns
sns.scatterplot(df['points'],df['price'])
from nltk import word_tokenize
desc=word_tokenize(str(df['description']))
print(desc)
from wordcloud import STOPWORDS,WordCloud
filtered_sentence=[i for i in desc if not i in STOPWORDS]
print(filtered_sentence)
import string
filtered_nopunc=[i for i in filtered_sentence if not i in string.punctuation]
print(filtered_nopunc)
filtered_sentence=" ".join(filtered_nopunc)
print(filtered_sentence)
import matplotlib.pyplot as plt
wordcloud=WordCloud(width=400,height=500,min_font_size=10,stopwords=STOPWORDS).generate(filtered_sentence)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
from wordcloud import STOPWORDS,WordCloud
import matplotlib.pyplot as plt
wordcloud=WordCloud(width=400,height=400,min_font_size=10,stopwords=STOPWORDS).generate(str(df['description']))
plt.figure(figsize=(10,10))
plt.imshow(wordcloud)
import seaborn as sns
sns.heatmap(df.corr(),annot=True)
import matplotlib.pyplot as plt
ax=sns.countplot(df['country'])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.figure(figsize=(300,300))
plt.show()
import matplotlib.pyplot as plt
x=df['province'].value_counts().head(10).index
y=df['province'].value_counts().head(10).values
ax=sns.barplot(x,y)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40)
plt.title('top 10')
plt.show()
sns.FacetGrid(df,'points','price')