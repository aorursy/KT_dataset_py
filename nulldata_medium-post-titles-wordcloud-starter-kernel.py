import pandas as pd

#from sklearn.feature_extraction.text import CountVectorizer

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

medium = pd.read_csv("../input/medium_post_titles.csv")
medium.dtypes
medium.shape
medium1k = medium.sample(1000)
medium1k.head(10)
text = medium1k.title.values
wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(text))

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
