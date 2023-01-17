import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import wordcloud
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
reviews = pd.read_csv("../input/GrammarandProductReviews.csv")
reviews.head(5)
text = reviews['reviews.text'].astype(str)
text.head()
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words="english", max_features=500, strip_accents='unicode',token_pattern ='[A-z]+')
wnl = WordNetLemmatizer()
cleaned = []
for i in range(len(text)):
    cleaned.append(''.join([wnl.lemmatize(text[i])]))
transformed = cv.fit_transform(cleaned)
print(cv.get_feature_names())
len(cv.get_feature_names())
from wordcloud import WordCloud
wordcloud = WordCloud( width = 800, height = 800, max_font_size=22, background_color= 'white', max_words= 400).generate(str(cv.get_feature_names()))
plt.figure(figsize = (16,16))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
title = reviews['reviews.title'].astype(str)
title.head(15)
import random

from wordcloud import STOPWORDS

stopwords = set(STOPWORDS)

def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

wordcloud = WordCloud(width = 600, height =400 ,  max_font_size=35, 
                      background_color= 'black', max_words= 500, 
                      regexp = '[A-z]+', 
                      stopwords=stopwords).generate(str(set(title))) #showing only 500 words.
plt.figure(figsize = (15,8))
plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=10), interpolation="bilinear")
plt.axis("off")
plt.show()