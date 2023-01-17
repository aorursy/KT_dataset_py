import re

import jovian

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

import emoji

from collections import Counter
me = "Prajwal Prashanth"
comment_words = ' '

stopwords = STOPWORDS.update(['lo', 'ge', 'Lo', 'illa', 'yea', 'ella', 'en', 'na', 'En', 'yeah', 'alli', 'ide', 'okay', 'ok', 'will'])

  

for val in df.msg.values: 

    val = str(val) 

    tokens = val.split() 

        

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

          

    for words in tokens: 

        comment_words = comment_words + words + ' '

  

  

wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='black', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words) 

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
wordcloud.to_image()