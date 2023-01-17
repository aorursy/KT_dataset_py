import numpy as np

import pandas as pd 

import re

import nltk

from nltk.corpus import stopwords, comparative_sentences

from collections import Counter

import os
import json

texts = []

for dirname, _ ,filenames in os.walk("/kaggle/input/CORD-19-research-challenge"):

    for file in filenames:

        if file.endswith(".json"):

            with open(os.path.join(dirname,file)) as f:

                texts.append(json.loads(f.read()))
stopwords_eng = stopwords.words("english")



text = "".join([x.get('body_text')[0].get('text') for x in texts])

relev_words = [x for x in text.split(" ") if x.lower() not in stopwords_eng and len(x) >= 3]

relev_text = " ".join([x.lower() for x in relev_words])
from wordcloud import WordCloud

import matplotlib.pyplot as plt

%matplotlib inline







wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords_eng, 

                min_font_size = 10).generate(relev_text) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 