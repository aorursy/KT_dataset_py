import numpy as np 

import pandas as pd

from os import path 

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
df = pd.read_csv(r"/kaggle/input/york-shop-covid-closed-signs/York_covid_closed_signs.csv")
df.head(20)
word_string = ''

for row in df['text']:

    row = row+" "

    word_string += str(row)



stopwords = ["will","be","please"] + list(STOPWORDS)





wordcloud = WordCloud(width = 800, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(word_string) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 