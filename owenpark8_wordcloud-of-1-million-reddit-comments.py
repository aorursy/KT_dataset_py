import os

import pandas as pd 

import seaborn as sns

from os import path

from wordcloud import WordCloud



# get data directory (using getcwd() is needed to support running example in generated IPython notebook)

d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()



# Read the whole text.

text = open(path.join(d, "../input/1-million-reddit-comments-from-40-subreddits/kaggle_RC_2019-05.csv")).read()



# Generate a word cloud image

wordcloud = WordCloud().generate(text)



# Display the generated image:

# the matplotlib way:

import matplotlib.pyplot as plt

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")



# lower max_font_size

wordcloud = WordCloud(max_font_size=40).generate(text)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()


