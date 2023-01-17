import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS



def generate_word_cloud(dataframe,column):

    img = WordCloud(width = 600, height = 400,

                              background_color='black', colormap = 'Oranges',

                              stopwords=words_to_exclude,

                              max_words=500,

                              max_font_size=100,

                              random_state=7

                             ).generate(str(dataframe[column]))



    plt.figure(figsize=(30,20))

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.imshow(img)

    plt.axis('off')

    plt.show()

    

fox = pd.read_csv('/kaggle/input/coronavirus-covid19-television-coverage/TelevisionNews/FOXNEWS.202003.csv')

bbc = pd.read_csv('/kaggle/input/coronavirus-covid19-television-coverage/TelevisionNews/BBCNEWS.202003.csv')

words_to_exclude = ["my","to","at","for","it","the","with","from","would","there","or","if","it","but","dtype"]
from IPython.display import Video

Video('https://archive.org/serve/FOXNEWSW_20200305_210000_Your_World_With_Neil_Cavuto/FOXNEWSW_20200305_210000_Your_World_With_Neil_Cavuto.mp4?t=1677/1712&exact=1&ignore=x.mp4', width=800, height=450)
print('Fox News:')

generate_word_cloud(fox,'Snippet')
from IPython.display import Video

Video('https://ia801504.us.archive.org/32/items/BBCNEWS_20200310_184500_BBC_News/BBCNEWS_20200310_184500_BBC_News.mp4?start=638&end=673&exact=1&ignore=x.mp4', width=800, height=450)
print('BBC News:')

generate_word_cloud(bbc,'Snippet')