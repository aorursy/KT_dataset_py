import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# opening and reading a file
# make sure the path for the file is correct
with open('../input/modi-2014-speech/2014.txt', 'r') as myfile:
    data=myfile.read().replace('\n', '')


def plot_wordcloud(text, max_words=400, max_font_size=120, figure_size=(12.0,8.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'Brothers', 'sisters', 'country','Independence','Rs','government', 'will','crore','one','now','years','live','used','nation','take','want','many','come','way', 'India','people','time','year','Indian','new','Today', 'mother', 'world','whether','every','therefore','dear','countrymen'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42)
    wordcloud.generate(text)
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'green', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
comments_text = data
plot_wordcloud(comments_text, max_words=200, max_font_size=70, 
               title = 'Most common words in the 2014 speech', title_size=50)
with open('../input/2018-speech-data/2018.txt', 'r') as myfile:
    data=myfile.read().replace('\n', '')
comments_text = data
plot_wordcloud(comments_text, max_words=200, max_font_size=70, 
               title = 'Most common words in the 2018 speech', title_size=50)