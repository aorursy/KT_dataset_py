import numpy as np # linear algebra

import pandas as pd 

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

from wordcloud import WordCloud, STOPWORDS



#mpl.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)

mpl.rcParams['font.size']=12                #10 

mpl.rcParams['savefig.dpi']=150             #72 

mpl.rcParams['figure.subplot.bottom']=.1 





stopwords = set(STOPWORDS)

data = pd.read_csv("../input/live.csv")



wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stopwords,

                          max_words=200,

                          max_font_size=40, 

                          random_state=42

                         ).generate(str(data['title']))





plt.imshow(wordcloud)

plt.axis('off')

plt.show()