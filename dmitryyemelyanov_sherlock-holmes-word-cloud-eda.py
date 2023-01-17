import numpy as np

import matplotlib.pyplot as plt

import glob

from wordcloud import WordCloud,STOPWORDS

from PIL import Image
Image.open('/kaggle/input/sherlock-holmes-silhouette/sherlock-holmes-head.jpg')
Image.open('/kaggle/input/sherlock-holmes-silhouette/sherlock-holmes-and-watson.jpg')
open("/kaggle/input/sherlock-holmes-stories/sherlock/sherlock/fina.txt").readlines()[0:25]
def masked_wordcloud(text, mask):

    wordcloud = WordCloud(background_color='white',

                        stopwords = STOPWORDS,

                        max_words = 15000,

                        max_font_size = 86, 

                        random_state = 42,

                        mask = mask)

    wordcloud.generate(text)

    figure_size=(24.0,16.0)

    plt.figure(figsize=figure_size)

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
list(STOPWORDS)[:15]
text=open("/kaggle/input/sherlock-holmes-stories/sherlock/sherlock/fina.txt").read()

mask = np.array(Image.open('/kaggle/input/sherlock-holmes-silhouette/sherlock-holmes-and-watson.jpg'))

masked_wordcloud(text, mask)
text=open("/kaggle/input/sherlock-holmes-stories/sherlock/sherlock/fina.txt").read()

mask = np.array(Image.open('/kaggle/input/sherlock-holmes-silhouette/sherlock-holmes-head.jpg'))

masked_wordcloud(text, mask)