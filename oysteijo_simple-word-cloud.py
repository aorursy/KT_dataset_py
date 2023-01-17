import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline



from wordcloud import WordCloud, STOPWORDS
df = pd.read_csv("../input/comeytestimony/qa.csv") # updated to match uploaded set

df.head()
questions = df["Full Question"]

answers = df["Comey Response"]
wordcloud_q = WordCloud(

                          background_color='white',

                          stopwords=set(STOPWORDS),

                          max_words=250,

                          max_font_size=40, 

                          random_state=1705

                         ).generate(str(questions))

wordcloud_a = WordCloud(

                          background_color='white',

                          stopwords=set(STOPWORDS),

                          max_words=250,

                          max_font_size=40, 

                          random_state=1705

                         ).generate(str(answers))
def cloud_plot(wordcloud):

    fig = plt.figure(1, figsize=(20,15))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
cloud_plot(wordcloud_q)
cloud_plot(wordcloud_a)