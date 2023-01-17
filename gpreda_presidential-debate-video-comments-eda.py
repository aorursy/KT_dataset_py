import numpy as np 

import pandas as pd

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

from wordcloud import WordCloud, STOPWORDS

import os
data_df = pd.read_csv("/kaggle/input/presidential-debate-video-comments/presidential_debate_video_comments.tab", sep="\t")
data_df.head()
data_df.info()
def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 0.2,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()    
plot_count("authorName", "Most frequent commentors (first 20)", data_df, size=4)
stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=50,

        max_font_size=40, 

        scale=5,

        random_state=1

    ).generate(str(data))



    fig = plt.figure(1, figsize=(10,10))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
show_wordcloud(data_df['text'], title = 'Prevalent words in comments')