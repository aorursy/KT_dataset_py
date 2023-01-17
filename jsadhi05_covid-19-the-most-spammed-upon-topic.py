# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy.stats as stats

from sklearn import ensemble, tree, linear_model, preprocessing

import missingno as msno

import pandas_profiling

import plotly.express as px
Original_Data= pd.read_csv('../input/dataisbeautiful/r_dataisbeautiful_posts.csv')
Original_Data.head()
removed_thread = Original_Data[Original_Data['removed_by'].notna()]

original_thread = Original_Data[(Original_Data['removed_by'].isna())]
removed_thread.head()
removers=[]



for name in removed_thread['removed_by']:

    if name not in removers:

        removers.append(name)

        

print(removers)
from wordcloud import WordCloud, STOPWORDS



# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=50, figure_size=(15.0,15.0), 

                   title = None, title_size=20, image_color=False,color = 'black'):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color=color,

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

    

plot_wordcloud(removed_thread['title'].values, title="Word Cloud of Removed Posts Titles.")
import plotly.express as px

from wordcloud import WordCloud, STOPWORDS

from collections import Counter

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

import warnings 

warnings.filterwarnings('ignore')



def count_ngrams(dataframe,column,begin_ngram,end_ngram):

    # adapted from https://stackoverflow.com/questions/36572221/how-to-find-ngram-frequency-of-a-column-in-a-pandas-dataframe

    word_vectorizer = CountVectorizer(ngram_range=(begin_ngram,end_ngram), analyzer='word')

    sparse_matrix = word_vectorizer.fit_transform(removed_thread['title'].dropna())

    frequencies = sum(sparse_matrix).toarray()[0]

    most_common = pd.DataFrame(frequencies, 

                               index=word_vectorizer.get_feature_names(), 

                               columns=['frequency']).sort_values('frequency',ascending=False)

    most_common['ngram'] = most_common.index

    most_common.reset_index()

    return most_common



def word_cloud_function(df,column,number_of_words):

    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    word_string=str(popular_words_nonstop)

    wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white',

                          max_words=number_of_words,

                          width=1000,height=1000,

                         ).generate(word_string)

    plt.clf()

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()



def word_bar_graph_function(df,column,title):

    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    plt.barh(range(50), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:50])])

    plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))

    plt.title(title)

    plt.show()

    

three_gram = count_ngrams(removed_thread,'title',3,3)

words_to_exclude = ["my","to","at","for","it","the","with","from","would","there","or","if","it","but","of","in","as","and",'NaN','dtype']
plt.figure(figsize=(10,10))

word_bar_graph_function(removed_thread,'title','Most common words in titles removed threads in Reddit')
bi_gram = count_ngrams(removed_thread,'title',2,2)

fig = px.bar(bi_gram.sort_values('frequency',ascending=False)[0:10], 

             x="frequency", 

             y="ngram",

             title='Most Common 2-Words in Titles of removed posts in Reddit',

             orientation='h')

fig.show()

fig = px.bar(three_gram.sort_values('frequency',ascending=False)[0:10], 

             x="frequency", 

             y="ngram",

             title='Most Common 3-Words in Titles of removed posts in Reddit',

             orientation='h')

fig.show()
