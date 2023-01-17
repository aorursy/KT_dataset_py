import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# !pip install chart_studio



import re

# Tutorial about Python regular expressions: https://pymotw.com/2/re/ import string

from pandas_profiling import ProfileReport

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec 

from gensim.models import KeyedVectors 

import matplotlib.pyplot as plt

import pickle

from tqdm import tqdm

import os

import nltk

import seaborn as sns

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from plotly import tools

# import chart_studio.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff



from collections import Counter # suppress warnings

import warnings

warnings.filterwarnings("ignore")

sns.set(style="ticks", color_codes=True)

BASE_PATH = '../input/tweet-sentiment-extraction/'



train_df = pd.read_csv(BASE_PATH + 'train.csv')

test_df = pd.read_csv( BASE_PATH + 'test.csv')

submission_df = pd.read_csv( BASE_PATH + 'sample_submission.csv')
print("Number of data points in train data frame", train_df.shape)

print("The attributes of train data :", train_df.columns.values)

print('-'*50)

print("Number of data points in test data frame", train_df.shape)

print("The attributes of test data :", test_df.columns.values)
train_profile = ProfileReport(train_df, title='Train Data Profiling Report', html={'style':{'full_width':True}})
train_profile.to_file(output_file="train_profile.html")

train_profile.to_notebook_iframe()
test_profile = ProfileReport(test_df, title='Test Data Profiling Report', html={'style':{'full_width':True}})
test_profile.to_file(output_file="test_profile.html")

test_profile.to_notebook_iframe()
def jaccard_similarity(text1, text2):

    intersection = set(text1).intersection(set(text2))

    union = set(text1).union(set(text2))

    return len(intersection)/len(union)
str1 = 'President greets the press in Chicago'

str2 = 'Obama speaks in Illinois'
jaccard_similarity(str1, str2)
nltk.jaccard_distance(set(str1), set(str2))
1 - nltk.jaccard_distance(set(str1), set(str2))
train_df.sentiment.value_counts()
sns.catplot(x="sentiment", kind="count", palette="ch:.25", data=train_df);
test_df.sentiment.value_counts()
sns.catplot(x="sentiment", kind="count", palette="ch:.25", data=test_df);
# https://www.datacamp.com/community/tutorials/wordcloud-python



def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), color = 'white',

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    

    # Create a word cloud image

    wordcloud = WordCloud(background_color=color,

                   stopwords = stopwords,

                   max_words = max_words,

                   max_font_size = max_font_size,

                   random_state = 42,

                   mask=mask,

                   width=200,

                   height=100,

                   contour_width=2, 

                   contour_color='firebrick')

    

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
train_df = train_df.dropna()



neutral_text = train_df.loc[train_df['sentiment'] == 'neutral', 'text'].append(test_df.loc[test_df['sentiment'] == 'neutral'])

positive_text = train_df.loc[train_df['sentiment'] == 'positive', 'text'].append(test_df.loc[test_df['sentiment'] == 'positive'])

negative_text = train_df.loc[train_df['sentiment'] == 'negative', 'text'].append(test_df.loc[test_df['sentiment'] == 'negative'])

## util to create masked image compatible for WordCloud

wine_mask = np.array(Image.open("../input/wine-mask/wine_mask.png"))



def transform_format(val):

    if val == 0:

        return 255

    else:

        return val

    

# Transform your mask into a new one that will work with the function:

transformed_wine_mask = np.ndarray((wine_mask.shape[0],wine_mask.shape[1]), np.int32)



for i in range(len(wine_mask)):

    transformed_wine_mask[i] = list(map(transform_format, wine_mask[i]))
plot_wordcloud(neutral_text, transformed_wine_mask, max_words=1000, max_font_size=120, title = 'Word Cloud of Neutral tweets', title_size=50)
plot_wordcloud(positive_text,transformed_wine_mask, max_words=1000, max_font_size=100, 

               title = 'Word Cloud of Positive tweets', title_size=50)
plot_wordcloud(negative_text,transformed_wine_mask, max_words=1000, max_font_size=100, 

               title = 'Word Cloud of Negative tweets', title_size=50)
def plot_text_features(data):

    

    fig = go.Figure()

    for val in data:

        fig.add_trace(go.Histogram(x=val['x'],name = val['label']))



    # Overlay both histograms

    fig.update_layout(barmode='stack')

    # Reduce opacity to see both histograms

    fig.update_traces(opacity=0.75)

    fig.show()

    
train_num_words = train_df['text'].apply(lambda x: len(str(x).split(' ')))

test_num_words = test_df['text'].apply(lambda x: len(str(x).split(' ')))

selected_text_num_words = train_df['selected_text'].apply(lambda x: len(str(x).split(' ')))





data_num_words = [

    {'x': train_num_words, 'label': 'Num of words in text of train data'},

    {'x': test_num_words, 'label': 'Num of words in text of test data'},

    {'x': selected_text_num_words, 'label': 'Num of words in selected text'},

]



plot_text_features(data_num_words)
train_num_chars = train_df['text'].apply(lambda x: len(x))

test_num_chars = test_df['text'].apply(lambda x: len(x))

selected_text_num_chars = train_df['selected_text'].apply(lambda x: len(x))





data_num_chars = [

    {'x': train_num_chars, 'label': 'Num of chars in text of train data'},

    {'x': test_num_chars, 'label': 'Num of chars in text of test data'},

    {'x': selected_text_num_chars, 'label': 'Num of chars in selected text'},

]



plot_text_features(data_num_chars)
train_num_uniq_words = train_df['text'].apply(lambda x: len(set(str(x).split(' '))))

test_num_uniq_words = test_df['text'].apply(lambda x: len(set(str(x).split(' '))))

selected_text_num_uniq_words = train_df['selected_text'].apply(lambda x: len(set(str(x).split(' '))))





data_num_uniq_words = [

    {'x': train_num_uniq_words, 'label': 'Num of unique words in text of train data'},

    {'x': test_num_uniq_words, 'label': 'Num of unique words in text of test data'},

    {'x': selected_text_num_uniq_words, 'label': 'Num of unique words in selected text'},

]



plot_text_features(data_num_uniq_words)
from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

stop_words = set(stopwords.words('english')) 





train_num_stop_words = train_df['text'].apply(lambda x: len([w for w in word_tokenize(x) if w in stop_words]))

test_num_stop_words = test_df['text'].apply(lambda x: len([w for w in word_tokenize(x) if w in stop_words]))

selected_text_num_stop_words = train_df['selected_text'].apply(lambda x: len([w for w in word_tokenize(x) if w in stop_words]))





data_num_stop_words = [

    {'x': train_num_stop_words, 'label': 'Num of stop words in text of train data'},

    {'x': test_num_stop_words, 'label': 'Num of stop words in text of test data'},

    {'x': selected_text_num_stop_words, 'label': 'Num of stop words in selected text'},

]



plot_text_features(data_num_stop_words)
from string import punctuation







train_num_puncs = train_df['text'].apply(lambda x: len([w for w in word_tokenize(x) if w in punctuation]))

test_num_puncs = test_df['text'].apply(lambda x: len([w for w in word_tokenize(x) if w in punctuation]))

selected_text_num_puncs = train_df['selected_text'].apply(lambda x: len([w for w in word_tokenize(x) if w in punctuation]))





data_num_puncs = [

    {'x': train_num_puncs, 'label': 'Num of punctuation in text of train data'},

    {'x': test_num_puncs, 'label': 'Num of punctuation in text of test data'},

    {'x': selected_text_num_puncs, 'label': 'Num of punctuation in selected text'},

]



plot_text_features(data_num_puncs)

neutral_text = train_df.loc[train_df['sentiment'] == 'neutral', 'text'].append(test_df.loc[test_df['sentiment'] == 'neutral'])

positive_text = train_df.loc[train_df['sentiment'] == 'positive', 'text'].append(test_df.loc[test_df['sentiment'] == 'positive'])

negative_text = train_df.loc[train_df['sentiment'] == 'negative', 'text'].append(test_df.loc[test_df['sentiment'] == 'negative'])



neutral_text_num_words = neutral_text['text'].apply(lambda x: len(str(x).split(' ')))

positive_text_num_words = positive_text['text'].apply(lambda x: len(str(x).split(' ')))

negative_text_num_words = negative_text['text'].apply(lambda x: len(str(x).split(' ')))





data_num_words = [

    {'x': neutral_text_num_words, 'label': 'Num of words in neutral text'},

    {'x': positive_text_num_words, 'label': 'Num of words in positive text'},

    {'x': negative_text_num_words, 'label': 'Num of words in negative text'},

]



plot_text_features(data_num_words)



# test_df['selected_text'] = test_df['text']

# test_df.loc[test_df.sentiment != 'neutral', 'selected_text'] = test_df.loc[test_df['sentiment'] != 'neutral', 'text'].apply(lambda x: " ".join(x.strip().split(' ')[-5:]))



submission_df['selected_text'] = test_df['text']

submission_df.to_csv("submission.csv", index=False)

display(submission_df.head(10))