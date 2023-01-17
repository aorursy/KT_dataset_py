# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!ls ../input/seaborn/
!pip uninstall -y seaborn

!mkdir -p /tmp/pip/cache/

!cp ../input/seaborn/seaborn-0.11.0-py3-none-any.whl /tmp/pip/cache/
!pip install --no-index --find-links /tmp/pip/cache/ seaborn
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_theme(style="darkgrid")

%matplotlib inline

from plotly import graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff

from collections import Counter



from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator





import nltk

from nltk.corpus import stopwords



from tqdm import tqdm

import os

import nltk

import spacy

import random

from spacy.util import compounding

from spacy.util import minibatch



import warnings

warnings.filterwarnings("ignore")
def color_generator(number_of_colors):

        return ['#'+''.join(random.choice('0123456789ABCDEF') for x in range(6)) for i in range(0,number_of_colors)]
color_generator(6)
train_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
train_data.head()
test_data.head()
train_data.info()
train_data.describe()
train_data.dropna(inplace=True)
train_data.groupby('sentiment')['textID'].count().reset_index()
data = train_data.groupby('sentiment')['textID'].count().reset_index()

data.columns = ['sentiment','count']

sns.barplot(x="sentiment", y="count", data=data)
train_data.head()
train_data.head()
train_data['text_words'] = train_data['text'].apply(lambda x:len(str(x).split()))

train_data['selected_text_words'] = train_data['selected_text'].apply(lambda x:len(str(x).split()))

train_data.head()
# train_data[train_data.text_words <= 1]
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    try:

        return float(len(c)) / (len(a) + len(b) - len(c))

    except:

        print(str1)

        return 1
train_data
train_data['jaccard_similarity'] = train_data[['text','selected_text']].apply(lambda x:jaccard(x['text'],x['selected_text']),axis = 1)
train_data['difference_of_words'] = train_data['text_words'] - train_data['selected_text_words']
train_data
sns.displot(train_data, x="text_words", kind='kde',hue="sentiment", fill=True)
sns.displot(train_data, x="selected_text_words", kind='kde',hue="sentiment", fill=True)
sns.displot(train_data, x="difference_of_words", kind='kde',hue="sentiment", fill=True)
sns.displot(train_data[train_data.sentiment == 'neutral'], x="jaccard_similarity", kind='kde',hue="sentiment", fill=True)
sns.displot(train_data[train_data.sentiment != 'neutral'], x="jaccard_similarity", kind='kde',hue="sentiment", fill=True)
# train_data[train_data.sentiment != 'neutral'].shape

perct = train_data[(train_data.sentiment != 'neutral') & (train_data.jaccard_similarity > 0.95)].shape[0]/train_data[train_data.sentiment != 'neutral'].shape[0]

print("No of Positive and Negative Sentiment Tweets with Jaccard Similarity greater then 0.95 are "+ str(round(perct*100,2)) + "%")
train_data[(train_data.sentiment != 'neutral') & (train_data.jaccard_similarity > 0.95)]['text_words'].quantile([.1, .5,0.8,0.9,1])
import re

import string

def clean_text(text):

    

    

    text = text.lower()

    

#     text = re.sub(r"won\'t", "will not", text)

#     text = re.sub(r"can\'t", "can not", text)



#     # general

#     text = re.sub(r"n\`t", " not", text)

#     text = re.sub(r"\`re", " are", text)

#     text = re.sub(r"\`s", " is", text)

    

#     text = re.sub(r"\'d", " would", text)

#     text = re.sub(r"\`d", " would", text)

    

#     text = re.sub(r"\'ll", " will", text)

#     text = re.sub(r"\`ll", " will", text)

    

#     text = re.sub(r"\`t", " not", text)

#     text = re.sub(r"\'t", " not", text)

    

#     text = re.sub(r"\'ve", " have", text)

#     text = re.sub(r"\`ve", " have", text)

    

#     text = re.sub(r"\'m", " am", text)

#     text = re.sub(r"\`m", " am", text)

    

    text = re.sub(r"\d+", "", text) #removing numbers

    text = text.translate(str.maketrans('', '', string.punctuation)) #removing punctuation

    text = text.strip() #removing white spaces

    

    text = re.sub(r'\b\w{1,3}\b', '', text) #removing words with less then 3 characters



    

    

    return text

    
train_data['clean_text'] = train_data['text'].map(clean_text)

train_data['clean_selected_text'] = train_data['selected_text'].map(clean_text)
from nltk.probability import FreqDist



words = []



for sentence in train_data[train_data.sentiment == 'positive']['clean_text']:

    words.extend(sentence.split())



# print(len(words))

fdist = FreqDist(words)



words = pd.DataFrame(fdist.most_common(len(words)),columns = ['word','count'])

words['len'] = words['word'].str.len()

words.style.background_gradient(cmap='Blues')

top_words = words.head(20)

least_common_words = words.tail(20)



plt.figure(figsize = (15,6))

sns.set_theme(style="whitegrid")

sns.barplot(x="count", y="word", data=top_words).set_title('Most common words in Positive Sentiment Sentences')

plt.figure(figsize = (15,6))

sns.set_theme(style="whitegrid")

sns.barplot(x="count", y="word", data=words[(words['len'] > 8) & (words['count'] > 10)].head(20)).set_title('Unique words in Positive Sentiment Sentences')
tuples = [tuple(x) for x in words[['word','count']].values]

wordcloud = WordCloud(background_color='white').generate_from_frequencies(dict(tuples))

plt.figure(figsize = (12, 12), facecolor = None) 

plt.imshow(wordcloud, interpolation="bilinear")
from nltk.probability import FreqDist



words = []



for sentence in train_data[train_data.sentiment == 'negative']['clean_text']:

    words.extend(sentence.split())



# print(len(words))

fdist = FreqDist(words)



words = pd.DataFrame(fdist.most_common(len(words)),columns = ['word','count'])

words['len'] = words['word'].str.len()

words.style.background_gradient(cmap='Blues')

top_words = words.head(20)



plt.figure(figsize = (15,6))

sns.set_theme(style="whitegrid")

sns.barplot(x="count", y="word", data=top_words).set_title('Most common words in Negative Sentiment Sentences')

plt.figure(figsize = (15,6))

sns.set_theme(style="whitegrid")

sns.barplot(x="count", y="word", data=words[(words['len'] > 8) & (words['count'] > 10)].head(20)).set_title('Unique words in Negative Sentiment Sentences')
tuples = [tuple(x) for x in words[['word','count']].values]

wordcloud = WordCloud(background_color='white').generate_from_frequencies(dict(tuples))

plt.figure(figsize = (12, 12), facecolor = None) 

plt.imshow(wordcloud, interpolation="bilinear")
from nltk.probability import FreqDist



words = []



for sentence in train_data[train_data.sentiment == 'neutral']['clean_text']:

    words.extend(sentence.split())



# print(len(words))

fdist = FreqDist(words)



words = pd.DataFrame(fdist.most_common(len(words)),columns = ['word','count'])

words.style.background_gradient(cmap='Blues')

words['len'] = words['word'].str.len()

top_words = words.head(20)

least_common_words = words.tail(20)



plt.figure(figsize = (15,6))

sns.set_theme(style="whitegrid")

sns.barplot(x="count", y="word", data=top_words).set_title('Most common words in Neutral Sentiment Sentences')

plt.figure(figsize = (15,6))

sns.set_theme(style="whitegrid")

sns.barplot(x="count", y="word", data=words[(words['len'] > 8) & (words['count'] > 10)].head(20)).set_title('Unique words in Neutral Sentiment Sentences')
tuples = [tuple(x) for x in words[['word','count']].values]

wordcloud = WordCloud(background_color='white').generate_from_frequencies(dict(tuples))

plt.figure(figsize = (12, 12), facecolor = None) 

plt.imshow(wordcloud, interpolation="bilinear")
pos_words = {}

neg_words = {}

neutral_words = {}

from sklearn.feature_extraction.text import TfidfVectorizer



tfv = TfidfVectorizer(max_df=0.95, min_df=2,

                                     max_features=10000,

                                     stop_words='english',use_idf=True)







x_pos_idf = tfv.fit_transform(train_data[train_data.sentiment == 'positive']['text'])

# pos_words = dict(zip(tfv.get_feature_names(),np.array(1/((tfv.idf_)))))

pos_words = dict(zip(map(str, tfv.get_feature_names()),1/(2**np.array(tfv.idf_))))



x_neg_idf = tfv.fit_transform(train_data[train_data.sentiment == 'negative']['text'])

neg_words = dict(zip(map(str, tfv.get_feature_names()),1/(2**np.array(tfv.idf_))))



x_neutral_idf = tfv.fit_transform(train_data[train_data.sentiment == 'neutral']['text'])

neutral_words = dict(zip(map(str, tfv.get_feature_names()),1/(2**np.array(tfv.idf_))))



pos_words_new = {}

neg_words_new = {}

neutral_words_new = {}



for word in pos_words:

    if word not in neg_words:neg_words[word] = 0

    if word not in neutral_words:neutral_words[word] = 0

    pos_words_new[word] = pos_words[word] - (neg_words[word]+neutral_words[word])

    

for word in neg_words:

    if(neg_words[word] == 0): continue

    if word not in pos_words:pos_words[word] = 0

    if word not in neutral_words:neutral_words[word] = 0

    neg_words_new[word] = neg_words[word] - (pos_words[word]+neutral_words[word])

    

for word in neutral_words:

    if(neutral_words[word] == 0): continue

    if word not in pos_words:pos_words[word] = 0

    if word not in neg_words:neg_words[word] = 0

    neutral_words_new[word] = neutral_words[word] - (pos_words[word]+neg_words[word])









# pos_count_df = pd.DataFrame(X_train_cv.toarray(), columns=cv.get_feature_names())

# for word in cv.get_feature_names():

    

#     pos_words[word] = 0
def calculate_selected_text(df_row, tol = 0.001):

        tweet = df_row['text']

        sentiment = df_row['sentiment']



        if(sentiment == 'neutral'):

            return tweet

        elif(len(tweet.split()) <= 3):

            return tweet

        else:

            words = tweet.lower().split()

            words_len = len(words)

            subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]

            

            if(sentiment == 'positive'):

                dict_to_use = pos_words_new # Calculate word weights using the pos_words dictionary

            elif(sentiment == 'negative'):

                dict_to_use = neg_words_new



            score = 0

            selection_str = '' # This will be our choice

            lst = sorted(subsets, key = len) # Sort candidates by length

#             print(subsets)



            for i in range(len(subsets)):



                new_sum = 0 # Sum for the current substring



                # Calculate the sum of weights for each word in the substring

                for p in range(len(lst[i])):

                    if(lst[i][p].translate(str.maketrans('','',string.punctuation)) in dict_to_use.keys()):

                        new_sum += dict_to_use[lst[i][p].translate(str.maketrans('','',string.punctuation))]

                        



                # If the sum is greater than the score, update our current selection

                if(new_sum > score + tol):

                    score = new_sum

                    selection_str = lst[i]



            # If we didn't find good substrings, return the whole text

            if(len(selection_str) == 0):

                    selection_str = words



            return ' '.join(selection_str)
# calculate_selected_text(train_data.iloc[4], tol = 0.001)

train_data['derived'] = train_data.apply(calculate_selected_text,axis = 1)
test_data['selected_text'] = test_data.apply(calculate_selected_text,axis = 1)
test_data.head()
test_data[['textID','selected_text']].to_csv('submission.csv',index = False)