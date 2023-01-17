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
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

print("Shape of train dataset is", train.shape)

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

print("Shape of test dataset is", test.shape)
# with below method we can display maximum number of rows and columns we want to display.



pd.set_option('display.max_rows', 10)

pd.set_option('display.max_columns', 10)

train.head(10)
import matplotlib.pyplot as plt

import seaborn as sns



def bar_plot(feature):

    sns.set(style="darkgrid")

    ax = sns.countplot(x=feature , data=train)

    

print("Total number of different target categories is", train.target.value_counts().count())

count_0 = train.target.value_counts()[0]

count_1 = train.target.value_counts()[1]

print("target with count 1 is {}".format(count_1))

print("target with count 0 is {}".format(count_0))

bar_plot("target")
print("Total different categories in keyword is :", train.keyword.value_counts().count())

print("Total different categories in location is :", train.location.value_counts().count())
train.isna().sum()
train[~train["location"].isna()]["location"].tolist()[0:40]
import geopy

import numpy as np

import pycountry



from geopy.geocoders import Nominatim

geolocator = Nominatim("navneet")

def get_location(region=None):

 

    if region:

        try:    

            return geolocator.geocode(region)[0].split(",")[-1] 

        except:

            return region

    return None



train["country"] = train["location"].apply(get_location)
train[~train["country"].isna()]["country"].tolist()[30:50]
train[~train["country"].isna()]["country"].nunique()
train[~train["country"].isna()]["country"].head()
set(train[~train["keyword"].isna()]["keyword"].tolist())
def split_keywords(keyword):

    try:

        return keyword.split("%20")

    except:

        return [keyword]

    



train["keyword"] = train["keyword"].apply(split_keywords)

train[~train["keyword"].isna()]["keyword"].tolist()[100:110]
def count_keywords_in_text(keywords, text):

    if not keywords[0]:

        return 0

    count = 0

    for keyword in keywords:

        each_keyword_count = text.count(str(keyword))

        count = count + each_keyword_count

    return count



train["keyword_count_in_text"] = train.apply(lambda row: count_keywords_in_text(row["keyword"] , row['text']), axis=1)
train.tail()
train["text"].tolist()[0:100]
def get_count_of_hash(text):

    if not text:

        return -1

    return text.count("#")



train["count_#"] = train["text"].apply(get_count_of_hash)
def get_count_of_at_rate(text):

    if not text:

        return -1

    return text.count("@")



train["count_@"] = train["text"].apply(get_count_of_at_rate)
train["count_@"].to_list()[100:110]
train.head()
import re



print("Before---------")

print(train["text"].tolist()[31])



train['text'] = train['text'].str.replace('http:\S+', '', case=False)

train['text'] = train['text'].str.replace('https:\S+', '', case=False)

print("After----------")

print(train["text"].tolist()[31])

import string

exclude = set(string.punctuation)

exclude_hash = {"#"}

exclude = exclude - exclude_hash

print("Length of punctuations to be excluded :",len(exclude))



print("Before---------")

print(train["text"].tolist()[0])



for punctuation in exclude:

  train['text'] = train['text'].str.replace(punctuation, '', regex=True)



print("After----------")

print(train["text"].tolist()[0])
import nltk

nltk.download('stopwords')

from stop_words import get_stop_words

from nltk.corpus import stopwords



stop_words = list(get_stop_words('en'))         #About 900 stopwords

nltk_words = list(stopwords.words('english')) #About 179 stopwords

stop_words = sorted(set(stop_words).union(set(nltk_words)) - exclude_hash)  # removing hash from stop words



print("total stop words to be removed :", len(stop_words))


print("Before--------")

print(train["text"].tolist()[0])

preprocessed_text = []

# tqdm is for printing the status bar

for sentance in train['text'].values:

    sent = ' '.join(e for e in sentance.split() if e not in stop_words)

    preprocessed_text.append(sent.lower().strip())



train["text"] = preprocessed_text

print("After----------")

print(train["text"].tolist()[0])
import spacy



# Initialize spacy 'en' model, keeping only tagger component needed for lemmatization

nlp = spacy.load('en', disable=['parser', 'ner'])



print("Before--------")

print(train["text"].tolist()[2])



lemet_text = []

# tqdm is for printing the status bar

for sentance in train['text'].values:

    sent = " ".join([token.lemma_ for token in nlp(sentance)])

    lemet_text.append(sent.lower().strip())



train["text"] = lemet_text



train["text"] = lemet_text

print("After----------")

print(train["text"].tolist()[2])
nltk.download('punkt')



train['text'] = train.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)

train["text"].tolist()[2]
from gensim.models import Word2Vec

# train model

model = Word2Vec(train.text.values, min_count=1, size = 300)



# summarize vocabulary

words = list(model.wv.vocab)

#print(words)



# save model

model.save('model.bin')

# load model

new_model = Word2Vec.load('model.bin')

print(new_model)
print(model.most_similar('disaster', topn = 20))