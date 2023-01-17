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



import warnings

warnings.filterwarnings('ignore')



import copy

from collections import Counter

import collections

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import nltk, re, string, collections

from nltk.util import ngrams # function for making ngrams



import random

import token

# import bs4 as bs

# import urllib.request



from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 

from wordcloud import WordCloud, STOPWORDS
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')

display(train.head())

display(test.head())
print('train.text NA', train.text.isna().sum())

print('test.text NA', test.text.isna().sum())
train_text = train[['text']]

test_text = test[['text']]
combined_dataset = pd.concat(objs = [train_text, test_text], axis = 0)

print('combined', combined_dataset.shape)
combined_dataset['text'] = combined_dataset['text'].str.replace("[^a-zA-Z#]", " ")

combined_dataset['text'] = combined_dataset['text'].apply(lambda x : " ".join(x.lower() for x in x.split()) )

combined_dataset['text'] = combined_dataset['text'].apply(lambda x: x.replace('#',''))
# Removing URLs, html tags, and emojis

example ="New competition launched :https://www.kaggle.com/c/nlp-getting-started <div> <h1>Real or Fake</h1> <p>Kaggle </p> </div> I am a #king Omg another Earthquake 😔😔"

re_list = ['https?://\S+|www\.\S+', '<.*?>', 

                           "["u"\U0001F600-\U0001F64F"  # emoticons

                              u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                              u"\U0001F680-\U0001F6FF"  # transport & map symbols

                              u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                              u"\U00002702-\U000027B0"

                              u"\U000024C2-\U0001F251"

                           "]+"]

concat_re = re.compile( '|'.join( re_list) ).sub(f'', example)

concat_re
# Sample test on dataframe

df = {'Text':['https://www.kaggle.com/c/nlp-getting-started co', '<div> <h1>Sharingan or Byakugan</h1>', 'Summoning Jutsu 😔😔']}

df = pd.DataFrame(df)

print('Before')

display(df)

df['Text'] = df['Text'].apply(lambda x: re.compile( '|'.join( re_list)).sub(f'', x))

print('After')

display(df)
# Applying on the combined dataset

combined_dataset['text'] = combined_dataset['text'].apply(lambda x: re.compile( '|'.join( re_list)).sub(f'', x))
# Removing punctuations

def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



example="I ***will become a H^&o*#k--,a...#g#e ##"

print(remove_punct(example))
# Applying on the combined dataset

combined_dataset['text'] = combined_dataset['text'].apply(lambda x : remove_punct(x))
train_text_len = len(train_text)

train_text = copy.copy(combined_dataset[:train_text_len])

test_text = copy.copy(combined_dataset[train_text_len:])

print('combined dataset ', combined_dataset.shape)

print('train_text        ', train_text.shape)

print('test_text         ', test_text.shape)
stopwords = stopwords.words('english')

stopwords = set(STOPWORDS)
def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='orange',

        stopwords=stopwords,

        max_words=100,

        max_font_size=40, 

        scale=3,

        #random_state=1

                         ).generate(str(data))



    fig = plt.figure(1, figsize=(20, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.4)



    plt.imshow(wordcloud,interpolation="bilinear")

    plt.show()
show_wordcloud(train_text, "Commonly occuring words - train_text")

show_wordcloud(train_text, "Commonly occuring words - test")
# Train: Converting the entries to a list

train_text_list = train_text['text'].tolist()

train_text_list
# Train: Creating a counter for the types of entries under text and printing the occurrences

Counter = Counter(train_text_list) 

most_occurences = Counter.most_common()  

print(most_occurences) 
# Train: Converting the occurrences into a dataframe

pd.set_option('display.max_colwidth', 150)

train_text_summary = pd.DataFrame(most_occurences, columns = ['Content' , 'Count']) 

train_text_summary.head(10)
with open('train_text.txt', 'w') as f:

    for item in train_text_list:

        f.write("%s\n" % item)
# Train: reading input file; the encoding is specified here 

file = open('train_text.txt', encoding="utf8")

a= file.read()



# Stopwords

# stopwords = set(line.strip() for line in open('stopwords.txt')) - if you have custom defined txt file of stop words

stopwords = stopwords

stopwords = stopwords.union(set(['mr','mrs','one','two','said','http','https','a','b','c','d','e','f','g','h','i','j','k',

                                 'l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','co','via','re']))

# Instantiate a dictionary, and for every word in the file, 

# Add to the dictionary if it doesn't exist. If it does, increase the count.

wordcount = {}

# To eliminate duplicates, remember to split by punctuation, and use case demiliters.

for word in a.lower().split():

    word = word.replace(".","")

    word = word.replace(",","")

    word = word.replace(":","")

    word = word.replace("\"","")

    word = word.replace("!","")

    word = word.replace("â€œ","")

    word = word.replace("â€˜","")

    word = word.replace("*","")

    if word not in stopwords:

        if word not in wordcount:

            wordcount[word] = 1

        else:

            wordcount[word] += 1

# Print most common word

# n_print = int(input("How many most common words to print: ")) - Use this to get the option to enter the number of most common words you want to be populated

# Disabled for kernel commit purpose



print("\nOK. The {} most common words are as follows\n".format(10))

word_counter = collections.Counter(wordcount)

for word, count in word_counter.most_common(10):

    print(word, ": ", count)

# Close the file

file.close()
# Train: Visual of above selection

lst = word_counter.most_common(10)

df = pd.DataFrame(lst, columns = ['Word', 'Count'])

colors = list('rgbkymc')

df.plot.barh(x='Word',y='Count', color = colors, figsize = (20,8))
def listToString(l):  

    # initialize an empty string 

    s = " " 

    # return string   

    return (s.join(l))



string = listToString(train_text_list)
# Train: getting individual words

tokenized = string.split()

tokenized
train_text_filtered = [w for w in tokenized if not w in stopwords] 

  

train_text_filtered = [] 

  

for w in tokenized: 

    if w not in stopwords: 

        train_text_filtered.append(w) 

  

print(train_text_filtered) 
# Train

# Getting list of all the bi-grams

train_bigrams = ngrams(train_text_filtered, 2)

train_bigrams



# Getting the frequency of each bigram

train_bigrams_freq = collections.Counter(train_bigrams)



# The ten most popular bigrams

train_bigrams_freq.most_common(10)
train_trigrams = ngrams(train_text_filtered, 3)

train_trigrams



train_trigrams_freq = collections.Counter(train_trigrams)



train_trigrams_freq.most_common(10)
train_quadgrams = ngrams(train_text_filtered, 4)

train_quadgrams



train_quadgrams_freq = collections.Counter(train_quadgrams)



train_quadgrams_freq.most_common(10)
train_pentagrams = ngrams(train_text_filtered, 5)

train_pentagrams



train_pentagrams_freq = collections.Counter(train_pentagrams)



train_pentagrams_freq.most_common(10)