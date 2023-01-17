# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input").decode("utf8"))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
% matplotlib inline
from wordcloud import WordCloud
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
##########VIETNAM PRESIDENT DIED Tweet Wordcloud
# Read data
data = pd.read_csv('../input/vietnamtxt.csv')
data1 = pd.read_csv('../input/vietnam.csv')
data.info()
data1.info()

print("There are a total of {} tweets in the dataset".format(len(data)))
print("There are a total of {} tweets in the dataset".format(len(data1)))
print(data.columns)
print(data1.columns)
import numpy as np
import pandas as pd

data = pd.read_csv('../input/vietnamtxt.csv')
data1 = pd.read_csv('../input/vietnam.csv')
print("There are a total of {} tweets in the dataset".format(len(data)))
print("There are a total of {} tweets in the dataset".format(len(data1)))
data['longitude'].replace(np.nan, '0', inplace=True)
data['latitude'].replace(np.nan, '0', inplace=True)
wordcld = pd.Series(data['text'].tolist()).astype(str)
# Most frequent words in the data set. Using a beautiful wordcloud
cloud = WordCloud(width=900, height=900,
                  stopwords=('https', 'https co', 'co'), 
                  colormap='hsv').generate(''.join(wordcld.astype(str)))
plt.figure(figsize=(15, 15))
plt.imshow(cloud)
plt.axis('off')
plt.show()
data['clean_text'] = data.text.str.replace(r'http.*(\s|$)','<link>')
def create_structures(corpus):
    # So we'll construct a dictionary that looks like
    # {'word':[a,a,a,b]}
    # Then when we construct a sentence we'll look at the key and pick
    # a random thing from it's values

    markov_dict = {'<EOT>':[]}

    # Also keep a list of starting_words to easily choose a starting word
    # based on how often they show up
    # Note: we're doing this in a sorta convoluted way, but we're already iterating
    # over all the words, no reason not to do that here while we're at it.

    starting_words = []

    for tweet in corpus.clean_text:
        tok_tweet = tweet.split()
        word_count = len(tok_tweet)
        for index, word in enumerate(tok_tweet):
            if word not in markov_dict.keys():
                markov_dict[word] = []

            if index == word_count - 1:
                markov_dict[word].append("<EOT>")
                #couplet = (tok_tweet[index], "<EOT>")
            else:
                if index == 0:
                    starting_words.append(word)
                markov_dict[word].append(tok_tweet[index+1])
    return markov_dict, starting_words
                
markov_dict, starting_words = create_structures(data)
def write_tweet(starting_word, chain):
    tweet = starting_word
    current_word = starting_word
    
    while len(tweet) <= 140:        
        next_word = np.random.choice(chain[current_word])
        if next_word == '<EOT>':
            return tweet
        
        new_tweet = tweet + ' ' + next_word
        if  len(new_tweet) > 140:
            return tweet
        else:
            tweet = new_tweet
            current_word = next_word

for x in range(0,15):
    starting_word = np.random.choice(starting_words)
    print(write_tweet(starting_word, markov_dict)+"\n")
def simplified_structures(corpus):
    markov_dict = {'<EOT>':[]}
    starting_words = []

    for tweet in corpus.clean_text:
        tweet = tweet.upper()
        tweet = tweet.replace(",","")
        tweet = tweet.replace("\"","")
        tok_tweet = tweet.split()
        word_count = len(tok_tweet)
        for index, word in enumerate(tok_tweet):
            if word not in markov_dict.keys():
                markov_dict[word] = []

            if index == word_count - 1:
                markov_dict[word].append("<EOT>")
            else:
                if index == 0:
                    starting_words.append(word)
                markov_dict[word].append(tok_tweet[index+1])
    return markov_dict, starting_words
simple_markov, simple_start = simplified_structures(data)
print("Original structure has {} total results".format(len(markov_dict)))
print("Simplified structure has {} total results".format(len(simple_markov)))

simplification = np.round(100 * len(simple_markov) / len(markov_dict))
print("We've reduced the dataset to {}% the original size".format(simplification))
for x in range(0,15):
    starting_word = np.random.choice(simple_start)
    print(write_tweet(starting_word, simple_markov)+"\n")
print("Options to follow FACE:")
print("\nNumber of Options to follow THE:")
print(len(set(simple_markov['THE'])))
print("\nOptions to follow MEDIA:")
print(set(simple_markov['MEDIA']))
print("\nOptions to follow CREATION.:")
columns = ['text','id']
data = pd.DataFrame(data, columns=columns)
data.head()
data_by_date = data.copy()
data_by_date['id'] = pd.to_datetime(data['id'], yearfirst=True)
data_by_date['id'] = data_by_date['id'].dt.month
data_by_date = pd.DataFrame(data_by_date.groupby(['id']).size().sort_values(ascending=True).rename('tweets'))
data_by_date
data
