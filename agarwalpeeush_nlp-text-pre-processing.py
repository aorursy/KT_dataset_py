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
data = pd.read_csv('/kaggle/input/customer-support-on-twitter/twcs/twcs.csv', nrows=5000)
data.head()
data.shape
df = data[['text']].astype(str)
df.head()
df.shape
import nltk
import string
import re
df['text_lower'] = df['text'].str.lower()
df.head()
PUNCTUATIONS = string.punctuation
PUNCTUATIONS
def remove_punctuations(text):
    return text.translate(str.maketrans('','',PUNCTUATIONS))
str.maketrans('','', PUNCTUATIONS)
df['text_wo_punct'] = df['text_lower'].apply(lambda text: remove_punctuations(text))
df.head()
# Remove text_lower column
df.drop('text_lower', axis=1, inplace=True)
df.head()
from nltk.corpus import stopwords

" ".join(stopwords.words('english'))
STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in text.split() if word not in STOPWORDS])

remove_stopwords('I love my country'.lower())
df['text_wo_stop'] = df['text_wo_punct'].apply(lambda text: remove_stopwords(text))
df.head()
df.drop('text_wo_punct', axis=1, inplace=True)
df.head()
from collections import Counter

cnt = Counter()

for text in df['text_wo_stop'].values:
    for word in text.split():
        cnt[word] += 1

cnt.most_common(10)
FREQWORDS = set([word for (word, wordcnt) in cnt.most_common(10)])
def remove_freqwords(text):
    return " ".join([word for word in text.split() if word not in FREQWORDS])
df['text_wo_stopfreq'] = df['text_wo_stop'].apply(lambda text: remove_freqwords(text))
df.head()
df.drop('text_wo_stop', axis=1, inplace=True)
df.head()
n_rare_words = 10
RAREWORDS = set([word for (word, wordcnt) in cnt.most_common()[:-n_rare_words-1:-1]])
RAREWORDS
def remove_rarewords(text):
    return " ".join([word for word in text.split() if word not in RAREWORDS])
df['text_wo_stopfreqrare'] = df['text_wo_stopfreq'].apply(lambda text: remove_rarewords(text))
df.head()
df.drop('text_wo_stopfreq', axis=1, inplace=True)
df.head()
# To remove URLs, we can use Re package
def remove_Urls(text):
    urls_pattern = re.compile(r'https?://\S+|www.\S+')
    return urls_pattern.sub(r'', text)
df['text_wo_stopfreqrare_urls'] = df['text_wo_stopfreqrare'].apply(lambda text: remove_Urls(text))
df.head(10)
df.drop('text_wo_stopfreqrare', axis=1, inplace=True)
df.head()
from bs4 import BeautifulSoup

#def remove_html(text):
#    html_pattern = re.compile(r'<.*?>')
#    return html_pattern.sub(r'', text)
    
def remove_html(text):
    return BeautifulSoup(text, 'lxml').text
df['text_cleaned'] = df['text_wo_stopfreqrare_urls'].apply(lambda text: remove_html(text))
df.head(10)
df.drop('text_wo_stopfreqrare_urls', axis=1, inplace=True)
df.tail(10)
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}

def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])
df['text_lemmatized'] = df['text_cleaned'].apply(lambda text: lemmatize_words(text))
df.head()
