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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
fake = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv', delimiter = ',')
true = pd.read_csv('../input/fake-and-real-news-dataset/True.csv', delimiter = ',')
fake.head()
fake_text = fake['text'].tolist()
true_text = true['text'].tolist()
def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text
fake_string = combine_text(fake_text)
true_string = combine_text(true_text)
# remove words inside brackets, punctuation and words that have number in them
import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub('\(.*?\)', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
fake_cleaned_text = clean_text(fake_string)
true_cleaned_text = clean_text(true_string)
combined_clean_text_dict = {"fake":fake_cleaned_text, "true":true_cleaned_text}
data_df = pd.DataFrame.from_dict(combined_clean_text_dict, orient='index', columns=['text'])
data_df.head()
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_df.text)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_df.index
data_dtm
# Find the top 30 words used by fake news and true news
data_dtm = data_dtm.transpose()
top_dict = {}
for c in data_dtm.columns:
    top = data_dtm[c].sort_values(ascending=False).head(30)
    top_dict[c]= list(zip(top.index, top.values))

top_dict
# add the top 30 word from each label
words = []
for label in data_dtm.columns:
    top = [word for (word, count) in top_dict[label]]
    for t in top:
        words.append(t)
words
from collections import Counter

Counter(words).most_common()
from sklearn.feature_extraction import text

# add the common words to stop word list
new_stop_words = [word for word, count in Counter(words).most_common() if count >=2]
stop_words = text.ENGLISH_STOP_WORDS.union(new_stop_words)

# repeat the process using counterVectorizer
cv = CountVectorizer(stop_words = stop_words)
data_cv = cv.fit_transform(data_df.text)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_df.index

data_dtm = data_dtm.transpose()
top_dict = {}
for c in data_dtm.columns:
    top = data_dtm[c].sort_values(ascending=False).head(30)
    top_dict[c]= list(zip(top.index, top.values))

top_dict
# fakew news tends to use more verbs and adverbs, and true news uses more adjectives and nouns
from wordcloud import WordCloud

wc = WordCloud(stopwords = stop_words, background_color = "white", colormap = "Dark2", max_font_size = 100, random_state = 40)
plt.rcParams['figure.figsize'] = [16,80]


for index, label in enumerate(data_dtm.columns):
    wc.generate(data_df.text[label])
    plt.subplot(3,4, index+1)
    plt.imshow(wc,interpolation = "bilinear")
    plt.axis("off")
    plt.title(label)
    
plt.show()
import string

pun_count_fake, pun_count_true = 0,0

for i in fake_string:
    if i in string.punctuation:
        pun_count_fake+=1
for i in true_string:
    if i in string.punctuation:
        pun_count_true+=1
pun_count_fake
pun_count_true
# fake news use 27.6798% more puctuations than true news