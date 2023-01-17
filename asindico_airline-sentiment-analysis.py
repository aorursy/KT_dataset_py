# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib

matplotlib.use('Agg')



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Tweets.csv')
df.head()
positive = df[df['airline_sentiment']=='positive']

negative = df[df['airline_sentiment']=='negative']

nr_group = negative.groupby(['negativereason',"airline"],as_index=False).count()

negative['negativereason'].unique()
negative['airline'].unique()
import matplotlib.pyplot as plt

import seaborn as sns

f,axarray = plt.subplots(1,2,figsize=(15,10))

g = sns.barplot(x= nr_group['negativereason'],y=nr_group['tweet_id'],ax=axarray[0])

for tick in axarray[0].get_xticklabels():

        tick.set_rotation(90)

sns.barplot(x= nr_group['airline'],y=nr_group['tweet_id'],ax=axarray[1])

for tick in axarray[1].get_xticklabels():

        tick.set_rotation(90)

axarray[0].xaxis.set_label_position('top') 

axarray[1].xaxis.set_label_position('top') 
import re 

text = [re.sub(r"@(\S*)",r"",text) for text in df['text'].astype('str')]
text[0:5]
text = [re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'," ",t) for t in text]
#happyness

text = [re.sub(r':\)|:-\)|:D|;\)|;-\)|:-D',"HAPPY",t) for t in text]

#sadness

text = [re.sub(r':\(|:-\(|;\(|;-\(',"SAD",t) for t in text]

text[0:5]
import unicodedata

#text = [re.sub('u[\U0001F602-\U0001F64F]', lambda m: unicodedata.name(m.group()), t,flags=re.UNICODE) for t in text]
text = [re.sub("[^a-zA-Z]"," ",t).lower() for t in text]

text = pd.Series(text)
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word",   

                             tokenizer = None,    

                             preprocessor = None, 

                             stop_words = None,   

                             max_features = 5000) 

tdf = vectorizer.fit_transform(text)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



clf = RandomForestClassifier(n_estimators = 20) 

scores = cross_val_score(clf, tdf.toarray(), df["airline_sentiment"], cv=5)

scores.mean()
vectorizer = CountVectorizer(analyzer = "word",   

                             tokenizer = None,    

                             preprocessor = None, 

                             stop_words = None,

                             ngram_range=(3,3),   

                             max_features = 5000) 

tdf = vectorizer.fit_transform(text) 
clf = RandomForestClassifier(n_estimators = 20) 

scores = cross_val_score(clf, tdf.toarray(), df["airline_sentiment"], cv=5)

scores.mean()
text.head()
import nltk

tokens = [nltk.word_tokenize(t) for t in text]
tokens[0:2]
tags = [nltk.pos_tag(t) for t in tokens]
tags[4]
adjectives = []

for i in range(len(tags)):

    tmp = []

    for j in range(len(tags[i])):

        if tags[i][j][1]=='JJ' or tags[i][j][1]== 'JJS' or tags[i][j][1]=='JJR' or tags[i][j][1]=='VBG' or tags[i][j][1]== 'VBG' or tags[i][j][1]=='VBD':

            tmp.append(tags[i][j][0])

    adjectives.append(tmp)
adjectives[0:5]
adjectives = [" ".join(adj) for adj in adjectives]
from sklearn.feature_extraction.text import CountVectorizer



adj_vec = CountVectorizer(analyzer = "word",   

                             tokenizer = None,    

                             preprocessor = None, 

                             stop_words = None,

                             ngram_range=(1,1),   

                             max_features = 5000) 

adf = adj_vec.fit_transform(adjectives) 
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



clf = RandomForestClassifier(n_estimators = 10) 

scores = cross_val_score(clf, adf.toarray(), df["airline_sentiment"], cv=5)

scores.mean()