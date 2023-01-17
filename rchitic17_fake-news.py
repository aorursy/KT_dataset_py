# import dependencies
%matplotlib inline
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import re
from nltk.stem.porter import PorterStemmer
import time
from nltk import FreqDist
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from subprocess import check_output
df=pd.read_csv('../input/fake_or_real_news.csv')
df.head()
def initial_clean(text):
    """
    Function to clean text of websites, email addresess and any punctuation
    We also lower case the text
    """
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub("[^a-zA-Z ]", "", text)
    text = text.lower() # lower case the text
    text = nltk.word_tokenize(text)
    return text

stop_words = stopwords.words('english')
def remove_stop_words(text):
    """
    Function that removes all stopwords from text
    """
    return [word for word in text if word not in stop_words]

stemmer = PorterStemmer()
def stem_words(text):
    """
    Function to stem words, so plural and singular are treated the same
    """
    try:
        text = [stemmer.stem(word) for word in text]
        text = [word for word in text if len(word) > 1] # make sure we have no 1 letter words
    except IndexError: # the word "oed" broke this, so needed try except
        pass
    return text

def apply_all(text):
    """
    This function applies all the functions above into one
    """
    return stem_words(remove_stop_words(initial_clean(text)))
# clean text and title and create new column "tokenized"
t1 = time.time()
df['tokenized'] = df['text'].apply(apply_all) + df['title'].apply(apply_all)
t2 = time.time()
print("Time to clean and tokenize", len(df), "articles:", (t2-t1)/60, "min")
# first get a list of all words
all_words = [word for item in list(df['tokenized']) for word in item]
# use nltk fdist to get a frequency distribution of all words
fdist = FreqDist(all_words)
len(fdist) # number of unique words
k = 15000
top_k_words = fdist.most_common(k)
top_k_words[-10:]
# define a function only to keep words in the top k words
top_k_words,_ = zip(*fdist.most_common(k))
top_k_words = set(top_k_words)
def keep_top_k_words(text):
    return [word for word in text if word in top_k_words]
df['tokenized'] = df['tokenized'].apply(keep_top_k_words)
# document length
df['doc_len'] = df['tokenized'].apply(lambda x: len(x))
doc_lengths = list(df['doc_len'])
df.drop(labels='doc_len', axis=1, inplace=True)

print("length of list:",len(doc_lengths),
      "\naverage document length", np.average(doc_lengths),
      "\nminimum document length", min(doc_lengths),
      "\nmaximum document length", max(doc_lengths))
# plot a histogram of document length
num_bins = 1000
fig, ax = plt.subplots(figsize=(12,6));
# the histogram of the data
n, bins, patches = ax.hist(doc_lengths, num_bins, normed=1)
ax.set_xlabel('Document Length (tokens)', fontsize=15)
ax.set_ylabel('Normed Frequency', fontsize=15)
ax.grid()
ax.set_xticks(np.logspace(start=np.log10(50),stop=np.log10(2000),num=8, base=10.0))
plt.xlim(0,2000)
ax.plot([np.average(doc_lengths) for i in np.linspace(0.0,0.0035,100)], np.linspace(0.0,0.0035,100), '-',
        label='average doc length')
ax.legend()
ax.grid()
fig.tight_layout()
plt.show()
# only keep articles with more than 30 tokens, otherwise too short
df = df[df['tokenized'].map(len) >= 40]
# make sure all tokenized items are lists
df = df[df['tokenized'].map(type) == list]
df.reset_index(drop=True,inplace=True)
print("After cleaning and excluding short aticles, the dataframe now has:", len(df), "articles")
df.head()
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2)

from nltk.corpus import stopwords 
train.columns.values
train.head()
import re
def refineWords(s):
    letters_only = re.sub("[^a-zA-Z]", " ", s) 
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    #print( " ".join( meaningful_words ))
    return( " ".join( meaningful_words ))

train["text"].fillna(" ",inplace=True)    
train["text"] = train["text"].apply(refineWords)
train["title"].fillna(" ",inplace=True)    
train["title"] = train["title"].apply(refineWords)

train_two = train.copy()
train.head()
train = train_two.copy()
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 
#print(train_one["title"].head())
#temp  = (vectorizer.fit_transform(train_one["text"]))
#train_one["text"] = temp.to_array()
train["text"] = vectorizer.fit_transform(train["text"]).toarray()
train["title"] = vectorizer.fit_transform(train["title"]).toarray()
train.head()
#print(train_one["isSpam"])
from sklearn.ensemble import RandomForestClassifier
#forest = RandomForestClassifier(n_estimators = 100)
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
features_forest = train[["text", "title"]].values
my_forest = forest.fit(features_forest, train["label"])
target = train["label"].values
print(my_forest.score(features_forest, target))
test["text"].fillna(" ",inplace=True)    
test["text"] = test["text"].apply(refineWords)
test["title"].fillna(" ",inplace=True)    
test["title"] = test["title"].apply(refineWords)

test_two = test.copy()

test["text"] = vectorizer.fit_transform(test["text"]).toarray()
test["title"] = vectorizer.fit_transform(test["title"]).toarray()
test_features = test[["text", "title"]].values
my_prediction = my_forest.predict(test_features)
print(len(my_prediction),len(test["label"]))
count = 0
pred = my_prediction.tolist()
test_spam = test["label"].tolist()
for i in range(len(pred)):
    if pred[i] == test_spam[i]:
        count += 1
print(count,float(count)/len(my_prediction))
#print(my_prediction)
#print(test_spam)
