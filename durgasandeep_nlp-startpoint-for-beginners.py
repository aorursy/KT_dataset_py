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
import numpy as np

import pandas as pd

import re

import nltk

import spacy

import string

from collections import Counter

import matplotlib.pyplot as plt

import seaborn as sns
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
# A helper function to clean the data, you can lines based on the dataset. Note that this dataset

# is lot cleaner than usual NLP problems. In usual NLP, you need to keenly look at the data and then clean.

# Remember, the uncleaned text is very crucial for feature engineering(Metafeatures). So keep them in one column

def clean_text(text):

    import re

    text = text.lower()

    text = re.sub(r"i'm", "i am", text)

    text = re.sub(r"you'll", "you will", text)

    text = re.sub(r"i'll", "i will", text)

    text = re.sub(r"she'll", "she will", text)

    text = re.sub(r"he'll", "he will", text)

    text = re.sub(r"he's", "he is", text)

    text = re.sub(r"she's", "she is", text)

    text = re.sub(r"that's", "that is", text)

    text = re.sub(r"what's", "what is", text)

    text = re.sub(r"where's", "where is", text)

    text = re.sub(r"there's", "there is", text)

    text = re.sub(r"here's", "here is", text)

    text = re.sub(r"who's", "who is", text)

    text = re.sub(r"how's", "how is", text)

    text = re.sub(r"\'ll", " will", text)

    text = re.sub(r"\'ve", " have", text)

    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"\'d", " would", text)

    text = re.sub(r"can't", "cannot", text)

    text = re.sub(r"won't", "will not", text)

    text = re.sub(r"don't", "do not", text)

    text = re.sub(r"shouldn't", "should not", text)

    text = re.sub(r"n't", " not", text)

    text = re.sub(r"[^a-z]", " ", text) # This removes anything other than lower case letters(very imp)

    text = re.sub(r"   ", " ", text) # Remove any extra spaces

    return text
train_df['clean_text'] = train_df['text'].apply(clean_text)

test_df['clean_text'] = test_df['text'].apply(clean_text)
# Removal of punctuations(we had already done this in the clean text, but this way also faster to compute)

PUNCT_TO_REMOVE = string.punctuation

def remove_punctuation(text):

    """custom function to remove the punctuation"""

    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))



train_df["clean_text"] = train_df["clean_text"].apply(lambda text: remove_punctuation(text))

test_df["clean_text"] = test_df["clean_text"].apply(lambda text: remove_punctuation(text))

# Removal of stopwords

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def remove_stopwords(text):

    """custom function to remove the stopwords"""

    return " ".join([word for word in str(text).split() if word not in STOPWORDS])



train_df["clean_text"] = train_df["clean_text"].apply(lambda text: remove_stopwords(text))
test_df["clean_text"] = test_df["clean_text"].apply(lambda text: remove_stopwords(text))
# Removal of frequent words(here it seems some important words like fire, so I am removing only top 10.)

# Note: Always print yourself the frequent words and then decide it on how many to remove.

cnt = Counter()

for text in train_df["clean_text"].values:

    for word in text.split():

        cnt[word] += 1

FREQWORDS = set([w for (w, wc) in cnt.most_common(5)])

def remove_freqwords(text):

    """custom function to remove the frequent words"""

    return " ".join([word for word in str(text).split() if word not in FREQWORDS])



train_df["clean_text"] = train_df["clean_text"].apply(lambda text: remove_freqwords(text))





# Removal of frequent words on test data



cnt = Counter()

for text in test_df["clean_text"].values:

    for word in text.split():

        cnt[word] += 1

cnt.most_common(5)

FREQWORDS = set([w for (w, wc) in cnt.most_common(5)])

test_df["clean_text"] = test_df["clean_text"].apply(lambda text: remove_freqwords(text))

# Removal of rare words (this is also important). But I have not tried any efficient way of implementing it.

# Usually people remove last 10 or 20 words. But I usually do remove all the words with frequency == 1.

# If any one better idea for this, please help me. Sharing is caring:)



# # Removing rarewords which has frequency one

# freq = pd.Series(' '.join(train_df['clean_text']).split()).value_counts()

# rare_words = freq[freq <= 1]

# rare_words = list(rare_words.index)



# # Takes too much

# for i in range(len(train_df)):

#     print(i)

#     tokens = train_df['clean_text'][i].split()

#     output = []

#     for token in tokens:

#         if token not in rare_words:

#             output += [token]

#     train_df['clean_text'][i] = output
# Lemmatizing the words using WordNet

from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()

wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}

def lemmatize_words(text):

    pos_tagged_text = nltk.pos_tag(text.split())

    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])



train_df["clean_text"] = train_df["clean_text"].apply(lambda text: lemmatize_words(text))

test_df["clean_text"] = test_df["clean_text"].apply(lambda text: lemmatize_words(text))
train_df.head()
test_df.head()
## Number of words in the text ##

train_df["num_words"] = train_df["text"].apply(lambda x: len(str(x).split()))

test_df["num_words"] = test_df["text"].apply(lambda x: len(str(x).split()))



## Number of unique words in the text ##

train_df["num_unique_words"] = train_df["text"].apply(lambda x: len(set(str(x).split())))

test_df["num_unique_words"] = test_df["text"].apply(lambda x: len(set(str(x).split())))



## Number of characters in the text ##

train_df["num_chars"] = train_df["text"].apply(lambda x: len(str(x)))

test_df["num_chars"] = test_df["text"].apply(lambda x: len(str(x)))



## Number of stopwords in the text ##



eng_stopwords = set(stopwords.words("english"))

train_df["num_stopwords"]=train_df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

test_df["num_stopwords"] = test_df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))



## Number of punctuations in the text ##

train_df["num_punctuations"] =train_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

test_df["num_punctuations"] =test_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )



## Number of title case words in the text ##

train_df["num_words_upper"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

test_df["num_words_upper"] = test_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



## Number of title case words in the text ##

train_df["num_words_title"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

test_df["num_words_title"] = test_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))



## Average length of the words in the text ##

train_df["mean_word_len"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test_df["mean_word_len"] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))



train_df['hastags'] = train_df['text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))

test_df['hastags'] = test_df['text'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
plt.figure(figsize=(12,8))

sns.violinplot(x='target', y='mean_word_len', data=train_df)

plt.xlabel('Event occurred or not', fontsize=12)

plt.ylabel('Mean word length in text', fontsize=12)

plt.title("Number of words by each class", fontsize=15)

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn import ensemble, metrics, model_selection, naive_bayes

import xgboost as xgb
cols_to_drop = ['id', 'text','keyword','location','clean_text']

train_X = train_df.drop(cols_to_drop+['target'], axis=1)

train_y = train_df['target']

test_X = test_df.drop(cols_to_drop, axis=1)
### Fit transform the tfidf vectorizer ###

tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2))

full_tfidf = tfidf_vec.fit_transform(train_df['clean_text'].values.tolist() + test_df['clean_text'].values.tolist())

train_tfidf = tfidf_vec.transform(train_df['clean_text'].values.tolist())

test_tfidf = tfidf_vec.transform(test_df['clean_text'].values.tolist())
def runMNB(train_X, train_y, test_X, test_y, test_X2):

    model = naive_bayes.MultinomialNB()

    #model = linear_model.RidgeClassifier()

    model.fit(train_X, train_y)

    pred_test_y = model.predict(test_X)

    pred_test_y2 = model.predict(test_X2)

    return pred_test_y, pred_test_y2, model
cv_scores = []

pred_full_test = 0

pred_train = np.zeros([train_df.shape[0]])

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):

    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)

    pred_full_test = pred_full_test + pred_test_y

    pred_train[val_index] = pred_val_y

    cv_scores.append(metrics.f1_score(val_y,pred_val_y))

print(cv_scores)
# Add the SVD on tfidf to add some more information(You can also try NMF)

n_comp = 20

svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')

svd_obj.fit(full_tfidf)

train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))

test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))

    

train_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]

test_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]

train_df = pd.concat([train_df, train_svd], axis=1)

test_df = pd.concat([test_df, test_svd], axis=1)

del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd
### Fit transform the tfidf vectorizer ###

tfidf_vec = TfidfVectorizer(ngram_range=(1,7), analyzer='char')

full_tfidf = tfidf_vec.fit_transform(train_df['clean_text'].values.tolist() + test_df['clean_text'].values.tolist())

train_tfidf = tfidf_vec.transform(train_df['clean_text'].values.tolist())

test_tfidf = tfidf_vec.transform(test_df['clean_text'].values.tolist())



cv_scores = []

pred_full_test = 0

pred_train = np.zeros([train_df.shape[0]])

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):

    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)

    pred_full_test = pred_full_test + pred_test_y

    pred_train[val_index] = pred_val_y

    cv_scores.append(metrics.f1_score(val_y,pred_val_y))

print(cv_scores)

# SVD on character tfidf

n_comp = 20

svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')

svd_obj.fit(full_tfidf)

train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))

test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))

    

train_svd.columns = ['svd_char_'+str(i) for i in range(n_comp)]

test_svd.columns = ['svd_char_'+str(i) for i in range(n_comp)]

train_df = pd.concat([train_df, train_svd], axis=1)

test_df = pd.concat([test_df, test_svd], axis=1)

del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd
cols_to_drop = ['id', 'text','keyword','location','clean_text']

train_X = train_df.drop(cols_to_drop+['target'], axis=1)

train_y = train_df['target']

test_X = test_df.drop(cols_to_drop, axis=1)
train_X.head()
from sklearn.model_selection import train_test_split

from sklearn import linear_model

X_train,X_val,y_train,y_val  = train_test_split(train_X,train_y,test_size = 0.2)

model = linear_model.RidgeClassifier()

model.fit(train_X, train_y)

pred_test_y = model.predict(test_X)
sample_submission = pd.read_csv("sample_submission.csv")

sample_submission["target"] = pred_test_y

sample_submission.to_csv("submission_baseline.csv", index=False)