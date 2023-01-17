# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import train_test_split
import string
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
color = sns.color_palette()

%matplotlib inline

eng_stopwords = set(stopwords.words("english"))
pd.options.mode.chained_assignment = None
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("../input/spooky-author-identification/train.zip")
test_df = pd.read_csv("../input/spooky-author-identification/test.zip")
print("Number of rows in train dataset : ",train_df.shape[0])
print("Number of rows in test dataset : ",test_df.shape[0])
train_df.head()
train_df.author.value_counts(normalize=True)*100
cnt_srs = train_df['author'].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Author Name', fontsize=12)
plt.show()
for i in train_df[train_df['author']=='EAP']['text'][:5].values:
    print(i)
    print('****************')
for i in train_df[train_df['author']=='MWS']['text'][:5].values:
    print(i)
    print('****************')
for i in train_df[train_df['author']=='HPL']['text'][:5].values:
    print(i)
    print('****************')
corpus=[]
for i in train_df['text'].values:
    corpus.append(str(i).split(" "))
corpus[:1]
model = word2vec.Word2Vec(corpus, size=100, workers=4)

print(model.wv.most_similar('kill'))

def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota
train_df.author

lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train_df.author.values)
y
xtrain, xvalid, ytrain, yvalid = train_test_split(train_df.text.values, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)
print (xtrain.shape)
print (xvalid.shape)
tfidf_model = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfidf_model.fit(list(xtrain) + list(xvalid))
xtrain_tfidf_model =  tfidf_model.transform(xtrain) 
xvalid_tfidf_model = tfidf_model.transform(xvalid)
ct_vec_model = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')

# Fitting Count Vectorizer to both training and test sets (semi-supervised learning)
ct_vec_model.fit(list(xtrain) + list(xvalid))
xtrain_ct_vec_model =  ct_vec_model.transform(xtrain) 
xvalid_ct_vec_model = ct_vec_model.transform(xvalid)
# Fitting a simple Naive Bayes on Counts
clf = MultinomialNB()
clf.fit(xtrain_ct_vec_model, ytrain)
predictions = clf.predict_proba(xvalid_ct_vec_model)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
predictions = clf.predict_proba(ct_vec_model.transform(['As soon as I opened the door , I gasped..',]))
predictions
