# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from bs4 import BeautifulSoup
from gensim.models import word2vec
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm_notebook
import re
tsvread_params = {
    "delimiter": '\t',
    "quoting": 3,
    "header": 0
}
train = pd.read_csv("../input/labeledTrainData.tsv", **tsvread_params)
test = pd.read_csv("../input/testData.tsv", **tsvread_params)
unlabeled_train = pd.read_csv("../input/unlabeledTrainData.tsv", **tsvread_params)
print("Train data shape:", train.shape)
print("Test data shape:", test.shape)
print("Unlabeled data shape:", unlabeled_train.shape)
print("Columns:", train.columns.values)
def clean_review(raw_text, remove_stops=True, result_as_list=True):
    # remove HTML
    text = BeautifulSoup(raw_text, "lxml").get_text()
    
    # remove non-letters
    letters_text = re.sub(r"[^a-zA-Z]", ' ', text)
    
    # lower case and split
    words = letters_text.lower().split()
    
    # remove stop words
    if remove_stops:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    if not result_as_list:
        words = ' '.join(words)
    return words
sentences = [clean_review(review, remove_stops=False) for review in unlabeled_train["review"].values]
num_features = 3000
model_params = {
    "size": num_features, # Word vector dimensionality
    "min_count": 40,      # Minimum word count
    "workers": 4,         # Number of threads to run in parallel
    "window": 10,         # Context window size
    "sample": 1e-3,       # Downsample setting for frequent words
}

w2v = word2vec.Word2Vec(sentences, **model_params)
w2v.doesnt_match("king count queen princess".split())
w2v.most_similar("alien")
def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    # 
    # Loop through the reviews
    for review in tqdm_notebook(reviews, desc="Reviews preprocessed"):
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        #
        # Increment the counter
        counter += 1
    return reviewFeatureVecs
print("Creating average feature vecs for train reviews")
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(clean_review(review))

train_features = getAvgFeatureVecs(clean_train_reviews, w2v, num_features)

print("Creating average feature vecs for test reviews")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(clean_review(review))

test_features = getAvgFeatureVecs(clean_test_reviews, w2v, num_features)
model = RandomForestClassifier(n_estimators=100)
model.fit(train_features, train["sentiment"])
y_pred = model.predict(test_features)
output = pd.DataFrame(data={"id": test["id"], "sentiment": y_pred})
output.to_csv("out.csv", index=False, quoting=3)
