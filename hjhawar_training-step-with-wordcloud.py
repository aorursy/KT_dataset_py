import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from nltk.classify import ClassifierI
from statistics import mode
read_data = pd.read_table("../input/training_data.tsv",header=0,nrows = 9000)
data = read_data[['Statement','Label']]
list_stat= data['Statement'].tolist()
list_label= data['Label'].tolist()

# read_test_data = pd.read_table("eval_data.tsv",header=0)
# t_data = read_data[['Statement','Label']]
# list_test_stat= data['Statement'].tolist()

documents = []
for statement,label in zip(list_stat,list_label):
    documents.append((word_tokenize(statement),label))

# documents
random.shuffle(documents)
all_words=[]
for x in list_stat:
    all_words= all_words + word_tokenize(x.lower())
# all_words

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets
# testing_set = [find_features(word_tokenize(stmnt)) for stmnt in list_test_stat ]

Naive_classifier = nltk.NaiveBayesClassifier.train(training_set)
with open('naivebayes_wordfeat.pickle','wb') as f:
    pickle.dump(Naive_classifier,f)
    pickle.dump(word_features,f)
#print("Naive Bayes Algo accuracy:", (nltk.classify.accuracy(Naive_classifier, testing_set)))
