import random
import pickle

import nltk
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI, MultiClassifierI
from nltk.classify import NaiveBayesClassifier

import sklearn
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from statistics import mode 
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
def find_features(document, top_n=3000):
    word_features = list(all_words.keys())[:top_n]
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

def train_test_split(documents, random_seed=0, split_on=0.95, top_n=3000):
    custom_random = random.Random(random_seed)
    custom_random.shuffle(documents)
    featuresets = [(find_features(rev, top_n), category) for (rev, category) in documents]
    split_on_int = int(len(featuresets) * split_on)
    training_set = featuresets[:split_on_int]
    testing_set = featuresets[split_on_int:]
    return training_set, testing_set
training_set, testing_set = train_test_split(documents)
class VotraClassifier:
    def __init__(self, *classifiers_obj):
        self._classifiers_obj = classifiers_obj
        
    def train(self, training_set):
        self._classifiers = {}
        for clf_obj in self._classifiers_obj:
            if hasattr(clf_obj, '__name__') and clf_obj.__name__ == 'NaiveBayesClassifier':
                clf_name = 'NaiveBayesClassifier'
                print('Training', clf_name +'\t'+ str(clf_obj))
                clf_obj = NaiveBayesClassifier.train(training_set)
            else:
                clf_name = str(clf_obj).split('(')[1]
                print('Training', clf_name +'\t'+ str(clf_obj))
                clf_obj.train(training_set)
            self._classifiers[clf_name] = clf_obj

    def evaluate(self, testing_set):
        documents, labels = zip(*testing_set)
        predictions = self.classify_documents(documents)
        correct = [y == y_hat for y, y_hat in zip(labels, predictions)]
        if correct:
            return sum(correct) / len(correct)
        else:
            return 0
            
    def classify_documents(self, documents):
        return [self.classify_many(doc) for doc in documents]
        
    def classify_many(self, features):
        votes = []
        for clf_name, clf  in self._classifiers.items():
            v = clf.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
voted_classifier = VotraClassifier(NaiveBayesClassifier,
                                    SklearnClassifier(MultinomialNB()), 
                                    SklearnClassifier(BernoulliNB()), 
                                    SklearnClassifier(LogisticRegression()),
                                    SklearnClassifier(SGDClassifier()),
                                    SklearnClassifier(LinearSVC()),
                                    SklearnClassifier(NuSVC())
                                    )

voted_classifier.train(training_set)
for clf_name, clf in voted_classifier._classifiers.items():
    print(clf_name, '\t', nltk.classify.accuracy(clf, testing_set)*100)
print('-------------------------')
print('VotedClassifier', '\t', voted_classifier.evaluate(testing_set)*100)