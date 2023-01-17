import nltk
from nltk.corpus import movie_reviews
from nltk.classify import ClassifierI
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from statistics import mode
class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers = classifiers
        
    def classify(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self,features):
        votes = []
        for c in _classifiers:
            v = c.classify(features)
            voters.append(v)
        conf = voters.count(mode(voters))
        return conf/len(voters)
    
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

all_words = []
random.shuffle(documents)
for w in movie_reviews.words():
    all_words.append(w)
all_words = nltk.FreqDist(all_words)
    
voca = list(all_words.keys())[:3000]
    
def featureset(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
    
features = [(featureset(rev), category) for (rev, category) in documents]
    


training = features[:1600]
testing = features[1600:]
    
logreg = SklearnClassifier(LinearRegression())
sgd = SklearnClassifier(SGDClassifier())
muli = SklearnClassifier(MultinomialNB())
ber = SklearnClassifier(BernoulliNB())
    
logreg.train(training)
sgd.train(training)
muli.train(training)
ber.train(training)

voteclassifier = VoteClassifier(logreg,sgd,muli,ber)
print("class : ",voteclassifier.classify(testing[0][0])," confidence : ",voteclassifier.confidence(testing[0][0]))
print("class : ",voteclassifier.classify(testing[1][0])," confidence : ",voteclassifier.confidence(testing[1][0]))
print("class : ",voteclassifier.classify(testing[2][0])," confidence : ",voteclassifier.confidence(testing[2][0]))
print("class : ",voteclassifier.classify(testing[3][0])," confidence : ",voteclassifier.confidence(testing[3][0]))
