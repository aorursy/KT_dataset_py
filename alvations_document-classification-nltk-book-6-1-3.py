import random

import nltk

from nltk.corpus import movie_reviews

from nltk import FreqDist, NaiveBayesClassifier



# Read the documents.

documents = [(list(movie_reviews.words(fileid)), category)

            for category in movie_reviews.categories()

             for fileid in movie_reviews.fileids(category)]



# Shuffle the documents.

random.shuffle(documents)
all_words = FreqDist(w.lower() for w in movie_reviews.words())

word_features = list(all_words)[:2000] # [1]: 2000 most frequent words in the overall corpus



def document_features(document):   # [2]: checks whether word is present in a given document

    document_words = set(document) # [3]: set of all words in a document

    features = {}

    for word in word_features:

        features['contains({})'.format(word)] = (word in document_words)

    return features
print(document_features(movie_reviews.words('pos/cv957_8737.txt'))) 
featuresets = [(document_features(d), c) for (d,c) in documents]

train_set, test_set = featuresets[100:], featuresets[:100]

classifier = NaiveBayesClassifier.train(train_set)



print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(5)