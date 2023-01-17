def gender_features(word):

    return {'last_letter': word[-1]}



gender_features('Louis')
import nltk

from nltk.corpus import names

labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])



import random

random.shuffle(labeled_names)



labeled_names[:10] #list first 10 names
featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]

featuresets[:10] 
train_set, test_set = featuresets[500:], featuresets[:500]

classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier.classify(gender_features('Gandalf')), classifier.classify(gender_features('Bilbo'))
classifier.classify(gender_features('Katniss')), classifier.classify(gender_features('Dumbledore')) 
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(5)
def gender_features2(name):

    features = {}

    features["first_letter"] = name[0].lower()

    features["last_letter"] = name[-1].lower()

    for letter in 'abcdefghijklmnopqrstuvwxyz':

        features["count({})".format(letter)] = name.lower().count(letter)

        features["has({})".format(letter)] = (letter in name.lower())

    return features



gender_features2('Tommy') 
featuresets = [(gender_features2(n), gender) for (n, gender) in labeled_names]

train_set, test_set = featuresets[500:], featuresets[:500]

classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, test_set))
train_names = labeled_names[1500:]

devtest_names = labeled_names[500:1500]

test_names = labeled_names[:500]
train_set = [(gender_features2(n), gender) for (n, gender) in train_names]

devtest_set = [(gender_features2(n), gender) for (n, gender) in devtest_names]

test_set = [(gender_features2(n), gender) for (n, gender) in test_names]

classifier = nltk.NaiveBayesClassifier.train(train_set) #[1]

print(nltk.classify.accuracy(classifier, devtest_set)) #[2]
errors = []

for (name, tag) in devtest_names:

    guess = classifier.classify(gender_features2(name))

    if guess != tag:

        errors.append( (tag, guess, name) )
len(errors) # number of mislabeled names
for (tag, guess, name) in sorted(errors):

    print('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name))
def gender_features(word):

    return {'suffix1': word[-1:], 'suffix2': word[-2:]}
train_set = [(gender_features(n), gender) for (n, gender) in train_names]

devtest_set = [(gender_features(n), gender) for (n, gender) in devtest_names]

classifier = nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier, devtest_set))
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)

             for category in movie_reviews.categories()

             for fileid in movie_reviews.fileids(category)]



random.shuffle(documents)
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())

word_features = list(all_words)[:2000]



def document_features(document):

    document_words = set(document)

    features = {}

    for word in word_features:

        features['contains({})'.format(word)] = (word in document_words)

    return features
featuresets = [(document_features(d), c) for (d,c) in documents]

train_set, test_set = featuresets[100:], featuresets[:100]

classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(10)
import pandas as pd

import nltk



data = pd.read_csv('../input/Oct-7-2019-Supreme-Court-Confirmation-Hearing-Transcript.csv', encoding='latin1')



data = data.reset_index()



data.head()
words = data['Statements'].apply(nltk.word_tokenize)
words # the tokenized words are structured as a dataframe.
words = list(words) # the tokenized words are now structured as a list of lists
import itertools

all_words = (list(itertools.chain.from_iterable(words))) # we will use this information below when we build our naive bayes classifier.
data['Statements'] = data['Statements'].apply(lambda x: x.lower())

data['Statements'] = data['Statements'].apply(nltk.word_tokenize)
documents = list(zip(data['Statements'], data['Speaker (Party)(or nominated by)']))
all_words = nltk.FreqDist(w.lower() for w in all_words) #[1]

word_features = list(all_words)[:2000]



def document_features(document): #[2]

    document_words = set(document)

    features = {}

    for word in word_features:

        features['contains({})'.format(word)] = (word in document_words)

    return features
featuresets = [(document_features(d), c) for (d,c) in documents]

train_set, test_set = featuresets[100:], featuresets[:100]

classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier.show_most_informative_features(5)
print(nltk.classify.accuracy(classifier, test_set))
data = pd.read_csv('../input/Oct-7-2019-Supreme-Court-Confirmation-Hearing-Transcript.csv', encoding='latin1')

data = data.reset_index()





data = data[data['Statements'].notnull() & (data['Year'] == 2017)] # only hearings from 2017

words = data['Statements'].apply(nltk.word_tokenize)

words = list(words)



all_words = (list(itertools.chain.from_iterable(words)))





data['Statements'] = data['Statements'].apply(lambda x: x.lower())

data['Statements'] = data['Statements'].apply(nltk.word_tokenize)





documents = list(zip(data['Statements'], data['Speaker (Party)(or nominated by)']))



all_words = nltk.FreqDist(w.lower() for w in all_words)

word_features = list(all_words)[:2000]



def document_features(document):

    document_words = set(document)

    features = {}

    for word in word_features:

        features['contains({})'.format(word)] = (word in document_words)

    return features



featuresets = [(document_features(d), c) for (d,c) in documents]

train_set, test_set = featuresets[100:], featuresets[:100]

classifier = nltk.NaiveBayesClassifier.train(train_set)



print(nltk.classify.accuracy(classifier, test_set))
