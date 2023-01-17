import nltk

# nltk.download()
from nltk.book import *
text2
sent2
text1.concordance("cannibal")
text2.concordance("affection")
text3.concordance("lived")
text1.similar("monstrous")
text2.similar("monstrous")
text2.common_contexts(["monstrous", "very"])
text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])
text2.dispersion_plot(['liberty','constitution'])
len(text3)
sorted(set(text3))
len(set(text3))
len(set(text3)) / len(text3)
from nltk.tokenize import sent_tokenize, word_tokenize



example_text = "In a small village, a little boy lived with his father and mother. He was the only son. The parents of the little boy were very depressed due to his bad temper."

print(sent_tokenize(example_text))
print(word_tokenize(example_text))
from nltk.corpus import stopwords

print(set(stopwords.words('english')))
example_sent = "The boy used to get angry very soon and taunt others with his words. His bad temper made him use words that hurt others. He scolded kids, neighbours and even his friends due to anger. His friends and neighbours avoided him, and his parents were really worried about him."



stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]



# OR

# filtered_sentence = []



# for w in word_tokens:

#     if w not in stop_words:

#         filtered_sentence.append(w)



print(word_tokens)
print(filtered_sentence)
from nltk.stem import PorterStemmer

ps = PorterStemmer()

example_words = ["python","pythoner","pythoning","pythoned","pythonly"]



for w in example_words:

    print(ps.stem(w))
# Now let's try stemming a typical sentence, rather than some words:



new_text = "He asked his son to hammer one nail to the fence every time he became angry and lost his temper. The little boy found it amusing and accepted the task."



words = word_tokenize(new_text)

strem_sent=[]



for w in words:

    strem_sent.append(ps.stem(w))

    

print(strem_sent)
from nltk.corpus import state_union

from nltk.tokenize import PunktSentenceTokenizer
train_text = state_union.raw("2005-GWBush.txt")

sample_text = state_union.raw("2006-GWBush.txt")
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)
def process_content():

    try:

        for i in tokenized[:5]:

            words = nltk.word_tokenize(i)

            tagged = nltk.pos_tag(words)

            print(tagged)



    except Exception as e:

        print(str(e))





process_content()
train_text = state_union.raw("2005-GWBush.txt")

sample_text = state_union.raw("2006-GWBush.txt")



custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)



def process_content():

    try:

        for i in tokenized:

            words = nltk.word_tokenize(i)

            tagged = nltk.pos_tag(words)

            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""

            chunkParser = nltk.RegexpParser(chunkGram)

            chunked = chunkParser.parse(tagged)

            print(chunked)

    except Exception as e:

        print(str(e))



# process_content()
# chunked=process_content()

# for subtree in chunked.subtrees():

#     print(subtree)
# chunked=process_content()

# subtree_chunk=[]

# for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):

#     subtree_chunk.append(subtree)

# print(subtree_chunk[:1])
train_text = state_union.raw("2005-GWBush.txt")

sample_text = state_union.raw("2006-GWBush.txt")



custom_sent_tokenizer = PunktSentenceTokenizer(train_text)



tokenized = custom_sent_tokenizer.tokenize(sample_text)



def process_content():

    try:

        for i in tokenized[5:]:

            words = nltk.word_tokenize(i)

            tagged = nltk.pos_tag(words)



            chunkGram = r"""Chunk: {<.*>+}

                                    }<VB.?|IN|DT|TO>+{"""



            chunkParser = nltk.RegexpParser(chunkGram)

            chunked = chunkParser.parse(tagged)

            print(chunked)



    except Exception as e:

        print(str(e))
# To Experiment Uncomment below line

# process_content()
train_text = state_union.raw("2005-GWBush.txt")

sample_text = state_union.raw("2006-GWBush.txt")



custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)



def process_content():

    try:

        for i in tokenized[5:]:

            words = nltk.word_tokenize(i)

            tagged = nltk.pos_tag(words)

            namedEnt = nltk.ne_chunk(tagged, binary=True)

            print(namedEnt)

    except Exception as e:

        print(str(e))

# To test uncomment below line

# process_content()
def process_content():

    try:

        for i in tokenized[5:7]:

            words = nltk.word_tokenize(i)

            tagged = nltk.pos_tag(words)

            namedEnt = nltk.ne_chunk(tagged, binary=False)

            print(namedEnt)

    except Exception as e:

        print(str(e))





process_content()
from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()



print(lemmatizer.lemmatize("dog"))

print(lemmatizer.lemmatize("dogi"))

print(lemmatizer.lemmatize("keeps"))

print(lemmatizer.lemmatize("rocks"))

print(lemmatizer.lemmatize("python"))

print(lemmatizer.lemmatize("better", pos="a"))

print(lemmatizer.lemmatize("best", pos="a"))

print(lemmatizer.lemmatize("run"))

print(lemmatizer.lemmatize("run",'v'))
print(nltk.__file__)
from nltk.corpus import brown

brown.words() # Returns a list of strings
len(brown.words()) # No. of words in the corpus
brown.sents() # Returns a list of list of strings 
from nltk.corpus import gutenberg



# sample text

sample = gutenberg.raw("bible-kjv.txt")



tok = sent_tokenize(sample)



for x in range(10):

    print(tok[x])
from nltk.corpus import wordnet
syns = wordnet.synsets("operation")

print(syns[0].name())
print(syns[0].lemmas()[0].name())
print(syns[0].definition())
print(syns[0].examples())
synonyms = []

antonyms = []



for syn in wordnet.synsets("fear"):

    for l in syn.lemmas():

        synonyms.append(l.name())

        if l.antonyms():

            antonyms.append(l.antonyms()[0].name())



print(set(synonyms))
print(set(antonyms))
w1 = wordnet.synset('bus.n.01')

w2 = wordnet.synset('car.n.01')

print(w1.wup_similarity(w2))
w1 = wordnet.synset('bike.n.01')

w2 = wordnet.synset('car.n.01')

print(w1.wup_similarity(w2))
w1 = wordnet.synset('car.n.01')

w2 = wordnet.synset('cat.n.01')

print(w1.wup_similarity(w2))
import random

from nltk.corpus import movie_reviews



documents = [(list(movie_reviews.words(fileid)), category)

             for category in movie_reviews.categories()

             for fileid in movie_reviews.fileids(category)]



random.shuffle(documents)



print(documents[1])
all_words = []

for w in movie_reviews.words():

    all_words.append(w.lower())



all_words = nltk.FreqDist(all_words)

print(all_words.most_common(15))
print(all_words["cat"])
all_words = []



for w in movie_reviews.words():

    all_words.append(w.lower())



all_words = nltk.FreqDist(all_words)



word_features = list(all_words.keys())[:3000]
word_features[:10]
def find_features(document):

    words = set(document)

    features = {}

    for w in word_features:

        features[w] = (w in words)



    return features
print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]
# set that we'll train our classifier with

training_set = featuresets[:1900]



# set that we'll test against.

testing_set = featuresets[1900:]

# Next, we can define, and train our classifier like:



classifier = nltk.NaiveBayesClassifier.train(training_set)

# First we just simply are invoking the Naive Bayes classifier, then we go ahead and use .train() to train it all in one line.



# Easy enough, now it is trained. Next, we can test it:

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)
import pickle

save_classifier = open("naivebayes.pickle","wb")

pickle.dump(classifier, save_classifier)

save_classifier.close()
classifier_f = open("naivebayes.pickle", "rb")

classifier = pickle.load(classifier_f)

classifier_f.close()
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
MNB_classifier = SklearnClassifier(MultinomialNB())

MNB_classifier.train(training_set)

print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set))



BNB_classifier = SklearnClassifier(BernoulliNB())

BNB_classifier.train(training_set)

print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testing_set))
from sklearn.linear_model import LogisticRegression,SGDClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)



MNB_classifier = SklearnClassifier(MultinomialNB())

MNB_classifier.train(training_set)

print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)



BernoulliNB_classifier = SklearnClassifier(BernoulliNB())

BernoulliNB_classifier.train(training_set)

print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)



LogisticRegression_classifier = SklearnClassifier(LogisticRegression())

LogisticRegression_classifier.train(training_set)

print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)



SGDClassifier_classifier = SklearnClassifier(SGDClassifier())

SGDClassifier_classifier.train(training_set)

print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)



SVC_classifier = SklearnClassifier(SVC())

SVC_classifier.train(training_set)

print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)



LinearSVC_classifier = SklearnClassifier(LinearSVC())

LinearSVC_classifier.train(training_set)

print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)



NuSVC_classifier = SklearnClassifier(NuSVC())

NuSVC_classifier.train(training_set)

print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
from nltk.classify import ClassifierI

from statistics import mode
import nltk

import random

from nltk.corpus import movie_reviews

from nltk.classify.scikitlearn import SklearnClassifier

import pickle



from sklearn.naive_bayes import MultinomialNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC



from nltk.classify import ClassifierI

from statistics import mode





class VoteClassifier(ClassifierI):

    def __init__(self, *classifiers):

        self._classifiers = classifiers



    def classify(self, features):

        votes = []

        for c in self._classifiers:

            v = c.classify(features)

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



documents = [(list(movie_reviews.words(fileid)), category)

             for category in movie_reviews.categories()

             for fileid in movie_reviews.fileids(category)]



random.shuffle(documents)



all_words = []



for w in movie_reviews.words():

    all_words.append(w.lower())



all_words = nltk.FreqDist(all_words)



word_features = list(all_words.keys())[:3000]



def find_features(document):

    words = set(document)

    features = {}

    for w in word_features:

        features[w] = (w in words)



    return features



#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))



featuresets = [(find_features(rev), category) for (rev, category) in documents]

        

training_set = featuresets[:1900]

testing_set =  featuresets[1900:]



#classifier = nltk.NaiveBayesClassifier.train(training_set)



classifier_f = open("naivebayes.pickle","rb")

classifier = pickle.load(classifier_f)

classifier_f.close()









print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(15)



MNB_classifier = SklearnClassifier(MultinomialNB())

MNB_classifier.train(training_set)

print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)



BernoulliNB_classifier = SklearnClassifier(BernoulliNB())

BernoulliNB_classifier.train(training_set)

print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)



LogisticRegression_classifier = SklearnClassifier(LogisticRegression())

LogisticRegression_classifier.train(training_set)

print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)



SGDClassifier_classifier = SklearnClassifier(SGDClassifier())

SGDClassifier_classifier.train(training_set)

print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)



##SVC_classifier = SklearnClassifier(SVC())

##SVC_classifier.train(training_set)

##print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)



LinearSVC_classifier = SklearnClassifier(LinearSVC())

LinearSVC_classifier.train(training_set)

print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)



NuSVC_classifier = SklearnClassifier(NuSVC())

NuSVC_classifier.train(training_set)

print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)





voted_classifier = VoteClassifier(classifier,

                                  NuSVC_classifier,

                                  LinearSVC_classifier,

                                  SGDClassifier_classifier,

                                  MNB_classifier,

                                  BernoulliNB_classifier,

                                  LogisticRegression_classifier)



print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)



print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)

print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)

print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)

print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)

print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)

print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)