%matplotlib inline

import gensim

import pandas

import nltk.corpus

import nltk.sentiment

import sklearn.linear_model

import textblob

import random

import numpy

import sklearn.metrics

import sklearn.ensemble

import seaborn

import re

import collections



sentence_splitter=re.compile(u"""[.?!]['"]*\s+""",re.UNICODE)

def sentence_structure_features(document):

    return ['_'.join((pos for (word,pos) in sentence.pos_tags))

            for sentence in textblob.blob.TextBlob(document).sentences]
class SentenceStructureCorpus(object):

    def __init__(self):

        lies=pandas.read_csv("../input/fake.csv")

        n_lies=lies.shape[0]

        self.vader=nltk.sentiment.vader.SentimentIntensityAnalyzer()

        print("Converting Fake News corpus")

        self.data=[sentence_structure_features('{0}\n{1}'.format(row['title'],row['text']))

                   for (index,row) in lies.iterrows()]

        sentiments=[self.analyse_sentiments('{0}\n{1}'.format(row['title'],row['text']))

                    for (index,row) in lies.iterrows()]

        reuters=nltk.corpus.reuters

        print('Converting Reuters corpus')

        self.data.extend([sentence_structure_features(reuters.raw(fileid))

                          for fileid in reuters.fileids()])

        sentiments.extend([self.analyse_sentiments(reuters.raw(fileid))

                           for fileid in reuters.fileids()])

        self.sentiments=numpy.array(sentiments)

        self.N=len(self.data)

        self.labels=numpy.ones(self.N)

        self.labels[:n_lies]=0

        self.test_sample=random.sample(range(self.N),self.N//10)

        print("Creating dictionary")

        self.dictionary=gensim.corpora.dictionary.Dictionary(self.data)

        

    def __iter__(self):

        return (self.dictionary.doc2bow(document) for document in self.data)

                          

    def analyse_sentiments(self,document):

        valences=numpy.array([[sent['pos'],sent['neg'],sent['neu']]

                             for sent in (self.vader.polarity_scores(sentence)

                                          for sentence in sentence_splitter.split(document))])

        return valences.sum(axis=0)

    

    def training_data(self):

        return [self.dictionary.doc2bow(document) for (i,document) in enumerate(self.data)

                if i not in self.test_sample]

                

    def training_labels(self):

        return self.labels[[i for i in range(self.N) if i not in self.test_sample]]

    

    def training_sentiments(self):

        return self.sentiments[[i for i in range(self.N) if i not in self.test_sample]]

    

    def test_sentiments(self):

        return self.sentiments[self.test_sample]

                

    def test_data(self):

        return [self.dictionary.doc2bow(self.data[i])

                for i in self.test_sample]

            

    def test_labels(self):

        return self.labels[self.test_sample]
ssf=SentenceStructureCorpus()

print("Training LSI")

lsi=gensim.models.lsimodel.LsiModel(ssf)
vectors=gensim.matutils.corpus2dense(lsi[ssf.training_data()],lsi.num_topics).T

classifier=sklearn.linear_model.LogisticRegression()

print("Training classifier")

classifier.fit(vectors,ssf.training_labels())

print("Testing classifier")

confusion=sklearn.metrics.confusion_matrix(ssf.test_labels(),

                                           classifier.predict(gensim.matutils.corpus2dense(lsi[ssf.test_data()],

                                                                                           lsi.num_topics).T))

seaborn.heatmap(confusion,annot=True)
def precision(cm):

    return cm[1,1]/cm[:,1].sum()



def recall(cm):

    return cm[1,1]/cm[1].sum()



def accuracy(cm):

    return (cm[0,0]+cm[1,1])/cm.sum()



def matthews(cm):

    return (cm[0,0]*cm[1,1]-cm[1,0]*cm[0,1])/numpy.sqrt(cm[0].sum()*cm[1].sum()*cm[:,0].sum()*cm[:,1].sum())
precision(confusion)
recall(confusion)
accuracy(confusion)
matthews(confusion)
sentiment_classifier=sklearn.linear_model.LogisticRegression()

sentiment_classifier.fit(ssf.training_sentiments(),ssf.training_labels())

confusion=sklearn.metrics.confusion_matrix(ssf.test_labels(),

                                           sentiment_classifier.predict(ssf.test_sentiments()))

seaborn.heatmap(confusion,annot=True)
precision(confusion)
recall(confusion)
accuracy(confusion)
matthews(confusion)
enhanced_vectors=numpy.hstack([vectors,ssf.training_sentiments()])

combined_classifier=sklearn.linear_model.LogisticRegression()

print("Training classifier")

combined_classifier.fit(enhanced_vectors,ssf.training_labels())

print("Testing classifier")

enhanced_test_vectors=numpy.hstack([gensim.matutils.corpus2dense(lsi[ssf.test_data()],

                                                                 lsi.num_topics).T,

                                    ssf.test_sentiments()])

confusion=sklearn.metrics.confusion_matrix(ssf.test_labels(),

                                           combined_classifier.predict(enhanced_test_vectors))

seaborn.heatmap(confusion,annot=True)
precision(confusion)
recall(confusion)
accuracy(confusion)
matthews(confusion)
forest0=sklearn.ensemble.RandomForestClassifier(n_estimators=100)

forest0.fit(vectors,ssf.training_labels())

confusion=sklearn.metrics.confusion_matrix(ssf.test_labels(),

                                           forest0.predict(gensim.matutils.corpus2dense(lsi[ssf.test_data()],

                                                                                           lsi.num_topics).T))

seaborn.heatmap(confusion,annot=True)
precision(confusion)
recall(confusion)
accuracy(confusion)
matthews(confusion)
forest1=sklearn.ensemble.RandomForestClassifier(n_estimators=100)

forest1.fit(ssf.training_sentiments(),ssf.training_labels())

confusion=sklearn.metrics.confusion_matrix(ssf.test_labels(),

                                           forest1.predict(ssf.test_sentiments()))

seaborn.heatmap(confusion,annot=True)
precision(confusion)
recall(confusion)
accuracy(confusion)
matthews(confusion)
forest2=sklearn.ensemble.RandomForestClassifier(n_estimators=100)

forest2.fit(enhanced_vectors,ssf.training_labels())

confusion=sklearn.metrics.confusion_matrix(ssf.test_labels(),

                                           forest2.predict(enhanced_test_vectors))

seaborn.heatmap(confusion,annot=True)
precision(confusion)
recall(confusion)
accuracy(confusion)
matthews(confusion)
keys = collections.defaultdict(int)

for doc in ssf:

    for (key,count) in doc:

        keys[key]+=1

pandas.Series(keys).value_counts()
repeated_keys = collections.defaultdict(int)

for doc in ssf:

    repeated_keys[len([key for (key,value) in doc if keys[key]>1])]+=1

pandas.Series(repeated_keys).sort_values()
stopwords = nltk.corpus.stopwords.words("english")

stopwords
def sentence_structure_features(document):

    blob = textblob.blob.TextBlob(document)

    return ['_'.join((pos for (word,pos) in sentence.pos_tags))

            for sentence in textblob.blob.TextBlob(document).sentences] +[word.lower() 

                                                                          for word in blob.words

                                                                          if  word.lower() in stopwords]

ssf2 = SentenceStructureCorpus()

transform = gensim.models.LsiModel(ssf2,id2word=ssf2.dictionary)

training_data = gensim.matutils.corpus2dense(transform[ssf2.training_data()],

                                             transform.num_topics).T

test_data = gensim.matutils.corpus2dense(transform[ssf2.test_data()],

                                         transform.num_topics).T

classifier = sklearn.linear_model.LogisticRegression()

classifier.fit(training_data,ssf2.training_labels())

confusion = sklearn.metrics.confusion_matrix(ssf2.test_labels(),

                                             classifier.predict(test_data))

seaborn.heatmap(confusion,annot=True)
precision(confusion)
recall(confusion)
accuracy(confusion)
matthews(confusion)