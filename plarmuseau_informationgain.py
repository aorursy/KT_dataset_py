import time

start = time.clock()



#open data

import pandas as pd

import numpy as np

import nltk

import codecs

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))





SampleSize=100000



datas = pd.read_csv('../input/train.csv')

datas = datas[300000:300000+SampleSize]





#datas=datas[datas['is_duplicate'] == 1]

#datas=datas.sample(SampleSize)

#datas=datas[0:SampleSize]

datas = datas.fillna('leeg')





def cleantxt(x):    # aangeven sentence

    x = x.lower()

    # Removing non ASCII chars

    x = x.replace(r'[^\x00-\x7f]',r' ')

    # Pad punctuation with spaces on both sides

    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:

        x = x.replace(char, ' ' + char + ' ')

    return x



datas['question1']=datas['question1'].map(cleantxt)

datas['question2']=datas['question2'].map(cleantxt)



end = time.clock()

print('open:',end-start)



def difwords(row):

    q1words = str(row[3]).lower().split()

    q2words = str(row[4]).lower().split()

    equq1 = [w for w in q1words if w in q2words]

    difq1 = [w for w in q1words if w not in q2words]

    difq2 = [w for w in q2words if w not in q1words]

    signw = [i for i in difq1+difq2 if i not in stop]

    return ' '.join(signw)



def lengte(row):

    return len(row[6].split())





# all questions with only one word different, leading to a 0 or a 1



datas['diff'] = datas.apply(difwords, axis=1, raw=True)

datas['tell']= datas.apply(lengte, axis=1, raw=True)

uni_diff=datas[datas['tell']>0]

uni_w1=uni_diff[uni_diff['is_duplicate']==1]

uni_w0=uni_diff[uni_diff['is_duplicate']==0]

print(len(uni_w1))

#print('non differecings words',' '.join(uni_w1['diff']))

print(len(uni_w0))

#print('differencing words',' '.join(uni_w0['diff']))





# all questions with two words different, leading to a 0 or a 1



uni_diff=datas[datas['tell']==2]

end = time.clock()

print('finding relevant and non relevant words:',end-start)

print(uni_w1.head())

print(uni_w0.head())
from nltk.tokenize import word_tokenize

from nltk.probability import FreqDist

fdist = FreqDist(' '.join(uni_w1['diff']))

print(fdist)

fdist.most_common(50)

from nltk.tokenize import word_tokenize

from nltk.probability import FreqDist,LaplaceProbDist

fdist_pos = LaplaceProbDist(word.lower() for word in word_tokenize(' '.join(uni_w1['diff'])))

fdist_neg = LaplaceProbDist(word.lower() for word in word_tokenize(' '.join(uni_w0['diff'])))

print(fdist_pos.most_common(200))

print(fdist_neg.most_common(200))
def unusual_words(text):

    text_vocab = set(w.lower() for w in text if w.isalpha())

    english_vocab = set(w.lower() for w in nltk.corpus.words.words())

    unusual = text_vocab - english_vocab

    return sorted(unusual)



text=' '.join(datas['question1'])

unusual_words(text.split())
print(datas.head())

import collections, itertools

from nltk.metrics import precision,recall

import nltk.classify.util

from nltk.classify import NaiveBayesClassifier

from nltk.corpus import stopwords

from nltk.collocations import BigramCollocationFinder

from nltk.metrics import BigramAssocMeasures, scores

from nltk.probability import FreqDist, ConditionalFreqDist



negx=[]

posx=[]

def evaluate_classifier(featx):

    negids = uni_w0

    posids = uni_w1



    for xi in range (len(uni_w0)):

        words= uni_w0.iloc[xi].question1.split() + uni_w0.iloc[xi].question2.split()

        negx.append(word_feats(words))

    negy=[]

    for yi in range (len(negx)):

        negy.append('neg')

    negfeats=list(zip(negx,negy))



    for xi in range (len(uni_w1)):

        words= uni_w1.iloc[xi].question1.split() + uni_w1.iloc[xi].question2.split()

        posx.append(word_feats(words))

    posy=[]

    for yi in range (len(posx)):

        posy.append('pos')

    posfeats=list(zip(posx,posy))



 

    negcutoff = int(len(negfeats)*3/4)

    poscutoff = int(len(posfeats)*3/4)

    print(negcutoff)

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]

    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

    end = time.clock()

    print('sampling:',end-start)

    print(negfeats[:10])

 

    classifier = NaiveBayesClassifier.train(trainfeats)

    refsets = collections.defaultdict(set)

    testsets = collections.defaultdict(set)

 

    for i, (feats, label) in enumerate(testfeats):

            refsets[label].add(i)

            observed = classifier.classify(feats)

            testsets[observed].add(i)

 

    print ('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))

    #print ('pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos']))

    #print ('pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos']))

    #print ('neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg']))

    #print ('neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg']))

    classifier.show_most_informative_features()

 

def word_feats(words):

    return dict([(word, True) for word in words])

 

print ('evaluating single word features')

evaluate_classifier(word_feats)

 

word_fd = []

#FreqDist(' '.join(datas['question1']) + ' '.join(datas['question2']) )





    

label_word_fd = ConditionalFreqDist()



for word in posx:

    word_fd[word]+=1

    label_word_fd['pos'][word]+=1



for word in negx:

    word_fd[word]+=1

    label_word_fd['neg'][word]+=1



end = time.clock()

print('freq distr:',end-start)



pos_word_count = label_word_fd['pos'].N()

neg_word_count = label_word_fd['neg'].N()

total_word_count = pos_word_count + neg_word_count

 

word_scores = {}

 

for word, freq in word_fd.iteritems():

    pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],

        (freq, pos_word_count), total_word_count)

    neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],

        (freq, neg_word_count), total_word_count)

    word_scores[word] = pos_score + neg_score

 

best = sorted([word_scores(v,k) for (k,v) in d.items()], reverse=True)

print(best)

bestwords = set([w for w, s in best])

 

def best_word_feats(words):

    return dict([(word, True) for word in words if word in bestwords])

 

print ('evaluating best word features')

evaluate_classifier(best_word_feats)

 

def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):

    bigram_finder = BigramCollocationFinder.from_words(words)

    bigrams = bigram_finder.nbest(score_fn, n)

    d = dict([(bigram, True) for bigram in bigrams])

    d.update(best_word_feats(words))

    return d

 

print ('evaluating best words + bigram chi_sq word features')

evaluate_classifier(best_bigram_word_feats)
import collections, itertools

import nltk.classify.util, nltk.metrics

from nltk.classify import NaiveBayesClassifier

from nltk.corpus import movie_reviews, stopwords

from nltk.collocations import BigramCollocationFinder

from nltk.metrics import BigramAssocMeasures

from nltk.probability import FreqDist, ConditionalFreqDist

 

def evaluate_classifier(featx):

    negids = movie_reviews.fileids('neg')

    posids = movie_reviews.fileids('pos')

 

    negfeats = [(featx(movie_reviews.words(fileids=[f])), 'neg') for f in negids]

    posfeats = [(featx(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

 

    negcutoff = len(negfeats)*3/4

    poscutoff = len(posfeats)*3/4

 

    trainfeats = negfeats[0:10] + posfeats[0:10]



    

    print(trainfeats)



def word_feats(words):

    return dict([(word, True) for word in words])

 

print ('evaluating single word features')

evaluate_classifier(word_feats)

 

word_fd = FreqDist()

label_word_fd = ConditionalFreqDist()

 

for word in movie_reviews.words(categories=['pos']):

    word_fd.inc(word.lower())

    label_word_fd['pos'].inc(word.lower())

 

for word in movie_reviews.words(categories=['neg']):

    word_fd.inc(word.lower())

    label_word_fd['neg'].inc(word.lower())

 

# n_ii = label_word_fd[label][word]

# n_ix = word_fd[word]

# n_xi = label_word_fd[label].N()

# n_xx = label_word_fd.N()

 

pos_word_count = label_word_fd['pos'].N()

neg_word_count = label_word_fd['neg'].N()

total_word_count = pos_word_count + neg_word_count



evaluate_classifier(best_word_feats)
from nltk.corpus import stopwords # Import the stop word list



from nltk import word_tokenize, ngrams

#print stopwords.words("english") simil_d=[]





simil_c=[]

simil_i=[]

simil_b=[]

simil_d=[]

for xyz in range(0,1000):

    q1=datas.iloc[xyz].question1

    q2=datas.iloc[xyz].question2

    uni_sent1 = nltk.wordpunct_tokenize(q1) #tokenize sentence

    uni_sent2 = nltk.wordpunct_tokenize(q2) 

    #sims=pd.DataFrame(data=None, index=uni_sent1, columns=uni_sent2)  #abs(np.random.randn(len(sent1), len(sent2))/1000000)

    equq1 = [w for w in uni_sent1 if w in uni_sent2]

    difq1 = [w for w in uni_sent1 if w not in uni_sent2]

    difq2 = [w for w in uni_sent2 if w not in uni_sent1]

    diftot = difq1+difq2

    difton = [w for w in diftot if not w in stopwords.words("english")]

    Q2no = [w for w in uni_sent2 if not w in stopwords.words("english")]



    canar=len(equq1)/(len(equq1)+len(diftot))

    simil_c.append(canar)

    

    



    if len(difton)==0 and datas.iloc[xyz].is_duplicate==0:

        simil_d.append(2)

    elif ('not' in uni_sent1) != ('not' in uni_sent2) :

        simil_d.append(3)

    #elif Q2no<3 :

    #    simil_d.append(4)

    else:

        simil_d.append(datas.iloc[xyz].is_duplicate)

        

end = time.clock()

print('first canary:',end-start)

  

 