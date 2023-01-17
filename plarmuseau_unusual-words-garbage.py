import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from textblob import TextBlob

from textblob.np_extractors import ConllExtractor

extractor = ConllExtractor()



#NLTK functioncs

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.corpus import wordnet



# timing function

import time   

start = time.clock() #_________________ measure efficiency timing





# read data

#train = pd.read_csv('c:/py/testQ.csv',encoding='utf8')#[:1000]  #_______________________ open data files

train = pd.read_csv('../input/train.csv',encoding='utf8')

print(train.head(30))

train.fillna(value='leeg',inplace=True)

end = time.clock()

print('open:',end-start)





for xi in range (0,30):

    q1= TextBlob(train.iloc[xi].question1)

    q2= TextBlob(train.iloc[xi].question2)

    blob1 = TextBlob(train.iloc[xi].question1, np_extractor=extractor)

    blob2 = TextBlob(train.iloc[xi].question2, np_extractor=extractor) 

    print(blob1.noun_phrases)

    print(blob2.noun_phrases)    

    print(q1.tags)

    print(q2.tags)

    print(q1.correct())

    print(q2.correct())

    print(q1.noun_phrases)

    print(q2.noun_phrases)    

    print(q1.sentiment)

    print(q2.sentiment)    

    print(q1.sentiment.polarity)

    print(q2.sentiment.polarity)    
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



def unusual_words(text):

    text_vocab = set(w.lower() for w in text if w.isalpha())

    english_vocab = set(w.lower() for w in nltk.corpus.words.words())

    unusual = text_vocab - english_vocab

    return sorted(unusual)



text=' '.join(datas['question1'])

#unusual_words(text.split())
import re

from collections import Counter



def words(text): return re.findall(r'\w+', text.lower())



WORDS = Counter(words(open('big.txt').read()))



def P(word, N=sum(WORDS.values())): 

    "Probability of `word`."

    return WORDS[word] / N



def correction(word): 

    "Most probable spelling correction for word."

    return max(candidates(word), key=P)



def candidates(word): 

    "Generate possible spelling corrections for word."

    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])



def known(words): 

    "The subset of `words` that appear in the dictionary of WORDS."

    return set(w for w in words if w in WORDS)



def edits1(word):

    "All edits that are one edit away from `word`."

    letters    = 'abcdefghijklmnopqrstuvwxyz'

    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]

    deletes    = [L + R[1:]               for L, R in splits if R]

    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]

    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]

    inserts    = [L + c + R               for L, R in splits for c in letters]

    return set(deletes + transposes + replaces + inserts)



def edits2(word): 

    "All edits that are two edits away from `word`."

    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
# all questions with only one word different, leading to a 0 or a 1

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





datas['diff'] = datas.apply(difwords, axis=1, raw=True)

datas['tell']= datas.apply(lengte, axis=1, raw=True)

uni_diff=datas[datas['tell']>0]

uni_w1=uni_diff[uni_diff['is_duplicate']==1]

uni_w0=uni_diff[uni_diff['is_duplicate']==0]

print(len(uni_w1))

#print('non differecings words',' '.join(uni_w1['diff']))

print(len(uni_w0))

#print('differencing words',' '.join(uni_w0['diff']))





def document_features(document):

    document_words = set(document)

    features = {}

    for word in word_features:

        features['contains({})'.format(word)] = (word in document_words)

    return features





#poswords=' '.join(uni_w1['diff'])

#negwords=' '.join(uni_w0['diff'])

print(uni_w1.head())

documents = [(list(uni_w1.ix[qnr].question1.split()), 'pos') for qnr in uni_w1['id']]

documents.append([(list(uni_w1.ix[qnr].question2.split()), 'pos') for qnr in uni_w1['id']])

documents.append([(list(uni_w0.ix[qnr].question1.split()), 'neg') for qnr in uni_w0['id']])

documents.append([(list(uni_w0.ix[qnr].question2.split()), 'neg') for qnr in uni_w0['id']])

featuresets = [(document_features(d), c) for (d,c) in documents]

train_set = featuresets[100:]

classifier = nltk.NaiveBayesClassifier.train(train_set)

classifier.show_most_informative_features(100)



# all questions with two words different, leading to a 0 or a 1



uni_diff=datas[datas['tell']==2]

end = time.clock()

print('finding relevant and non relevant words:',end-start)

print(uni_w1.head())

print(uni_w0.head())