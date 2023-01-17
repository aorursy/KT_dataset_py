# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import html

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
path = "/kaggle/input/aclimdb/aclImdb/"

positiveFiles = [x for x in os.listdir(path+"train/pos/")

                 if x.endswith(".txt")]

negativeFiles = [x for x in os.listdir(path+"train/neg/")

                if x.endswith(".txt")]

testFiles = [x for x in os.listdir(path+"test/") 

             if x.endswith(".txt")]
#positiveFiles
positiveReviews, negativeReviews, testReviews = [], [], []

for pfile in positiveFiles:

    with open(path+"train/pos/"+pfile, encoding="latin1") as f:

        positiveReviews.append(f.read())

for nfile in negativeFiles:

    with open(path+"train/neg/"+nfile, encoding="latin1") as f:

        negativeReviews.append(f.read())

for tfile in testFiles:

    with open(path+"test/"+tfile, encoding="latin1") as f:

        testReviews.append(f.read())
print(len(positiveReviews))

print(len(negativeReviews))

print(len(testReviews))
# testReviews
reviews = pd.concat([pd.DataFrame({"review":positiveReviews, "label":1,

                                   "file":positiveFiles}),

                    pd.DataFrame({"review":negativeReviews, "label":0,

                                   "file":negativeFiles}),

                    pd.DataFrame({"review":testReviews, "label":-1,

                                   "file":testFiles})

                    ], ignore_index=True).sample(frac=1, random_state=1)

                    
reviews.shape
reviews[0:10]
from nltk.corpus import stopwords

import re
stopWords = stopwords.words('english')
def CleanData(sentence):

    processedList = ""

    

    #convert to lowercase and ignore special charcter

    sentence = re.sub(r'[^A-Za-z0-9\s.]', r'', str(sentence).lower())

    sentence = re.sub(r'\n', r' ', sentence)

    

    sentence = " ".join([word for word in sentence.split() if word not in stopWords])

    

    return sentence
reviews.info()
reviews['review'][0]
CleanData(reviews['review'][0])
reviews['review'] = reviews['review'].map(lambda x: CleanData(x))
reviews['review'].head()
tmp_corpus = reviews['review'].map(lambda x:x.split('.'))
from tqdm import tqdm
#corpus [[w1, w2, w3,...],[...]]

corpus = []



for i in tqdm(range(len(reviews))):

    for line in tmp_corpus[i]:

        words = [x for x in line.split()]

        corpus.append(words)
len(corpus)
#removing blank list

corpus_new = []

for i in range(len(corpus)):

    if (len(corpus[i]) != 0):

        corpus_new.append(corpus[i])
# corpus[1:100]
num_of_sentences = len(corpus_new)

num_of_words = 0

for line in corpus_new:

    num_of_words += len(line)



print('Num of sentences - %s'%(num_of_sentences))

print('Num of words - %s'%(num_of_words))
from gensim.models import Word2Vec
# sg - skip gram |  window = size of the window | size = vector dimension

size = 100

window_size = 2 # sentences weren't too long, so

epochs = 100

min_count = 2

workers = 4



model = Word2Vec(corpus_new)
model.build_vocab(sentences= corpus_new, update=True)



for i in range(5):

    model.train(sentences=corpus_new, epochs=50, total_examples=model.corpus_count)

    
#save model

model.save('w2v_model')
model = Word2Vec.load('w2v_model')
model.wv.most_similar('movie')
reviews.head()
reviews = reviews[["review", "label", "file"]].sample(frac=1,

                                                      random_state=1)

train = reviews[reviews.label!=-1].sample(frac=0.6, random_state=1)

valid = reviews[reviews.label!=-1].drop(train.index)

test = reviews[reviews.label==-1]
print(train.shape)

print(valid.shape)

print(test.shape)
valid.head()
num_features = 100
index2word_set = set(model.wv.index2word)
model = model
def featureVecorMethod(words):

    featureVec = np.zeros(num_features, dtype='float32')

    nwords = 0

    

    for word in words:

        if word in index2word_set:

            nwords+= 1

            featureVec = np.add(featureVec, model[word])

            

    #average of feature vec

    featureVec = np.divide(featureVec, nwords)

    return featureVec
def getAvgFeatureVecs(reviews):

    counter = 0

    

    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype='float32')

    for review in reviews:

#         print(review)

        if counter%1000 == 0:

            print("Review %d of %d"%(counter, len(reviews)))

            

        reviewFeatureVecs[counter] = featureVecorMethod(review)

        counter = counter+1

    return reviewFeatureVecs
clean_train_reviews = []

for review in train['review']:

#     print(review)

    clean_train_reviews.append(list(CleanData(review).split()))

# print(len(clean_train_reviews))\



trainDataVecs = getAvgFeatureVecs(clean_train_reviews)

len(valid['review'])
clean_test_reviews = []

for review in valid['review']:

#     print(review)

    clean_test_reviews.append(list(CleanData(review).split()))

# print(len(clean_train_reviews))\



testDataVecs = getAvgFeatureVecs(clean_test_reviews)
print(len(print(len(testDataVecs))))

print(len(testDataVecs))
import sklearn
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100)



print("fitting data")

forest = forest.fit(trainDataVecs, train['label'])
# valid.index
result = forest.predict(testDataVecs)
output = pd.DataFrame(data={"id":valid.index, "sentiment": result})

from sklearn.metrics import accuracy_score
accuracy_score(valid['label'], result)