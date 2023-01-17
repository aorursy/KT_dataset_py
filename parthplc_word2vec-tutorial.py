# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
train.head()
import string

import re

def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text





train['text'] = train['text'].apply(lambda x: clean_text(x))

test['text'] = test['text'].apply(lambda x: clean_text(x))



# Let's take a look at the updated text

train['text'].head()
from gensim.models import Word2Vec, KeyedVectors

import nltk
# Tokenizing the training and the test set

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

train['text'] = train['text'].apply(lambda x: tokenizer.tokenize(x))

test['text'] = test['text'].apply(lambda x: tokenizer.tokenize(x))

train['text'].head()
# Lets remove the stopwords as it does not seems of any meaning
from nltk.corpus import stopwords 

def remove_stopwords(text):

    """

    Removing stopwords belonging to english language

    

    """

    words = [w for w in text if w not in stopwords.words('english')]

    return words
train['text'] = train['text'].apply(lambda x : remove_stopwords(x))

test['text'] = test['text'].apply(lambda x : remove_stopwords(x))

train.head(10)
test['target'] = 0
def combine_text(list_of_text):

    '''Takes a list of text and combines them into one large chunk of text.'''

    combined_text = ' '.join(list_of_text)

    return combined_text



train['text'] = train['text'].apply(lambda x : combine_text(x))

test['text'] = test['text'].apply(lambda x : combine_text(x))

train.head()
# Now we will make a corpus of words to start word2vec training.
df = pd.concat([train,test])
corpus = df['text'].values
corpus
Corpus_list = [nltk.word_tokenize(title) for title in corpus]
Corpus_list
model = Word2Vec(Corpus_list,min_count=1,size = 100)
model.most_similar('death')
!pwd
path = "../input/googles-trained-word2vec-model-in-python/GoogleNews-vectors-negative300.bin"
import gensim

model = gensim.models.KeyedVectors.load_word2vec_format(path,binary=True)
w = model["hello"]

print(len(w))
print(w)
class MeanEmbeddingVectorizer(object):



    def __init__(self, word_model):

        self.word_model = word_model

        self.vector_size = word_model.wv.vector_size



    def fit(self):  # comply with scikit-learn transformer requirement

        return self



    def transform(self, docs):  # comply with scikit-learn transformer requirement

        doc_word_vector = self.word_average_list(docs)

        return doc_word_vector



    def word_average(self, sent):

        """

        Compute average word vector for a single doc/sentence.





        :param sent: list of sentence tokens

        :return:

            mean: float of averaging word vectors

        """

        mean = []

        for word in sent:

            if word in self.word_model.wv.vocab:

                mean.append(self.word_model.wv.get_vector(word))



        if not mean:  # empty words

            # If a text is empty, return a vector of zeros.

            #logging.warning("cannot compute average owing to no vector for {}".format(sent))

            return np.zeros(self.vector_size)

        else:

            mean = np.array(mean).mean(axis=0)

            return mean





    def word_average_list(self, docs):

        """

        Compute average word vector for multiple docs, where docs had been tokenized.



        :param docs: list of sentence in list of separated tokens

        :return:

            array of average word vector in shape (len(docs),)

        """

        return np.vstack([self.word_average(sent) for sent in docs])
class TfidfEmbeddingVectorizer(object):

    def __init__(self, word2vec):

        self.word2vec = word2vec

        self.word2weight = None

        self.dim = len(word2vec.itervalues().next())



    def fit(self, X, y):

        tfidf = TfidfVectorizer(analyzer=lambda x: x)

        tfidf.fit(X)

        # if a word was never seen - it must be at least as infrequent

        # as any of the known words - so the default idf is the max of 

        # known idf's

        max_idf = max(tfidf.idf_)

        self.word2weight = defaultdict(

            lambda: max_idf,

            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])



        return self



    def transform(self, X):

        return np.array([

                np.mean([self.word2vec[w] * self.word2weight[w]

                         for w in words if w in self.word2vec] or

                        [np.zeros(self.dim)], axis=0)

                for words in X

            ])

mean_vec_tr = MeanEmbeddingVectorizer(model)

doc_vec = mean_vec_tr.transform(Corpus_list)
print('Shape of word-mean doc2vec...')

display(doc_vec.shape)
Corpus_train = train['text'].values
train_corpus = [nltk.word_tokenize(title) for title in Corpus_train]

doc_vec_1 = mean_vec_tr.transform(train_corpus)
len(train_corpus)

print('Shape of word-mean doc2vec...')

display(doc_vec_1.shape)
Corpus_test = test['text'].values

test_corpus = [nltk.word_tokenize(title) for title in Corpus_test]

doc_vec_2 = mean_vec_tr.transform(test_corpus)

print('Shape of word-mean doc2vec...')

display(doc_vec_2.shape)
X = doc_vec_1

y = train['target']
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import model_selection

clf = LogisticRegression(C=1.0)

scores = model_selection.cross_val_score(clf,X,y, cv=5, scoring="f1")

scores
clf.fit(X,y)
X_vec_test = doc_vec_2
sample_submission_1 = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

sample_submission_1["target"] = clf.predict(X_vec_test)

sample_submission_1.to_csv("submission.csv", index=False)
# Using Advance Algorithms
from xgboost import XGBClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import StratifiedKFold
# xgb_model = XGBClassifier()



# #brute force scan for all parameters, here are the tricks

# #usually max_depth is 6,7,8

# #learning rate is around 0.05, but small changes may make big diff

# #tuning min_child_weight subsample colsample_bytree can have 

# #much fun of fighting against overfit 

# #n_estimators is how many round of boosting

# #finally, ensemble xgboost with multiple seeds may reduce variance

# parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

#               'objective':['binary:logistic'],

#               'learning_rate': [0.05,0.01,0.1], #so called `eta` value

#               'max_depth': [6,7,8,10],

#               'min_child_weight': [11],

#               'silent': [1],

#               'subsample': [0.8],

#               'colsample_bytree': [0.7,0.6,.5],

#               'n_estimators': [100,1000], #number of trees, change it to 1000 for better results

#               'missing':[-999],

#               'seed': [1337]}





# clf = GridSearchCV(xgb_model, parameters, n_jobs=5,  

#                    scoring='roc_auc',

#                    verbose=2, refit=True)



# clf.fit(X,y)



# #trust your CV!

# print("Best parameters set found on development set:")

# print()

# print(clf.best_params_)
clf = XGBClassifier(colsample_bytree=0.7, learning_rate= 0.05, max_depth= 8,

                    min_child_weight=11, missing= -999, n_estimators= 1000,

                    nthread= 4, objective='binary:logistic', seed=1337, silent=1, subsample=0.8)

scores = model_selection.cross_val_score(clf,X,y, cv=5, scoring="f1")

scores

clf.fit(X,y)
sample_submission_1 = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

sample_submission_1["target"] = clf.predict(X_vec_test)

sample_submission_1.to_csv("submission_3.csv", index=False)