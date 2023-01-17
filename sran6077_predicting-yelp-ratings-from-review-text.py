import numpy as np

import pandas as pd

from scipy import sparse



from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.metrics import roc_auc_score



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D, GRU

from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping, ModelCheckpoint

%matplotlib inline
# This post was original written locally using the most up to date Yelp academic dataset. I have to recreate the sample

# I took here and since I did that in R, I can't guarantee the results will be the same. Just FYI

business = pd.read_csv("../input/yelp-dataset/yelp_business.csv")

review_all = pd.read_csv("../input/yelp-dataset/yelp_review.csv")
a = business[business['categories'].str.contains('Restaurant') == True]

rev = review_all[review_all.business_id.isin(a['business_id']) == True]
rev_samp = rev.sample(n = 350000, random_state = 42)

train = rev_samp[0:280000]

test = rev_samp[280000:]
train.shape, test.shape
#train = pd.read_csv('/home/adam/R/Yelp/dataset/model_train.csv', usecols = ['text', 'stars'])

train = train[['text', 'stars']]

train['stars'].hist();train.head()
train = pd.get_dummies(train, columns = ['stars'])

train.head()
#test = pd.read_csv('/home/adam/R/Yelp/dataset/model_test.csv', usecols=['text', 'stars'])

test = test[['text', 'stars']]

test = pd.get_dummies(test, columns = ['stars'])

train.shape, test.shape
# set frac = 1. to use the entire sample

train_samp = train.sample(frac = .1, random_state = 42)

test_samp = test.sample(frac = .1, random_state = 42)

train_samp.shape, test_samp.shape
# max_features is an upper bound on the number of words in the vocabulary

max_features = 2000

tfidf = TfidfVectorizer(max_features = max_features)
class NBFeatures(BaseEstimator):

    '''Class implementation of Jeremy Howards NB Linear model'''

    def __init__(self, alpha):

        # Smoothing Parameter: always going to be one for my use

        self.alpha = alpha

        

    def preprocess_x(self, x, r):

        return x.multiply(r)

    

    # calculate probabilities

    def pr(self, x, y_i, y):

        p = x[y == y_i].sum(0)

        return (p + self.alpha)/((y==y_i).sum()+self.alpha)

    

    # calculate the log ratio and represent as sparse matrix

    # ie fit the nb model

    def fit(self, x, y = None):

        self._r = sparse.csr_matrix(np.log(self.pr(x, 1, y) /self.pr(x, 0, y)))

        return self

    

    # apply the nb fit to original features x

    def transform(self, x):

        x_nb = self.preprocess_x(x, self._r)

        return x_nb
# Create pipeline using sklearn pipeline:

    # I basically create my tfidf features which are fed to my NB model 

    # for probability calculations. Then those are fed as input to my 

    # logistic regression model.

lr = LogisticRegression()

nb = NBFeatures(1)

p = Pipeline([

    ('tfidf', tfidf),

    ('nb', nb),

    ('lr', lr)

])
class_names = ['stars_1', 'stars_2', 'stars_3', 'stars_4', 'stars_5']

scores = []

preds = np.zeros((len(test_samp), len(class_names)))

for i, class_name in enumerate(class_names):

    train_target = train_samp[class_name]    

    cv_score = np.mean(cross_val_score(estimator = p, X = train_samp['text'].values, 

                                      y = train_target, cv = 3, scoring = 'accuracy'))

    scores.append(cv_score)

    print('CV score for class {} is {}'.format(class_name, cv_score))

    p.fit(train_samp['text'].values, train_target)

    preds[:,i] = p.predict_proba(test_samp['text'].values)[:,1]
train['text'][4971248]
t = metrics.classification_report(np.argmax(test_samp[class_names].values, axis = 1),np.argmax(preds, axis = 1))

print(t)
# I'm using GLoVe word vectors to get pretrained word embeddings

embed_size = 200 

# max number of unique words 

max_features = 20000

# max number of words from review to use

maxlen = 200



# File path

embedding_file = '../input/glove-global-vectors-for-word-representation/glove.twitter.27B.200d.txt'



# read in embeddings

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embedding_file))
class_names = ['stars_1', 'stars_2', 'stars_3', 'stars_4', 'stars_5']

# Splitting off my y variable

y = train_samp[class_names].values
tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train_samp['text'].values))

X_train = tokenizer.texts_to_sequences(train_samp['text'].values)

X_test = tokenizer.texts_to_sequences(test_samp['text'].values)

x_train = pad_sequences(X_train, maxlen = maxlen)

x_test = pad_sequences(X_test, maxlen = maxlen)
word_index = tokenizer.word_index



nb_words = min(max_features, len(word_index))

# create a zeros matrix of the correct dimensions 

embedding_matrix = np.zeros((nb_words, embed_size))

missed = []

for word, i in word_index.items():

    if i >= max_features: break

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector

    else:

        missed.append(word)
len(missed)
missed[0:10]
missed[1000:1010]
inp = Input(shape = (maxlen,))

x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = True)(inp)

x = SpatialDropout1D(0.5)(x)

x = Bidirectional(LSTM(40, return_sequences=True))(x)

x = Bidirectional(GRU(40, return_sequences=True))(x)

avg_pool = GlobalAveragePooling1D()(x)

max_pool = GlobalMaxPooling1D()(x)

conc = concatenate([avg_pool, max_pool])

outp = Dense(5, activation = 'sigmoid')(conc)



model = Model(inputs = inp, outputs = outp)

# patience is how many epochs to wait to see if val_loss will improve again.

earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3)

checkpoint = ModelCheckpoint(monitor = 'val_loss', save_best_only = True, filepath = 'yelp_lstm_gru_weights.hdf5')

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y, batch_size = 512, epochs = 20, validation_split = .1,

          callbacks=[earlystop, checkpoint])
y_test = model.predict([x_test], batch_size=1024, verbose = 1)
model.evaluate(x_test, test_samp[class_names].values, verbose = 1, batch_size=1024)
v = metrics.classification_report(np.argmax(test_samp[class_names].values, axis = 1),np.argmax(y_test, axis = 1))

print(v)
# Don't actually need to save it for the kernel

#model.save('yelp_nn_model.h5')
# This won't work here since I manually found the index and it doesn't exist in this sample

# this review was predicted as a 5 star but actually was a 1 star review

#test_samp['text'][367]