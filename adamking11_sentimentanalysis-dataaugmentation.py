!python3 -m spacy download en_core_web_md
import os, sys, csv, re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from tqdm.auto import tqdm

import en_core_web_md
nlp = en_core_web_md.load()

df = pd.read_csv("../input/sentiment-analysis-on-movie-reviews/train.tsv.zip", sep="\t")
df_test = pd.read_csv('../input/sentiment-analysis-on-movie-reviews/test.tsv.zip', delimiter = '\t')

if True:
    df = df.rename(columns = {'Phrase' : 'Text', 'Sentiment' : 'Score'})
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for i, (train_indices, test_indices) in enumerate(sss.split(df, df.Score)):
        df_test = df.iloc[test_indices].reset_index(drop = True)
        df = df.iloc[train_indices].reset_index(drop = True)

    #df_test = df_test.rename(columns = {'Phrase' : 'Text', 'Sentiment' : 'Score'})
else:
    df = df.rename(columns = {'Phrase' : 'Text'})
    df['Score'] = 0
    df.loc[df.Sentiment >= 4, 'Score'] = 1
    df.loc[df.Sentiment <= 0, 'Score'] = -1
    
    #df_test = df_test.rename(columns = {'Phrase' : 'Text'})
    #df_test['Score'] = 0
    #df_test.loc[df_test.Sentiment >= 4, 'Score'] = 1
    #df_test.loc[df_test.Sentiment <= 0, 'Score'] = -1
    for i, (train_indices, test_indices) in enumerate(sss.split(df, df.Score)):
        df_test = df.iloc[test_indices].reset_index(drop = True)
        df = df.iloc[train_indices].reset_index(drop = True)

print(len(df), len(df_test))
df.head()
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn

tokenizer = RegexpTokenizer(r"['\-\w\.\?!,]+")
lemmatizer = WordNetLemmatizer()
stops = stopwords.words('english')

def tkn(text):
    if type(text) == list:
        text = ' '.join(text)
    text = re.sub(r" (n't|'[a-z]{1,3})", r'\1', text)
    text = re.sub(r'[^a-z0-9\s\'\-\.\?!,\"]', '', text.lower())
    text = ' '.join([lemmatizer.lemmatize(w) for w in tokenizer.tokenize(text)])
    return text

tqdm.pandas()
print('train')
df['Tokens'] = df.Text.apply(tkn)
df['nb_words'] = df.Tokens.str.count(' ') + 1
df['rel_nb_words'] = df.nb_words / df.nb_words.max()
df['nb_chars'] = df.Tokens.str.len()
df['rel_nb_chars'] = df.nb_chars / df.nb_chars.max()


print('test')
df_test['Tokens'] = df_test.Text.apply(tkn)
df_test['nb_words'] = df_test.Tokens.str.count(' ') + 1
df_test['rel_nb_words'] = df_test.nb_words / df_test.nb_words.max()
df_test['nb_chars'] = df_test.Tokens.str.len()
df_test['rel_nb_chars'] = df_test.nb_chars / df_test.nb_chars.max()

df.head()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

oneword_sentiment = {}
df_len1 = df[df.nb_words == 1]
for label in df_len1.Score.value_counts().to_dict().keys():
    for word in df_len1.Tokens[df_len1.Score == label]:
        oneword_sentiment[word] = label


def vader_sentinet(word):
    vader_scores = sid.polarity_scores(word)
    v_neg = vader_scores['neg']
    v_neu = vader_scores['neu']
    v_pos = vader_scores['pos']
    
    senti_net = list(swn.senti_synsets(word))
    if len(senti_net) > 0:
        s_neg = senti_net[0].neg_score()
        s_pos = senti_net[0].pos_score()
        s_obj = senti_net[0].obj_score()
    else:
        s_neg, s_pos, s_obj = 0, 0, 0
    return v_neg, v_neu, v_pos, s_neg, s_pos, s_obj

def mean_senti_vader_score(text):
    if type(text) == str:
        text = text.split()
    score_lists = [[] for _ in range(6)]
    for w in text:
        scores = vader_sentinet(w)
        for i, s in enumerate(scores):
            score_lists[i].append(s)
    mean_scores = [sum(l) / max(1, len(l)) for l in score_lists]
    return mean_scores
        
def nb_senti_words(text):
    if type(text) == str:
        text = text.split()
    d = {-1 : 0, 0 : 0, 1 : 0, 2 : 0, 3 : 0, 4 : 0}
    for w in text:
        if w in oneword_sentiment:
            d[oneword_sentiment[w]] += 1
    for k in d:
        d[k] /= max(1, len(text))
        
    return d[0], d[1], d[2], d[3], d[4], d[-1]

print('train')
df = df.merge(df.Tokens.apply(lambda t: pd.Series(nb_senti_words(t))), 
                left_index=True, right_index=True)
df = df.rename(columns={
                        0 : "nb_senti_0", 
                        1 : "nb_senti_1", 
                        2 : "nb_senti_2", 
                        3 : "nb_senti_3",
                        4 : "nb_senti_4",
                        5 : "nb_senti_-1"
                       })

print('\tvader/sentinet')
df = df.merge(df.Tokens.apply(lambda t: pd.Series(mean_senti_vader_score(t))), 
                left_index=True, right_index=True)
df = df.rename(columns={
                        0 : "v_neg", 
                        1 : "v_neu", 
                        2 : "v_pos", 
                        3 : "s_neg",
                        4 : "s_pos",
                        5 : "s_obj"
                       })

print('test')
df_test = df_test.merge(df_test.Tokens.apply(lambda t: pd.Series(nb_senti_words(t))), 
                left_index=True, right_index=True)

df_test = df_test.rename(columns={
                        0 : "nb_senti_0", 
                        1 : "nb_senti_1", 
                        2 : "nb_senti_2", 
                        3 : "nb_senti_3",
                        4 : "nb_senti_4",
                        5 : "nb_senti_-1",
                       })
df_test = df_test.merge(df_test.Tokens.apply(lambda t: pd.Series(mean_senti_vader_score(t))), 
                left_index=True, right_index=True)

print('\tvader/sentinet')
df_test = df_test.rename(columns={
                        0 : "v_neg", 
                        1 : "v_neu", 
                        2 : "v_pos", 
                        3 : "s_neg",
                        4 : "s_pos",
                        5 : "s_obj"
                       })

df.head()
from nltk.corpus import subjectivity, movie_reviews
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC

print('subjectivity')
subj_docs = []
for l in tqdm(['subj', 'obj']):
    for s in subjectivity.sents(categories=l):
        subj_docs.append((l, tkn(s)))

print('polarity')
review_docs = []
for l in tqdm(['pos', 'neg']):
    for s in movie_reviews.sents(categories=l):
        review_docs.append((l, tkn(s)))

subj_df = pd.DataFrame(subj_docs, columns = ['label', 'text'])
review_df = pd.DataFrame(review_docs, columns = ['label', 'text'])

subj_cv = CountVectorizer(binary = True, min_df = 5, max_df = .5, dtype = np.int8).fit(subj_df.text)
review_cv = CountVectorizer(binary = True, min_df = 5, max_df = .5, dtype = np.int8).fit(review_df.text)

X_subj = subj_cv.transform(subj_df.text)
X_review = review_cv.transform(review_df.text)

print('subj_logit')
subj_logit = LogisticRegressionCV(n_jobs = 3, max_iter = 400, 
                                  cv=6, random_state=0, verbose = True).fit(X_subj, subj_df.label)
print('review_logit')
review_logit = LogisticRegressionCV(n_jobs = 3, max_iter = 400, 
                                  cv=6, random_state=0, verbose = True).fit(X_review, review_df.label)

print('subj', subj_logit.score(X_subj, subj_df.label))
print('polarity', review_logit.score(X_review, review_df.label))

df['subj_0'], df['subj_1'] = zip(*subj_logit.predict_proba(subj_cv.transform(df.Tokens)))
df['review_0'], df['review_1'] = zip(*review_logit.predict_proba(review_cv.transform(df.Tokens)))

df_test['subj_0'], df_test['subj_1'] = zip(*subj_logit.predict_proba(subj_cv.transform(df_test.Tokens)))
df_test['review_0'], df_test['review_1'] = zip(*review_logit.predict_proba(review_cv.transform(df_test.Tokens)))

del subj_df, subj_logit, subj_cv
del review_df, review_logit, review_cv

df.head()

def vader_score(text):
    scores = sid.polarity_scores(text)
    return scores['neg'], scores['neu'], scores['pos'], scores['compound']

print('train')
df = df.merge(df.Tokens.apply(lambda t: pd.Series(vader_score(t))), 
         left_index=True, right_index=True)
df = df.rename(columns={0 : "vader_neg", 1 : "vader_neu", 2 : "vader_pos", 3 : "vader_compound"})

print('test')
df_test = df_test.merge(df_test.Tokens.apply(lambda t: pd.Series(vader_score(t))), 
         left_index=True, right_index=True)
df_test = df_test.rename(columns={0 : "vader_neg", 1 : "vader_neu", 2 : "vader_pos", 3 : "vader_compound"})

df.head()
"""
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

from scipy.sparse import hstack, csr_matrix

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

texts = []
for i, t in tqdm(enumerate(df.Tokens)):
    texts.append(tkn(t).split())
    
dictionary = Dictionary(texts)
dictionary.filter_extremes(no_below=5, no_above=0.5)
stop_ids = [i for i, w in dictionary.items() if w in stops]
dictionary.filter_tokens(bad_ids = stop_ids)

nb_lda_features = 64
text_corpus = [dictionary.doc2bow(t) for t in texts]

print('train lda')
lda = LdaModel(text_corpus, id2word = dictionary,  
               num_topics = nb_lda_features, 
               alpha = 'auto', eta = 'auto',
               iterations = 400, eval_every = None)
              

X_lda = np.zeros((len(df), nb_lda_features), dtype = np.float32)
for i, t in tqdm(enumerate(text_corpus)):
    for j, topic_score in lda[t]:
        X_lda[i,j] = topic_score

print('test')
text_corpus_test = []
for i, t in tqdm(enumerate(df_test.Tokens)):
    text_corpus_test.append(dictionary.doc2bow(tkn(t).split()))

X_lda_test = np.zeros((len(df_test), nb_lda_features), dtype = np.float32)
for i, t in tqdm(enumerate(text_corpus_test)):
    for j, topic_score in lda[t]:
        X_lda_test[i,j] = topic_score

del texts
del text_corpus
del text_corpus_test
del lda
"""
from scipy.sparse import vstack, hstack, csr_matrix
from scipy.sparse.linalg import svds

print('data')
cv = CountVectorizer(binary = True, max_df = .5, min_df = 5, 
                        dtype = np.int8).fit(df.Tokens)
#cv = TfidfVectorizer(max_df = .5, min_df = 5, dtype = np.float32).fit(df.Tokens)

X_text = cv.transform(df.Tokens) #.toarray()
X_text_final = cv.transform(df_test.Tokens)
print('starting shape:', X_text.shape)
#X_text_svd, _, _ = svds(X_text, k=1000, return_singular_vectors = 'u')
#print('SVD shape:', X_text_svd.shape)


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder


tknzr = Tokenizer()
tknzr.fit_on_texts(df.Tokens)


X_seq_list = tknzr.texts_to_sequences(df.Tokens)

X_seq_list_final = tknzr.texts_to_sequences(df_test.Tokens)

seq_len = max(max([len(s) for s in X_seq_list]),
             max([len(s) for s in X_seq_list_final]))

X_seq = pad_sequences(X_seq_list, maxlen = seq_len)
X_seq_final = pad_sequences(X_seq_list_final, maxlen = seq_len)


X_other = np.array(df[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 
                       'v_neg', 'v_neu', 'v_pos', 's_neg', 's_pos', 's_obj',
                       'nb_words', 'nb_chars', 'rel_nb_words', 'rel_nb_chars',
                       'nb_senti_-1', 'nb_senti_0', 'nb_senti_1', 'nb_senti_2', 'nb_senti_3', 'nb_senti_4', 
                       'subj_0', 'subj_1', 'review_0']])
X_other_final = np.array(df_test[['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound', 
                       'v_neg', 'v_neu', 'v_pos', 's_neg', 's_pos', 's_obj',
                       'nb_words', 'nb_chars', 'rel_nb_words', 'rel_nb_chars',
                       'nb_senti_-1', 'nb_senti_0', 'nb_senti_1', 'nb_senti_2', 'nb_senti_3', 'nb_senti_4', 
                       'subj_0', 'subj_1', 'review_0']])

#X = hstack((X_text, csr_matrix(X_lda), csr_matrix(X_other))).tocsr()
#X_final = hstack((X_text_test, csr_matrix(X_lda_test), csr_matrix(X_other_test))).tocsr()

X = hstack((X_text, csr_matrix(X_other))).tocsr()
X_final = hstack((X_text_final, csr_matrix(X_other_final))).tocsr()

oh = OneHotEncoder().fit(df.Score.to_numpy().reshape(-1, 1))
y = oh.transform(df.Score.to_numpy().reshape(-1,1)).toarray()


print(X.shape, X_seq.shape)
print(y.shape)
del X_seq_list
del X_seq_list_final
from collections import Counter
def undersample_Xy(X, y):
    y_argmax = y.argmax(axis = 1)
    y_counts = Counter(y_argmax)
    nb_to_sample = sorted(y_counts.values())[1]
    
    def undr(label):
        X_by_label = X[y_argmax == label]
        y_by_label = y[y_argmax == label]
        count_for_label = X_by_label.shape[0]
        sampled_indices = np.random.choice(np.arange(count_for_label), 
                                   size = min(nb_to_sample, count_for_label), replace = False)
        undersampled_X = X_by_label[sampled_indices]
        undersampled_y = y_by_label[sampled_indices]
        return undersampled_X, undersampled_y
    
    Xs, ys = zip(*[undr(label) for label in y_counts.keys()])
    return vstack(Xs), vstack(ys)

def rnn_model():
    rnn_input_1 = Input(shape = (X_seq.shape[1],))
    embed_1 = Embedding(output_dim = wvec.shape[1],
                        weights = [wvec],
                        mask_zero = True,
                        input_dim = len(tkn.word_index) + 1, 
                        input_length = X_seq.shape[1])(rnn_input_1)
    lstm_1 = LSTM(128, dropout = .1, recurrent_dropout = .1)(embed_1)
    
    output_1 = Dense(y.shape[1], activation = 'softmax', )(lstm_1) #bias_regularizer = 'l2'
    
    model = Model(rnn_input_1, output_1)
    model.compile(optimizer = Adam(learning_rate=lr), 
                  metrics = ['accuracy'], loss = 'categorical_crossentropy')
    return model
logit = None

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM 
from keras.layers import BatchNormalization, Dropout, Concatenate

from keras.callbacks.callbacks import EarlyStopping
from keras.optimizers import Adagrad, Adam
        
wordvec_dim = 311

wvec = np.zeros((len(tknzr.word_index) + 1, wordvec_dim), dtype = np.float32)
for w, i in tqdm(tknzr.word_index.items()):
    if w in nlp.vocab:
        wvec[i,:300] = nlp.vocab[w].vector   
    else:
        wvec[i,:300] = (2 * np.random.rand(300)) - 1
     
    wvec[i,300:306] = vader_sentinet(w)
    if w in oneword_sentiment:
        wvec[i, 306+oneword_sentiment[w]] = 1
        
lr = .00005

dense_1_size = 250
dense_2_size = 80
cnn_1_size = 120



opm = Adam(learning_rate=lr)
def logit_model():
    input_1 = Input(shape = (X.shape[1],))
    output_1 = Dense(y.shape[1], activation = 'softmax', )(input_1)
    model = Model(input_1, output_1)
    model.compile(optimizer = opm, 
                  metrics = ['accuracy'], loss = 'categorical_crossentropy')
    return model

def nn_model():
    input_1 = Input(shape = (X.shape[1],))
    dense_1 = Dense(dense_1_size, activation = 'relu',)(input_1)
    dense_1 = Dropout(.2)(dense_1)
    #dropped_1 = Dropout(.1)(dense_1)
    #normed_1 = BatchNormalization()(dense_1)
    #dense_2 = Dense(32, activation = 'relu')(dropped_1)
    #dropped_2 = Dropout(.1)(dense_2)
    output_1 = Dense(y.shape[1], activation = 'softmax', 
                     kernel_regularizer = None, bias_regularizer = None)(dense_1) #
    
    model = Model(input_1, output_1)
    model.compile(optimizer = opm, 
                  metrics = ['accuracy'], loss = 'categorical_crossentropy')
    return model


def cnn_model():
    cnn_input_1 = Input(shape = (X_seq.shape[1],))
    embed_1 = Embedding(output_dim = wvec.shape[1],
                        weights = [wvec],
                        #mask_zero = True,
                        input_dim = len(tknzr.word_index) + 1, 
                        input_length = X_seq.shape[1])(cnn_input_1)
    cnn_1 = Conv1D(cnn_1_size, 2, activation = 'relu', 
                   kernel_regularizer='l2')(embed_1)
    #pool_1 = MaxPooling1D(pool_size = 2)(cnn_1)
    #cnn_2 = Conv1D(512, 2, activation = 'relu', strides = 1)(pool_1)
    pool_2 = GlobalMaxPooling1D()(cnn_1)
    
    dense_1 = Dense(dense_2_size, activation = 'relu')(pool_2)
    dense_1 = Dropout(.2)(dense_1)
    output_1 = Dense(y.shape[1], activation = 'softmax', 
                     kernel_regularizer = 'l2', bias_regularizer = None)(dense_1)
    
    model = Model(cnn_input_1, output_1)

    model.compile(optimizer = Adam(learning_rate=lr), 
                  metrics = ['accuracy'], loss = 'categorical_crossentropy')
    return model

def cnn_dual_model():
    input_1 = Input(shape = (X.shape[1],))
    dense_1 = Dense(dense_1_size, activation = 'relu')(input_1)
    dense_1 = Dropout(.25)(dense_1)
    #normed_1 = BatchNormalization()(dense_1)
    
    cnn_input_1 = Input(shape = (X_seq.shape[1],))
    embed_1 = Embedding(output_dim = wvec.shape[1],
                        weights = [wvec],
                        #mask_zero = True,
                        input_dim = len(tknzr.word_index) + 1, 
                        input_length = X_seq.shape[1])(cnn_input_1)
    cnn_1 = Conv1D(cnn_1_size, 3, activation = 'relu', 
                   kernel_regularizer='l2')(embed_1)
    pool_1 = GlobalMaxPooling1D()(cnn_1)
    
    concat_1 = Concatenate()([dense_1, pool_1])
    dense_2 = Dense(dense_2_size, activation = 'relu')(concat_1)
    dense_2 = Dropout(.25)(dense_2)
    output_1 = Dense(y.shape[1], activation = 'softmax', 
                     kernel_regularizer = 'l2', bias_regularizer = 'l2')(dense_2) 
    
    model = Model([input_1, cnn_input_1], output_1)

    model.compile(optimizer = opm, 
                  metrics = ['accuracy'], loss = 'categorical_crossentropy')
    return model


from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_curve
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids


do_logit = False
do_svm = False

#sm = SMOTE(k_neighbors = 5)

max_epochs = 200
batch_size = 512
patience = 5
class_weight = {0 : 1, 1 : 1, 
                       2 : 1, 
                4 : 1, 3 : 1,}

es = EarlyStopping(monitor = 'val_accuracy', patience = patience, min_delta = -.001,
                   restore_best_weights = True)
print('fit')


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
cc = ClusterCentroids(random_state=0)
for i, (train_indices, test_indices) in enumerate(sss.split(X, y)):
    print(i)
    print()
    
    X_train = X[train_indices] #.to_numpy().reshape(-1, 1)
    X_other_train = X_other[train_indices]
    X_seq_train = X_seq[train_indices]
    y_train = y[train_indices]
    
    X_test = X[test_indices] #.to_numpy().reshape(-1, 1)
    X_other_test = X_other[test_indices]
    X_seq_test = X_seq[test_indices]
    y_test = y[test_indices]
    
    """
    print(X_train.shape, y_train.shape)
    X_train, y_train = undersample_Xy(X_train, y_train)
    print(X_train.shape, y_train.shape)
    """
    """
    if do_logit:
        print()
        print('\tlogit')
        
        model = LogisticRegression(n_jobs = 3, verbose = True, max_iter = 400).fit(X_train, y_train.argmax(axis = 1))
        y_hat, y_preds = model.predict(X_test), model.predict_proba(X_test)
    
        print('\t\t', f1_score(y_test.argmax(axis = 1), y_hat, average = 'weighted'), 
                    accuracy_score(y_test.argmax(axis = 1), y_hat))
        print(classification_report(y_test.argmax(axis = 1), y_hat))

        
    if do_svm:
        print()
        print('\tsvm')
        model = SVC(verbose = True).fit(X_train, y_train.argmax(axis = 1))
        y_hat = model.predict(X_test)
        print('\t\t', f1_score(y_test.argmax(axis = 1), y_hat, average = 'weighted'), 
              accuracy_score(y_test.argmax(axis = 1), y_hat))
    """
    if do_logit or logit is None:
        print()
        print()
        print('\tlogit')
        logit = logit_model()
        logit.fit(X_train, y_train, epochs = max_epochs, 
              batch_size = batch_size, 
              class_weight = class_weight,
              validation_split=0.1, callbacks = [es], 
              verbose = 1)

    """
    print()
    print()
    print('\tNN')
    n_model = nn_model()
    n_model.fit(X_train, y_train, epochs = max_epochs, 
              batch_size = batch_size, 
              class_weight = class_weight,
              validation_split=0.1, callbacks = [es], 
              verbose = 1)    
    
    print('\n\n')
    print('\tCNN')
    c_model = cnn_model()
    c_model.fit(X_seq_train, y_train, epochs = max_epochs, 
              batch_size = batch_size, 
              class_weight = class_weight,
              validation_split=0.1, callbacks = [es], 
              verbose = 1)    
    """
    print('\n\n')
    print('\tdual NN/CNN')
    d_model = cnn_dual_model()
    d_model.fit([X_train, X_seq_train], y_train, epochs = max_epochs, 
              batch_size = batch_size, 
              class_weight = class_weight,
              validation_split=0.1, callbacks = [es], 
              verbose = 1)
    
    continue
    print('\n\n')
    print('\tRNN') 
    r_model = rnn_model()
    r_model.fit(X_seq_train, y_train, epochs = max_epochs, 
              batch_size = batch_size, 
              class_weight = class_weight,
              validation_split=0.1, callbacks = [es], 
              verbose = 1)
    y_hat = model.predict(X_seq_test)
    
    print('\t\t', f1_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1), average = 'weighted'),
                  accuracy_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1)))
    print(classification_report(y_test.argmax(axis = 1), y_hat.argmax(axis = 1)))

    
print()
print()

print('logit')
y_hat = logit.predict(X_test)
print('\t\t', f1_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1), average = 'weighted'), 
                  accuracy_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1)))

print('nn')
y_hat = n_model.predict(X_test)
print('\t\t', f1_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1), average = 'weighted'), 
                  accuracy_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1)))

print('cnn')
y_hat = c_model.predict(X_seq_test)
print('\t\t', f1_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1), average = 'weighted'),
                  accuracy_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1)))

print('dual')
y_hat = d_model.predict([X_test, X_seq_test])
print('\t\t', f1_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1), average = 'weighted'),
                  accuracy_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1)))

print('logit')
y_hat = logit.predict(X_test)
print('\t\t', f1_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1), average = 'weighted'), 
                  accuracy_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1)))
print(classification_report(y_test.argmax(axis = 1), y_hat.argmax(axis = 1)))
    

print('nn')
y_hat = n_model.predict(X_test)
print('\t\t', f1_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1), average = 'weighted'), 
                  accuracy_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1)))
print(classification_report(y_test.argmax(axis = 1), y_hat.argmax(axis = 1)))


print('cnn')
y_hat = c_model.predict(X_seq_test)
print('\t\t', f1_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1), average = 'weighted'),
                  accuracy_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1)))
print(classification_report(y_test.argmax(axis = 1), y_hat.argmax(axis = 1)))

print('dual')
y_hat = d_model.predict([X_test, X_seq_test])
print('\t\t', f1_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1), average = 'weighted'),
                  accuracy_score(y_test.argmax(axis = 1), y_hat.argmax(axis = 1)))
print(classification_report(y_test.argmax(axis = 1), y_hat.argmax(axis = 1)))
df_test['Sentiment'] = d_model.predict([X_final, X_seq_final]).argmax(axis = 1)
df_final = df_test[['PhraseId', 'Sentiment']]
df_test.head()
print(classification_report(df_test.Score, df_test.Sentiment))
df_final.to_csv('submit.csv', index = False)
df_final.head()
