# import required packages



import pandas as pd 

import numpy as np

import os, gc, time, warnings



from scipy.misc import imread

from scipy import sparse

import scipy.stats as ss

from scipy.sparse import csr_matrix, hstack, vstack



import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec 

import seaborn as sns

from wordcloud import WordCloud ,STOPWORDS

from PIL import Image

import matplotlib_venn as venn

import pydot, graphviz

from IPython.display import Image



import string, re, nltk, collections

from nltk.util import ngrams

from nltk.corpus import stopwords

import spacy

from nltk import pos_tag

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer 

from nltk.tokenize import word_tokenize

from nltk.tokenize import TweetTokenizer   



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils.validation import check_X_y, check_is_fitted

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn import metrics

from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split



import tensorflow as tf

import keras.backend as K

from keras.models import Model, Sequential

from keras.utils import plot_model

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, BatchNormalization

from keras.layers import GRU, LSTM, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D

from keras.preprocessing import text, sequence

from keras.callbacks import Callback
# settings



os.environ['OMP_NUM_THREADS'] = '4'

start_time = time.time()

color = sns.color_palette()

sns.set_style("dark")

warnings.filterwarnings("ignore")



eng_stopwords = set(stopwords.words("english"))

lem = WordNetLemmatizer()

ps = PorterStemmer()

tokenizer = TweetTokenizer()



%matplotlib inline
# import the dataset



train = pd.read_csv("../input/midasiiitd/V1.4_Training.csv", encoding = 'latin-1')

dev = pd.read_csv("../input/midasiiitd/SubtaskA_Trial_Test_Labeled.csv", encoding = 'latin-1')

test = pd.read_csv("../input/midasiiitd/SubtaskA_EvaluationData.csv", encoding = 'latin-1')
# quick look at a few training examples



train.head(10)
print("Training data...")

train.info()
# class-imbalance in training data



suggestion_count = (train['label'].values == 1).astype(int).sum()

non_suggestion_count = (train['label'].values == 0).astype(int).sum()

print("Total sentences : " + str(train.shape[0]))

print("Total suggestions : " + str(suggestion_count))

print("Total non_suggestions : " + str(non_suggestion_count))
# oversampling to balance the training data



suggestions = train[train['label'].values == 1]



while suggestion_count < non_suggestion_count:

    random_suggestion = suggestions.sample()

    train = train.append(random_suggestion, ignore_index = True)

    suggestion_count = suggestion_count + 1



train.info()
# exploring the development/validation data



print("Development Set...")

dev.info()
# class-imbalance in development data



suggestion_count = (dev['label'].values == 1).astype(int).sum()

non_suggestion_count = (dev['label'].values == 0).astype(int).sum()

print("Total sentences : " + str(dev.shape[0]))

print("Total suggestions : " + str(suggestion_count))

print("Total non_suggestions : " + str(non_suggestion_count))
stopword = set(STOPWORDS)



# wordcloud for sentences with 'suggestion' label

subset = train[train.label == 1]

content = subset.sentence.values

wc = WordCloud(background_color = "black", max_words = 2000, stopwords = stopword)

wc.generate(" ".join(content))

plt.figure(figsize = (20,20))

plt.subplot(221)

plt.axis("off")

plt.title("Words frequented in 'suggestion' sentences", fontsize = 20)

plt.imshow(wc.recolor(colormap = 'viridis', random_state = 17), alpha = 0.98)



# wordcloud for sentences with 'non-suggestion' label

subset = train[train.label == 0]

content = subset.sentence.values

wc = WordCloud(background_color = "black", max_words = 2000, stopwords = stopword)

wc.generate(" ".join(content))

plt.subplot(222)

plt.axis("off")

plt.title("Words frequented in 'non-suggestion' sentences", fontsize = 20)

plt.imshow(wc.recolor(colormap = 'viridis', random_state = 17), alpha = 0.98)



plt.show()
# Aphost lookup dict



APPO = {

    "aren't" : "are not",

    "can't" : "cannot",

    "couldn't" : "could not",

    "didn't" : "did not",

    "doesn't" : "does not",

    "don't" : "do not",

    "hadn't" : "had not",

    "hasn't" : "has not",

    "haven't" : "have not",

    "he'd" : "he would",

    "he'll" : "he will",

    "he's" : "he is",

    "i'd" : "I would",

    "i'd" : "I had",

    "i'll" : "I will",

    "i'm" : "I am",

    "isn't" : "is not",

    "it's" : "it is",

    "it'll":"it will",

    "i've" : "I have",

    "let's" : "let us",

    "mightn't" : "might not",

    "mustn't" : "must not",

    "shan't" : "shall not",

    "she'd" : "she would",

    "she'll" : "she will",

    "she's" : "she is",

    "shouldn't" : "should not",

    "that's" : "that is",

    "there's" : "there is",

    "they'd" : "they would",

    "they'll" : "they will",

    "they're" : "they are",

    "they've" : "they have",

    "we'd" : "we would",

    "we're" : "we are",

    "weren't" : "were not",

    "we've" : "we have",

    "what'll" : "what will",

    "what're" : "what are",

    "what's" : "what is",

    "what've" : "what have",

    "where's" : "where is",

    "who'd" : "who would",

    "who'll" : "who will",

    "who're" : "who are",

    "who's" : "who is",

    "who've" : "who have",

    "won't" : "will not",

    "wouldn't" : "would not",

    "you'd" : "you would",

    "you'll" : "you will",

    "you're" : "you are",

    "you've" : "you have",

    "'re": " are",

    "wasn't": "was not",

    "we'll":" will",

    "didn't": "did not",

    "tryin'":"trying"

}
def clean(sentence):

    sentence = sentence.lower()

    sentence = re.sub('<.*>', '', sentence)

    sentence = re.sub("\\n", "", sentence)

    sentence = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", sentence)

    sentence = re.sub("\[\[.*\]", "", sentence)

    sentence = re.sub("[" + re.sub("\.","",string.punctuation) + "]", "", sentence)

    

    words = tokenizer.tokenize(sentence)

    

    words = [APPO[word] if word in APPO else word for word in words]

    words = [ps.stem(word) for word in words]

    words = [lem.lemmatize(word, "v") for word in words]

    words = [w for w in words if not w in eng_stopwords]

    

    clean_sent = " ".join(words)

    

    return(clean_sent)
# obtaining separate clean corpora for suggestion and non-suggestion classes



suggestion_corpus = train[train['label'].values == 1].sentence

suggestion_corpus = suggestion_corpus.append(dev[dev['label'].values == 1].sentence)

clean_suggestion_corpus = ""

for sentence in suggestion_corpus:

    clean_suggestion_corpus += clean(sentence)



non_suggestion_corpus = train[train['label'].values == 0].sentence

non_suggestion_corpus = non_suggestion_corpus.append(dev[dev['label'].values == 0].sentence)

clean_non_suggestion_corpus = ""

for sentence in non_suggestion_corpus:

    clean_non_suggestion_corpus += clean(sentence)
# top 20 bigrams in suggestion sentences



suggestion_bigrams = ngrams(clean_suggestion_corpus.split(), 2)

suggestion_bigrams_freq = collections.Counter(suggestion_bigrams)

suggestion_bigrams_freq.most_common(20)
# top 20 bigrams in non-suggestion sentences



non_suggestion_bigrams = ngrams(clean_non_suggestion_corpus.split(), 2)

non_suggestion_bigrams_freq = collections.Counter(non_suggestion_bigrams)

non_suggestion_bigrams_freq.most_common(20)
del(suggestions)

del(subset)

del(content)

del(stopword)

del(wc)

del(suggestion_corpus)

del(clean_suggestion_corpus)

del(non_suggestion_corpus)

del(clean_non_suggestion_corpus)

gc.collect()
# plot of sentence length against label



df = pd.concat([train, dev])

df['count_word'] = df['sentence'].apply(lambda x : len(x.split()))



plt.figure(figsize = (12, 6))

plt.suptitle("How is sentence length related to its label?", fontsize = 15)

count_word = df['count_word'].astype(int)

df['count_word'].loc[df['count_word'] > 100] = 100

plt.plot()

sns.violinplot(y = 'count_word', x = 'label', data = df, split = True, inner = "quart")

plt.xlabel('Suggestion?', fontsize = 12)

plt.ylabel('Number of words in a sentence', fontsize = 12)

plt.title("Number of sentences with a given word length", fontsize = 12)

plt.show()



del(df)

gc.collect()
# plot of mean word length against label



df = pd.concat([train, dev])

df['mean_word_len'] = df['sentence'].apply(lambda x : np.mean([len(word) for word in x.split()]))



plt.figure(figsize = (12, 6))

plt.suptitle("How is mean word length in a sentence related to its label?", fontsize = 15)

mean_word_len = df['mean_word_len'].astype(int)

df['mean_word_len'].loc[df['mean_word_len'] > 10] = 10

plt.plot()

sns.violinplot(y = 'mean_word_len', x = 'label', data = df, split = True, inner = "quart")

plt.xlabel('Suggestion?', fontsize = 12)

plt.ylabel('Mean word length in sentence', fontsize = 12)

plt.title("Number of sentences with a given mean word length", fontsize = 12)

plt.show()



del(df)

gc.collect()
# corpus containing all the sentences in train, development and test data



corpus = (pd.concat([train.iloc[:, 0:2], dev.iloc[:, 0:2], test.iloc[:, 0:2]])).sentence

clean_corpus = corpus.apply(lambda x : clean(x))
# tf-idf vectors with unigram features



unigram_tfv = TfidfVectorizer(strip_accents = 'unicode', analyzer = 'word', ngram_range = (1,1),

                              sublinear_tf = 1, stop_words = 'english')

unigram_tfv.fit(clean_corpus)



train_unigrams = unigram_tfv.transform(clean_corpus.iloc[:train.shape[0]])

dev_unigrams = unigram_tfv.transform(clean_corpus.iloc[train.shape[0]:train.shape[0]+dev.shape[0]])

test_unigrams = unigram_tfv.transform(clean_corpus.iloc[train.shape[0]+dev.shape[0]:])
# tf-idf vectors with bigram and trigram features



btgram_tfv = TfidfVectorizer(strip_accents = 'unicode', analyzer = 'word', ngram_range = (2,3),

            sublinear_tf = 1, stop_words = 'english')

btgram_tfv.fit(clean_corpus)



train_btgrams = btgram_tfv.transform(clean_corpus.iloc[:train.shape[0]])

dev_btgrams = btgram_tfv.transform(clean_corpus.iloc[train.shape[0]:train.shape[0]+dev.shape[0]])

test_btgrams = btgram_tfv.transform(clean_corpus.iloc[train.shape[0]+dev.shape[0]:])
# tf-idf vectors with char n-gram features



charngram_tfv = TfidfVectorizer(strip_accents = 'unicode', analyzer = 'char', ngram_range = (1,5),

                sublinear_tf = 1, stop_words = 'english')

charngram_tfv.fit(clean_corpus)



train_charngrams =  charngram_tfv.transform(clean_corpus.iloc[:train.shape[0]])

dev_charngrams = charngram_tfv.transform(clean_corpus.iloc[train.shape[0]:train.shape[0]+dev.shape[0]])

test_charngrams = charngram_tfv.transform(clean_corpus.iloc[train.shape[0]+dev.shape[0]:])
# evaluation functions for different models



def lgb_f1_score(preds, train_data):

    y_train = train_data.get_label()

    preds = (preds >= 0.5).astype(int)

    return 'f1_score', f1_score(y_train, preds), True



def xgb_f1_score(preds, train_data):

    y_train = train_data.get_label()

    preds = (preds >= 0.5).astype(int)

    return 'f1_score', f1_score(y_train, preds)



def nn_f1_score(y_true, y_pred):

    y_pred = tf.cast((y_pred >= 0.5), tf.float32)

    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis = 0)

    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis = 0)

    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis = 0)

    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis = 0)



    p = tp / (tp + fp + K.epsilon())

    r = tp / (tp + fn + K.epsilon())



    f1 = 2*p*r / (p+r+K.epsilon())

    f1_score = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)

    return K.mean(f1_score)
# dataframes for blending



train_labels = pd.DataFrame()

dev_labels = pd.DataFrame()
# preparing data for statistical and GBDT models



x_train = hstack((train_unigrams, train_btgrams, train_charngrams)).tocsr()

y_train = train['label'].values

x_dev = hstack((dev_unigrams, dev_btgrams, dev_charngrams)).tocsr()

y_dev = dev['label'].values

x_test = hstack((test_unigrams, test_btgrams, test_charngrams)).tocsr()
# logistic regression classifier



clf = LogisticRegression(C = 0.1, solver = 'liblinear')

clf.fit(x_train, y_train)



lr_dev_pred = clf.predict_proba(x_dev)[:, 1]

lr_test_pred = clf.predict_proba(x_test)[:, 1]



train_labels['lr'] = (clf.predict_proba(x_train)[:, 1] >= 0.5).astype(int)

dev_labels['lr'] = (clf.predict_proba(x_dev)[:, 1] >= 0.5).astype(int)



y_pred = (lr_dev_pred >= 0.5).astype(int)

lr_precision = precision_score(y_dev, y_pred)

lr_recall = recall_score(y_dev, y_pred)

lr_f1 = f1_score(y_dev, y_pred)



print("Logistic Regression...")

print("Precision score : " + str(lr_precision))

print("Recall score : " + str(lr_recall))

print("F1 score : " + str(lr_f1))
# SVM classifier



# reducing the number of features using Singular Value Decomposition

svd = TruncatedSVD(n_components = 15)

svd.fit(vstack((x_train, x_dev, x_test)).tocsr())

x_train_svd = svd.transform(x_train)

x_dev_svd = svd.transform(x_dev)

x_test_svd = svd.transform(x_test)



# scaling the data obtained from SVD

scaler = StandardScaler()

scaler.fit(np.concatenate((x_train_svd, x_dev_svd, x_test_svd)))

x_train_svd = scaler.transform(x_train_svd)

x_dev_svd = scaler.transform(x_dev_svd)

x_test_svd = scaler.transform(x_test_svd)



clf = SVC(C = 0.1, probability = True)

clf.fit(x_train_svd, y_train)



svm_dev_pred = clf.predict_proba(x_dev_svd)[:, 1]

svm_test_pred = clf.predict_proba(x_test_svd)[:, 1]



train_labels['svm'] = (clf.predict_proba(x_train_svd)[:, 1] >= 0.5).astype(int)

dev_labels['svm'] = (clf.predict_proba(x_dev_svd)[:, 1] >= 0.5).astype(int)



y_pred = (svm_dev_pred >= 0.5).astype(int)

svm_precision = precision_score(y_dev, y_pred)

svm_recall = recall_score(y_dev, y_pred)

svm_f1 = f1_score(y_dev, y_pred)



print("SVM Classifier...")

print("Precision score : " + str(svm_precision))

print("Recall score : " + str(svm_recall))

print("F1 score : " + str(svm_f1))
# lgbm classifier



import lightgbm as lgb



d_train = lgb.Dataset(x_train, label = y_train)

d_dev = lgb.Dataset(x_dev, label = y_dev)

valid_sets = [d_train, d_dev]



params = {'learning_rate': 0.2,

          'application': 'binary',

          'num_leaves': 31,

          'verbosity': -1,

          'bagging_fraction': 0.8,

          'feature_fraction': 0.6,

          'nthread': 4,

          'lambda_l1': 1,

          'lambda_l2': 1}



model = lgb.train(params,

                  train_set = d_train,

                  num_boost_round = 25,

                  valid_sets = valid_sets,

                  feval = lgb_f1_score,

                  verbose_eval = False)



lgb_dev_pred = model.predict(x_dev)

lgb_test_pred = model.predict(x_test)



train_labels['lgb'] = (model.predict(x_train) >= 0.5).astype(int)

dev_labels['lgb'] = (model.predict(x_dev) >= 0.5).astype(int)



y_pred = (lgb_dev_pred >= 0.5).astype(int)

lgb_precision = precision_score(y_dev, y_pred)

lgb_recall = recall_score(y_dev, y_pred)

lgb_f1 = f1_score(y_dev, y_pred)



print("LGBM Classifier...")

print("Precision score : " + str(lgb_precision))

print("Recall score : " + str(lgb_recall))

print("F1 score : " + str(lgb_f1))
del(d_train)

del(d_dev)

del(model)

gc.collect()
# XGBoost classifier



import xgboost as xgb



d_train = xgb.DMatrix(x_train, label = y_train)

d_dev = xgb.DMatrix(x_dev, label = y_dev)

d_test = xgb.DMatrix(x_test)

evallist = [(d_train, 'train'), (d_dev, 'valid')]



params = {'booster' : 'gbtree',

          'nthread' : 4,

          'eta' : 0.2,

          'max_depth' : 6,

          'min_child_weight' : 4,

          'subsample' : 0.7,

          'colsample_bytree' : 0.7,

          'objective' : 'binary:logistic'}



model = xgb.train(params, 

                  d_train, 

                  num_boost_round = 21,

                  evals = evallist,

                  feval = xgb_f1_score,

                  verbose_eval = False)



xgb_dev_pred = model.predict(d_dev, ntree_limit = 21)

xgb_test_pred = model.predict(d_test, ntree_limit = 21)



train_labels['xgb'] = (model.predict(d_train, ntree_limit = 21) >= 0.5).astype(int)

dev_labels['xgb'] = (model.predict(d_dev, ntree_limit = 21) >= 0.5).astype(int)



y_pred = (xgb_dev_pred >= 0.5).astype(int)

xgb_precision = precision_score(y_dev, y_pred)

xgb_recall = recall_score(y_dev, y_pred)

xgb_f1 = f1_score(y_dev, y_pred)



print("XGBoost Classifier...")

print("Precision score : " + str(xgb_precision))

print("Recall score : " + str(xgb_recall))

print("F1 score : " + str(xgb_f1))
del(x_train)

del(y_train)

del(x_dev)

del(y_dev)

del(d_train)

del(d_dev)

del(model)

gc.collect()
# preparing data for Neural Network



EMBEDDING_FILE = '../input/fasttext/crawl-300d-2M.vec'



max_features = 10760

maxlen = 600

embed_size = 300



pos_tags_train = train['sentence'].apply(lambda x : " ".join(item[1] for item in pos_tag(word_tokenize(x)))).values

pos_tags_dev = dev['sentence'].apply(lambda x : " ".join(item[1] for item in pos_tag(word_tokenize(x)))).values

pos_tags_test = test['sentence'].apply(lambda x : " ".join(item[1] for item in pos_tag(word_tokenize(x)))).values



x_train = train['sentence'].values + " " + pos_tags_train

y_train = train['label'].values

x_dev = dev['sentence'].values + " " + pos_tags_dev

y_dev = dev['label'].values

x_test = test['sentence'].values + " " + pos_tags_test



tokenizer = text.Tokenizer(num_words = max_features)

tokenizer.fit_on_texts(list(x_train) + list(x_dev) + list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)

x_dev = tokenizer.texts_to_sequences(x_dev)

x_test = tokenizer.texts_to_sequences(x_test)

x_train = sequence.pad_sequences(x_train, maxlen = maxlen)

x_dev = sequence.pad_sequences(x_dev, maxlen = maxlen)

x_test = sequence.pad_sequences(x_test, maxlen = maxlen)



def get_coefs(word, *arr): 

    return word, np.asarray(arr, dtype = 'float32')



embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))



word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.zeros((nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features:

        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
# Hybrid Neural Network classifier



inp = Input(shape = (maxlen, ))

x = Embedding(max_features, embed_size, weights = [embedding_matrix])(inp)

x = SpatialDropout1D(0.2)(x)

x = Bidirectional(GRU(100, return_sequences = True))(x)

x = Conv1D(50, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)

avg_pool = GlobalAveragePooling1D()(x)

max_pool = GlobalMaxPooling1D()(x)

conc = concatenate([avg_pool, max_pool])

outp = Dense(1, activation = "sigmoid")(conc)

    

model = Model(inputs = inp, outputs = outp)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = [nn_f1_score])

model.fit(x_train, y_train, batch_size = 128, epochs = 1, validation_data = (x_dev, y_dev), verbose = 1)
nn_dev_pred = model.predict(x_dev, batch_size = 128, verbose = 1)

nn_test_pred = model.predict(x_test, batch_size = 128, verbose = 1)



train_labels['nn'] = (model.predict(x_train, batch_size = 128, verbose = 1) >= 0.5).astype(int)

dev_labels['nn'] = (model.predict(x_dev, batch_size = 128, verbose = 1) >= 0.5).astype(int)



y_pred = (nn_dev_pred >= 0.5).astype(int)

nn_precision = precision_score(y_dev, y_pred)

nn_recall = recall_score(y_dev, y_pred)

nn_f1 = f1_score(y_dev, y_pred)



print("Hybrid Neural Network Classifier...")

print("Precision score : " + str(nn_precision))

print("Recall score : " + str(nn_recall))

print("F1 score : " + str(nn_f1))
plot_model(model, to_file = 'model.png')
def getmodel():

    model = Sequential()

    model.add(Dense(256, input_dim = 5, activation = 'relu'))

    model.add(Dense(64, activation = 'relu'))

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = [nn_f1_score])

    return model
# Stacking all the models



stacked_model = getmodel()



stacked_model.fit(train_labels, y_train, batch_size = 128, epochs = 2, validation_data = (dev_labels, y_dev), 

          verbose = 1)



stacked_dev_pred = stacked_model.predict(dev_labels, batch_size = 128, verbose = 1)



y_pred = (stacked_dev_pred >= 0.5).astype(int)

stack_precision = precision_score(y_dev, y_pred)

stack_recall = recall_score(y_dev, y_pred)

stack_f1 = f1_score(y_dev, y_pred)



print("Stacked Models Classifier...")

print("Precision score : " + str(stack_precision))

print("Recall score : " + str(stack_recall))

print("F1 score : " + str(stack_f1))
# saving the test labels to output csv file



y_test = (nn_test_pred[:, 0] >= 0.5).astype(int)

submission = pd.read_csv("../input/midasiiitd/SubtaskA_EvaluationData.csv")

submission.drop(['label'], axis = 1)

submission['label'] = y_test

submission.to_csv("sanket_rai.csv", index = False)