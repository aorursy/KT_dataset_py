import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import warnings
import collections
import sklearn as sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.feature_extraction import text 
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm, tree
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import gensim
from collections import defaultdict
from itertools import islice
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from keras import Sequential
from keras.layers import Bidirectional, GlobalMaxPool1D,Dense, Input, Embedding, Dropout, LSTM, CuDNNGRU
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


import time
%matplotlib inline
warnings.filterwarnings('ignore')
train_df = pd.DataFrame.from_csv("../input/train.csv")
test_df =  pd.DataFrame.from_csv("../input/test.csv")
train_df.head(5)
text =" ".join(train_df.question_text)
 # Create the wordcloud object
wordcloud = WordCloud(width=1024, height=1024, margin=0).generate(text)
 
# Display the generated image:
fig,ax = plt.subplots(1,1,figsize=(10,10))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
ax.margins(x=0, y=0)
plt.show()
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.set_title("Target Status")
explode=(0,0.1)
labels ='0','1'
ax.pie(list(dict(collections.Counter(list(train_df.target))).values()), explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
del text
X_train, X_test, y_train, y_test = train_test_split(train_df.question_text, train_df.target, test_size=0.33, random_state=42)
#initiation of countVectorizer
count_vectorizer = CountVectorizer(min_df=5, stop_words='english')
vect = count_vectorizer.fit(train_df.question_text)

X_vect_train = vect.transform(X_train) # documents-terms matrix of training set
X_vect_test = vect.transform(X_test) # documents-terms matrix of testing set

tf_train_transformer = TfidfTransformer(use_idf=False).fit(X_vect_train)
tf_test_transformer =  TfidfTransformer(use_idf=False).fit(X_vect_test)

xtrain_tf = tf_train_transformer.transform(X_vect_train)
xtest_tf = tf_test_transformer.transform(X_vect_test)
type(xtrain_tf),xtrain_tf.shape, train_df.shape
xtrain_tf[:5].todense()
count_vectorizer.get_feature_names()[0:5]
count_vectorizer.get_feature_names()[30000:30005]
count_vectorizer.get_feature_names()[-5:]
[{k:count_vectorizer.vocabulary_[k]} for k in list(count_vectorizer.vocabulary_)[:10]]
list(sklearn.feature_extraction.text.ENGLISH_STOP_WORDS)[:10]
results_df = pd.DataFrame()
# MULTINOMINA_NAIVE_BAYES
nb_ = MultinomialNB()
nb_clf = nb_.fit(X=xtrain_tf, y=y_train)
results_df.set_value("NB" , "countVectorizer" , accuracy_score(y_test,nb_clf.predict(xtest_tf)))
# RANDOM_FORES_CLASSFIER
rf_clf = RandomForestClassifier(n_estimators=25, max_depth=15,random_state=42)
rf_clf.fit(X=xtrain_tf,y=y_train)
results_df.set_value("RF" , "countVectorizer" , accuracy_score(y_test,rf_clf.predict(xtest_tf)))
#MLP_CLASSIFIER
mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-4,hidden_layer_sizes=(20,10, 2), random_state=42)
mlp_clf.fit(X=xtrain_tf, y=y_train)                         
results_df.set_value("MLP" , "countVectorizer" , accuracy_score(y_test,mlp_clf.predict(xtest_tf)))
#LoggicRegression
lreg_clf = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=42)
lreg_clf.fit(X=xtrain_tf, y=y_train)                         
results_df.set_value("LREG" , "countVectorizer" , accuracy_score(y_test,lreg_clf.predict(xtest_tf)))
results_df
fig,axes=plt.subplots(1,1,figsize=(8,8))
axes.set_ylabel("Accuracy")
plt.ylim((.92,.97))
results_df.plot(kind="bar",ax=axes)
text_ = train_df.question_text
targets_ = train_df.target
class GetSentences(object):
    def __iter__(self):
        counter = 0
        for sentence_iter in text_:
            tmp_sentence = sentence_iter
            counter += 1
            yield tmp_sentence.split()
len(text_)
num_features = 200  # Word vector dimensionality
min_word_count = 5  # Minimum word count
num_workers = 4  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words
get_sentence = GetSentences()
model = gensim.models.Word2Vec(sentences=get_sentence, min_count=min_word_count, size=num_features, workers=4)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))
next(iter(w2v.items()))
list(islice(model.wv.vocab.items(), 5))
"MOST SIMILAR WORD TO MAN: {} AND MOST SIMILAR WORD TO QUERA:{} ".format(model.most_similar(['man'],topn=1),model.most_similar(["quora"],topn=3))
model.most_similar(positive=["united","state"])
# most similar words to 'sex'
model.most_similar(positive=["sex"])
model.most_similar(positive=['gangbang', 'sex'], negative=['man'], topn=5) 
model.most_similar(positive=['woman', 'king'], negative=['man'], topn=2) 
model.save('word2vec.model')
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        # self.dim = len(word2vec.itervalues().next())
        self.dim = len(next(iter(self.word2vec.items()))[1])

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0) for words in X])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        # self.dim = len(word2vec.itervalues().next())
        self.dim = len(next(iter((word2vec.items()))))

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


"""
NOTE:
As these processes are time consuming, I have simulated these classifiers on my local machine. Committing these codes (In each commit waiting until finishing whole the 
classifiers was really annoying for me) so I have commented the lines which are representing training process. You can easily uncomment them and try to
train classifier by yourself ;-).
"""

print("EXTRA TREE ...")
# 1- MEAN VECTORIZER
etree_w2v = Pipeline([("w2v mean vectorizer", MeanEmbeddingVectorizer(w2v)), ("extra trees", ExtraTreesClassifier(n_estimators=25))])
# etree_w2v.fit(X=X_train, y=y_train)
# results_df.set_value("ExtraTree", "w2v_mean", accuracy_score(y_test, etree_w2v.predict(X_test)))

# 2- TFIDF VECTORIZER
etree_w2v_tfidf = Pipeline([("w2v tfidf vectorizer", TfidfEmbeddingVectorizer(w2v)), ("extra trees", ExtraTreesClassifier(n_estimators=25))])
# etree_w2v_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("ExtraTree", "w2v_tfidf", accuracy_score(y_test, etree_w2v_tfidf.predict(X_test)))

####SVM####
print("SVM ... ")
# 1- MAIN VECTORIZER
svm_w2v = Pipeline([("w2v mean vectorizer", MeanEmbeddingVectorizer(w2v)), ("SVM", LinearSVC(random_state=0, tol=1e-4))])
# svm_w2v.fit(X=X_train, y=y_train)
# results_df.set_value("SVM", "w2v_mean", accuracy_score(y_test, etree_w2v_tfidf.predict(X_test)))

# 2- TFIDF VECTORIZER
svm_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)), ("SVM", LinearSVC(random_state=0, tol=1e-4))])
# svm_w2v_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("SVM", "w2v_tfidf", accuracy_score(y_test, svm_w2v_tfidf.predict(X_test)))

####MLP####
print("MLP ... ")
# 1- MAIN VECTORIZER
mlp_w2v = Pipeline(
    [("w2v mean vectorizer", MeanEmbeddingVectorizer(w2v)),
     ("MLP", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,10, 2), random_state=1))])
# mlp_w2v.fit(X=X_train, y=y_train)
# results_df.set_value("MLP", "w2v_mean", accuracy_score(y_test, mlp_w2v.predict(X_test)))

# 2- TFIDF VECTORIZER
mlp_w2v_tfidf = Pipeline(
    [("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
     ("MLP", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,10,2), random_state=1))])
# mlp_w2v_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("MLP", "w2v_tfidf", accuracy_score(y_test, mlp_w2v_tfidf.predict(X_test)))
results_df.set_value("NB","w2v_mean",0.476575)
results_df.set_value("ExtraTree","w2v_mean",0.939119)
results_df.set_value("SVM","w2v_mean",0.939144)
results_df.set_value("MLP","w2v_mean",0.939035)
results_df.set_value("LREG","w2v_mean",0.938836)

results_df.set_value("NB","w2v_tfidf",0.260479)
results_df.set_value("ExtraTree","w2v_tfidf",0.939144)
results_df.set_value("SVM","w2v_tfidf",0.939033)
results_df.set_value("MLP","w2v_tfidf",0.939035)
results_df.set_value("LREG","w2v_tfidf",0.953311)
results_df
fig,axes=plt.subplots(1,1,figsize=(8,8))
axes.set_ylabel("Accuracy")
axes.set_title("word2vec results for 67% training and 23% testing")
# plt.ylim((.93,.95))
results_df[["w2v_mean","w2v_tfidf"]].dropna().plot(kind="bar",ax=axes)
del w2v
del model
import gc; gc.collect()
time.sleep(10)
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
glov = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))
next(iter(glov.values()))
"""
NOTE:
As these processes are time consuming, I have simulated them on my local machine. Committing these codes (Waiting until finishing whole the trainin process for
any commit really bothers me) so I have commented the lines which have belonged to training process. You can easily uncomment them and try to
train classifier by yourself ;-).
"""
# GLOVE MODEL

###NB####
print("Multinomina NB ...")
nb_glov = Pipeline([("glov mean vectorizer", MeanEmbeddingVectorizer(glov)), ("Guassian NB", GaussianNB())])
# nb_glov.fit(X=X_train, y=y_train)
# results_df.set_value("GB", "glov_mean", accuracy_score(y_test, nb_glov.predict(X_test)))

# 2- TFIDF VECTORIZER
nb_glov_tfidf = Pipeline([("glov tfidf vectorizer", TfidfVectorizer(glov)), ("transform", TfidfTransformer()), ("Guassian NB", MultinomialNB())])
# nb_glov_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("GB", "glov_tfidf", accuracy_score(y_test, nb_glov_tfidf.predict(X_test)))

###EXTRA TREE####
print("EXTRA TREE ...")
# 1- MEAN VECTORIZER
etree_glov = Pipeline([("glov mean vectorizer", MeanEmbeddingVectorizer(glov)), ("extra trees", ExtraTreesClassifier(n_estimators=25))])
# etree_glov.fit(X=X_train, y=y_train)
# results_df.set_value("ExtraTree", "glov_mean", accuracy_score(y_test, etree_glov.predict(X_test)))

# 2- TFIDF VECTORIZER
etree_glov_tfidf = Pipeline(
    [("glov tfidf vectorizer", TfidfVectorizer(glov)), ("transform", TfidfTransformer()), ("extra trees", ExtraTreesClassifier(n_estimators=25))])
# etree_glov_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("ExtraTree", "glov_tfidf", accuracy_score(y_test, etree_glov_tfidf.predict(X_test)))

####SVM####
print("SVM ... ")
# 1- MAIN VECTORIZER
svm_glov = Pipeline([("glov mean vectorizer", MeanEmbeddingVectorizer(glov)), ("SVM", LinearSVC(random_state=42, tol=1e-5))])
# svm_glov.fit(X=X_train, y=y_train)
# results_df.set_value("SVM", "glov_mean", accuracy_score(y_test, svm_glov.predict(X_test)))

# 2- TFIDF VECTORIZER
svm_glov_tfidf = Pipeline(
    [("glov tfidf vectorizer", TfidfVectorizer(glov)), ("transform", TfidfTransformer()), ("SVM", LinearSVC(random_state=0, tol=1e-5))])
# svm_glov_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("SVM", "glov_tfidf", accuracy_score(y_test, svm_glov_tfidf.predict(X_test)))

####MLP####
print("MLP ... ")
# 1- MAIN VECTORIZER
mlp_glov = Pipeline(
    [("glov mean vectorizer", MeanEmbeddingVectorizer(glov)),
     ("MLP", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 10, 2), random_state=42))])
# mlp_glov.fit(X=X_train, y=y_train)
# results_df.set_value("MLP", "glov_mean", accuracy_score(y_test, mlp_glov.predict(X_test)))

# 2- TFIDF VECTORIZER
mlp_glov_tfidf = Pipeline(
    [("glov tfidf vectorizer", TfidfVectorizer(glov)), ("transform", TfidfTransformer()),
     ("MLP", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 10, 2), random_state=42))])
# mlp_glov_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("MLP", "glov_tfidf", accuracy_score(y_test, mlp_glov_tfidf.predict(X_test)))


####LREG####
print("LREG ... ")
# 1- MAIN VECTORIZER
lreg_glov = Pipeline(
    [("glov mean vectorizer", MeanEmbeddingVectorizer(glov)),
     ("LREG", LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42))])
# lreg_glov.fit(X=X_train, y=y_train)
# results_df.set_value("LREG", "glov_mean", accuracy_score(y_test, lreg_glov.predict(X_test)))

# 2- TFIDF VECTORIZER
lreg_glov_tfidf = Pipeline(
    [("glov tfidf vectorizer", TfidfVectorizer(glov)), ("transform", TfidfTransformer()),
     ("LREG", LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42))])
# lreg_glov_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("LEREG", "glov_tfidf", accuracy_score(y_test, lreg_glov_tfidf.predict(X_test)))
results_df.set_value("NB","glov_mean",0.533364)
results_df.set_value("ExtraTree","glov_mean",0.939131)
results_df.set_value("SVM","glov_mean",0.939140)
results_df.set_value("MLP","glov_mean",0.938973)
results_df.set_value("LREG","glov_mean",0.938836)

results_df.set_value("NB","glov_tfidf",0.941720)
results_df.set_value("ExtraTree","glov_tfidf",0.945798)
results_df.set_value("SVM","glov_tfidf",0.953854)
results_df.set_value("MLP","glov_tfidf",0.939035)
results_df.set_value("LREG","glov_tfidf",0.953311)
# fig,axes=plt.subplots(1,1,figsize=(15,8))
# axes.set_ylabel("Accuracy")
# axes.set_title("GLOVE results for 67% training and 23% testing")
# results_df.plot(kind="bar",ax=axes)
del glov
import gc; gc.collect()
time.sleep(10)
EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
program = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE,encoding="latin1"))
next(iter(program.values()))
"""
NOTE:
As these processes are time consuming, I have simulated them on my local machine. Committing these codes (Waiting until finishing whole the trainin process for
any commit really bothers me) so I have commented the lines which have belonged to training process. You can easily uncomment them and try to
train classifier by yourself ;-).
"""

print("Multinomina NB ...")
nb_program = Pipeline([("program mean vectorizer", MeanEmbeddingVectorizer(program)), ("Guassian NB", GaussianNB())])
# nb_program.fit(X=X_train, y=y_train)
# results_df.set_value("GB", "program_mean", accuracy_score(y_test, nb_program.predict(X_test)))

# 2- TFIDF VECTORIZER
nb_program_tfidf = Pipeline([("program tfidf vectorizer", TfidfVectorizer(program)), ("transform", TfidfTransformer()), ("Guassian NB", MultinomialNB())])
# nb_program_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("GB", "program_tfidf", accuracy_score(y_test, nb_program_tfidf.predict(X_test)))


# PROGRAM-300 MODEL
####EXTRA TREE####
print("EXTRA TREE ...")
# 1- MEAN VECTORIZER
etree_program = Pipeline([("program mean vectorizer", MeanEmbeddingVectorizer(program)), ("extra trees", ExtraTreesClassifier(n_estimators=20))])
# etree_program.fit(X=X_train, y=y_train)
# results_df.set_value("ExtraTree", "program_mean", accuracy_score(y_test, etree_program.predict(X_test)))


# 2- TFIDF VECTORIZER
etree_program_tfidf = Pipeline([("program tfidf vectorizer", TfidfEmbeddingVectorizer(program)), ("extra trees", ExtraTreesClassifier(n_estimators=20))])
# etree_program_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("ExtraTree", "program_tfidf", accuracy_score(y_test, etree_program_tfidf.predict(X_test)))

####SVM####
print("SVM ... ")
#1- MAIN VECTORIZER
svm_program = Pipeline([("program mean vectorizer", MeanEmbeddingVectorizer(program)), ("SVM", LinearSVC(random_state=0, tol=1e-5))])
# svm_program.fit(X=X_train, y=y_train)
# results_df.set_value("SVM", "program_mean", accuracy_score(y_test, svm_program.predict(X_test)))

# 2- TFIDF VECTORIZER
svm_program_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(program)), ("SVM", LinearSVC(random_state=0, tol=1e-5))])
# svm_program_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("SVM", "program_tfidf", accuracy_score(y_test, svm_program_tfidf.predict(X_test)))

####MLP####
print("MLP ... ")
# 1- MAIN VECTORIZER
mlp_program = Pipeline(
    [("program mean vectorizer", MeanEmbeddingVectorizer(program)),
     ("MLP", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,10, 2), random_state=42))])
# mlp_program.fit(X=X_train, y=y_train)
# results_df.set_value("MLP", "program_mean", accuracy_score(y_test, mlp_program.predict(X_test)))

# 2- TFIDF VECTORIZER
mlp_program_tfidf = Pipeline(
    [("word2vec vectorizer", TfidfEmbeddingVectorizer(program)),
     ("MLP", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,10, 2), random_state=42))])
# mlp_program_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("MLP", "program_tfidf", accuracy_score(y_test, mlp_program_tfidf.predict(X_test)))

####LREG####
print("LREG ... ")
# 1- MAIN VECTORIZER
lreg_program = Pipeline(
    [("program mean vectorizer", MeanEmbeddingVectorizer(program)),
     ("LREG", LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42))])
# lreg_program.fit(X=X_train, y=y_train)
# results_df.set_value("LREG", "program_mean", accuracy_score(y_test, lreg_program.predict(X_test)))

# 2- TFIDF VECTORIZER
lreg_program_tfidf = Pipeline(
    [("program tfidf vectorizer", TfidfVectorizer(program)), ("transform", TfidfTransformer()),
     ("LREG", LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42))])
# lreg_program_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("LREG", "program_tfidf", accuracy_score(y_test, lreg_program_tfidf.predict(X_test)))
results_df.set_value("NB","program_mean",0.559782)
results_df.set_value("ExtraTree","program_mean",0.939124)
results_df.set_value("SVM","program_mean",0.939026)
results_df.set_value("MLP","program_mean",0.939035)
results_df.set_value("LREG","program_mean",0.938989)

results_df.set_value("NB","program_tfidf",0.941720)
results_df.set_value("ExtraTree","program_tfidf",0.945717)
results_df.set_value("SVM","program_tfidf",0.953854)
results_df.set_value("MLP","program_tfidf",0.939035)
results_df.set_value("LREG","program_tfidf",0.953311)
# fig,axes=plt.subplots(1,1,figsize=(15,8))
# axes.set_ylabel("Accuracy")
# axes.set_title("PROGRAM results for 67% training and 23% testing")
# results_df.plot(kind="bar",ax=axes)
del program
import gc; gc.collect()
time.sleep(10)
model = gensim.models.KeyedVectors.load_word2vec_format(
    fname='../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin',
    binary=True)
google = dict(zip(model.wv.index2word, model.wv.syn0))
next(iter(google.values()))
"""
NOTE:
As these processes are time consuming, I have simulated them on my local machine. Committing these codes (Waiting until finishing whole the trainin process for
any commit really bothers me) so I have commented the lines which have belonged to training process. You can easily uncomment them and try to
train classifier by yourself ;-).
"""



print("Multinomina NB ...")
google_google = Pipeline([("google mean vectorizer", MeanEmbeddingVectorizer(google)), ("Guassian NB", GaussianNB())])
# google_google.fit(X=X_train, y=y_train)
# results_df.set_value("GB", "google_mean", accuracy_score(y_test, google_google.predict(X_test)))

# 2- TFIDF VECTORIZER
google_google_tfidf = Pipeline([("google tfidf vectorizer", TfidfVectorizer(google)), ("transform", TfidfTransformer()), ("Guassian NB", MultinomialNB())])
# google_google_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("GB", "google_tfidf", accuracy_score(y_test, google_google_tfidf.predict(X_test)))


# GOOGLE_NEWS_VEC MODEL
####EXTRA TREE####
print("EXTRA TREE ...")
# 1- MEAN VECTORIZER
etree_google = Pipeline([("google mean vectorizer", MeanEmbeddingVectorizer(google)), ("extra trees", ExtraTreesClassifier(n_estimators=20))])
# etree_google.fit(X=X_train, y=y_train)
# results_df.set_value("ExtraTree", "google_mean", accuracy_score(y_test, etree_google.predict(X_test)))

# 2- TFIDF VECTORIZER
etree_google_tfidf = Pipeline([("google tfidf vectorizer", TfidfEmbeddingVectorizer(google)), ("extra trees", ExtraTreesClassifier(n_estimators=20))])
# etree_google_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("ExtraTree", "google_tfidf", accuracy_score(y_test, etree_google_tfidf.predict(X_test)))

####SVM####
print("SVM ... ")
#1- MAIN VECTORIZER
svm_google = Pipeline([("google mean vectorizer", MeanEmbeddingVectorizer(google)), ("SVM", LinearSVC(random_state=0, tol=1e-5))])
# svm_google.fit(X=X_train, y=y_train)
# results_df.set_value("SVM", "google_mean", accuracy_score(y_test, svm_google.predict(X_test)))

# 2- TFIDF VECTORIZER
svm_google_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(google)), ("SVM", LinearSVC(random_state=0, tol=1e-5))])
# svm_google_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("SVM", "google_tfidf", accuracy_score(y_test, svm_google_tfidf.predict(X_test)))

####MLP####
print("MLP ... ")
# 1- MAIN VECTORIZER
mlp_google = Pipeline(
    [("google mean vectorizer", MeanEmbeddingVectorizer(google)),
     ("MLP", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))])
# mlp_google.fit(X=X_train, y=y_train)
# results_df.set_value("MLP", "google_mean", accuracy_score(y_test, mlp_google.predict(X_test)))

# 2- TFIDF VECTORIZER
mlp_google_tfidf = Pipeline(
    [("word2vec vectorizer", TfidfEmbeddingVectorizer(google)),
     ("MLP", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))])
# mlp_google_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("MLP", "google_tfidf", accuracy_score(y_test, mlp_google_tfidf.predict(X_test)))

####LREG####
print("LREG ... ")
# 1- MAIN VECTORIZER
lreg_google = Pipeline(
    [("google mean vectorizer", MeanEmbeddingVectorizer(google)),
     ("LREG", LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42))])
# lreg_google.fit(X=X_train, y=y_train)
# results_df.set_value("LREG", "google_mean", accuracy_score(y_test, lreg_google.predict(X_test)))

# 2- TFIDF VECTORIZER
lreg_google_tfidf = Pipeline(
    [("google tfidf vectorizer", TfidfVectorizer(google)), ("transform", TfidfTransformer()),
     ("LREG", LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42))])
# lreg_google_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("LREG", "google_tfidf", accuracy_score(y_test, lreg_google_tfidf.predict(X_test)))
results_df.set_value("NB","google_mean",0.527487)
results_df.set_value("ExtraTree","google_mean",0.939126)
results_df.set_value("SVM","google_mean",0.939038)
results_df.set_value("MLP","google_mean",0.939035)
results_df.set_value("LREG","google_mean",0.938806)

results_df.set_value("NB","google_tfidf",0.941720)
results_df.set_value("ExtraTree","google_tfidf",0.945448)
results_df.set_value("SVM","google_tfidf",0.953854)
results_df.set_value("MLP","google_tfidf",0.939035)
results_df.set_value("LREG","google_tfidf",0.953311)
# fig,axes=plt.subplots(1,1,figsize=(15,8))
# axes.set_ylabel("Accuracy")
# axes.set_title("GOOGLE_NEWS results for 67% training and 23% testing")
# results_df.plot(kind="bar",ax=axes)
del google
del model
import gc; gc.collect()
time.sleep(10)
model = gensim.models.KeyedVectors.load_word2vec_format(fname='../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec',encoding='utf-8')
wiki = dict(zip(model.wv.index2word, model.wv.syn0))
next(iter(wiki.values()))
"""
NOTE:
As these processes are time consuming, I have simulated them on my local machine. Committing these codes (Waiting until finishing whole the trainin process for
any commit really bothers me) so I have commented the lines which have belonged to training process. You can easily uncomment them and try to
train classifier by yourself ;-).
"""


print("Multinomina NB ...")
wiki_wiki = Pipeline([("wiki mean vectorizer", MeanEmbeddingVectorizer(wiki)), ("Guassian NB", GaussianNB())])
# wiki_wiki.fit(X=X_train, y=y_train)
# results_df.set_value("GB", "wiki_mean", accuracy_score(y_test, wiki_wiki.predict(X_test)))

# 2- TFIDF VECTORIZER
wiki_wiki_tfidf = Pipeline([("wiki tfidf vectorizer", TfidfVectorizer(wiki)), ("transform", TfidfTransformer()), ("Guassian NB", MultinomialNB())])
# wiki_wiki_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("GB", "wiki_tfidf", accuracy_score(y_test, wiki_wiki_tfidf.predict(X_test)))

####EXTRA TREE####
print("EXTRA TREE ...")
# 1- MEAN VECTORIZER
etree_wiki = Pipeline([("wiki mean vectorizer", MeanEmbeddingVectorizer(wiki)), ("extra trees", ExtraTreesClassifier(n_estimators=20))])
# etree_wiki.fit(X=X_train, y=y_train)
# results_df.set_value("ExtraTree", "wiki_mean", accuracy_score(y_test, etree_wiki.predict(X_test)))

# 2- TFIDF VECTORIZER
etree_wiki_tfidf = Pipeline([("wiki tfidf vectorizer", TfidfVectorizer(wiki)), ("transform", TfidfTransformer())
                             , ("extra trees", ExtraTreesClassifier(n_estimators=20))])
# etree_wiki_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("ExtraTree", "wiki_tfidf", accuracy_score(y_test, etree_wiki_tfidf.predict(X_test)))

####SVM####
print("SVM ... ")
#1- MAIN VECTORIZER
svm_wiki = Pipeline([("wiki mean vectorizer", MeanEmbeddingVectorizer(wiki)), ("SVM", LinearSVC(random_state=0, tol=1e-5))])
# svm_wiki.fit(X=X_train, y=y_train)
# results_df.set_value("SVM", "wiki_mean", accuracy_score(y_test, svm_wiki.predict(X_test)))

# 2- TFIDF VECTORIZER
svm_wiki_tfidf = Pipeline([("wiki tfidf vectorizer", TfidfVectorizer(wiki)), ("transform", TfidfTransformer()),
                           ("SVM", LinearSVC(random_state=0, tol=1e-5))])
# svm_wiki_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("SVM", "wiki_tfidf", accuracy_score(y_test, svm_wiki_tfidf.predict(X_test)))

####MLP####
print("MLP ... ")
# 1- MAIN VECTORIZER
mlp_wiki = Pipeline(
    [("wiki mean vectorizer", MeanEmbeddingVectorizer(wiki)),
     ("MLP", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))])
# mlp_wiki.fit(X=X_train, y=y_train)
# results_df.set_value("MLP", "wiki_mean", accuracy_score(y_test, mlp_wiki.predict(X_test)))

# 2- TFIDF VECTORIZER
mlp_wiki_tfidf = Pipeline(
    [("wiki tfidf vectorizer", TfidfVectorizer(wiki)), ("transform", TfidfTransformer()),
     ("MLP", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))])
# mlp_wiki_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("MLP", "wiki_tfidf", accuracy_score(y_test, mlp_wiki_tfidf.predict(X_test)))

####LREG####
print("LREG ... ")
# 1- MAIN VECTORIZER
lreg_wiki = Pipeline(
    [("wiki mean vectorizer", MeanEmbeddingVectorizer(wiki)),
     ("LREG", LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42))])
# lreg_wiki.fit(X=X_train, y=y_train)
# results_df.set_value("LREG", "wiki_mean", accuracy_score(y_test, lreg_wiki.predict(X_test)))

# 2- TFIDF VECTORIZER
lreg_wiki_tfidf = Pipeline(
    [("wiki tfidf vectorizer", TfidfVectorizer(wiki)), ("transform", TfidfTransformer()),
     ("LREG", LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=42))])
# lreg_wiki_tfidf.fit(X=X_train, y=y_train)
# results_df.set_value("LREG", "wiki_tfidf", accuracy_score(y_test, lreg_wiki_tfidf.predict(X_test)))
results_df.set_value("NB","wiki_mean",0.939035)
results_df.set_value("ExtraTree","wiki_mean",0.939249)
results_df.set_value("SVM","wiki_mean",0.925018)
results_df.set_value("MLP","wiki_mean",0.074982)
results_df.set_value("LREG","wiki_mean",0.939035)

results_df.set_value("NB","wiki_tfidf",0.941720)
results_df.set_value("ExtraTree","wiki_tfidf",0.944973)
results_df.set_value("SVM","wiki_tfidf",0.953854)
results_df.set_value("MLP","wiki_tfidf",0.950935)
results_df.set_value("LREG","wiki_tfidf",0.953311)
results_df
del wiki
del model
import gc; gc.collect()
time.sleep(10)
fig,axes=plt.subplots(1,1,figsize=(15,8))
plt.ylim((.5,1))
axes.set_ylabel("Accuracy")
axes.set_title("Traditional Classfieris Results for 67% Training and 23% Testing with Two Types of Embedding")
results_df[results_df.index != "RF"].plot(kind="bar",ax=axes)
for name in dir():
    if not name.startswith('_'):
        del globals()[name]
for name in dir():
    if not name.startswith('_'):
        del locals()[name]
import gc; gc.collect()
import pandas as pd
from keras import Sequential
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.layers import Bidirectional, GlobalMaxPool1D,Dense, Input, Embedding, Dropout, LSTM, CuDNNGRU
from keras import Sequential
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
train_df = pd.DataFrame.from_csv("../input/train.csv")
X_train, X_test, y_train, y_test = train_test_split(train_df.question_text, train_df.target, test_size=0.33, random_state=42)
num_words = 5000
maxlen = 100
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_train[0]
len(X_train[0]),len(X_train[10])
max_len = 150
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
len(X_train[0]),len(X_train[10])
embedding_size = 300
model = Sequential()
model.add(Embedding(num_words, embedding_size))
## Bidirectional wrapper for RNNs. It involves duplicating the first recurrent
## layer in the network so that there are now two layers side-by-side, then 
## providing the input sequence as-is as input to the first layer and providing
## a reversed copy of the input sequence to the second.
model.add(Bidirectional(CuDNNGRU(64, return_sequences=True))) 
model.add(GlobalMaxPool1D())
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x=X_train, y=y_train, batch_size=512, epochs=1, validation_data=(X_test, y_test))