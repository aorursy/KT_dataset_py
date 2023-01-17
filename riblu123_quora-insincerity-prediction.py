# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from subprocess import check_output
train_data = pd.read_csv('../input/train.csv')
train_1 = train_data[train_data.target ==1]
train_0 = train_data[train_data.target ==0]


def split_dataframe_to_chunks(df, n):
    df_len = len(df)
    n_rows = df_len//n
    n_rem = df_len%n
    count_ = 0
    dfs = []
    for count in range(n):
        if count<n-1:
            start = count_
            count_ += n_rows
            #print("%s : %s" % (start, count))
            dfs.append(df.iloc[start : count_])
        else:
            start = count_
            count_ += n_rows+n_rem
            dfs.append(df.iloc[start : count_])
    return dfs
train_0_splits = split_dataframe_to_chunks(train_0,5)
train_0_splits[0].target.value_counts()
80810/245062
for i in range(len(train_0_splits)):
    train_0_splits[i] = train_0_splits[i].append(train_1,ignore_index=True)
    train_0_splits[i] = train_0_splits[i].sample(frac=1).reset_index(drop=True)
       
train_data_list = train_0_splits
del train_0_splits
#import libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
from sklearn.decomposition import SparsePCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from nltk.stem.wordnet import WordNetLemmatizer, wordnet
from nltk.tokenize import TweetTokenizer, sent_tokenize
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.multiclass import OneVsRestClassifier
from imblearn.under_sampling import RandomUnderSampler
import time
from sklearn.metrics import cohen_kappa_score
wordnet_lemmatizer = WordNetLemmatizer()
# stop_words = stopwords.words('english')
tokenizer_words = TweetTokenizer()

#clean data
def Clean_data(raw_text):
    #remove HTML
#     data = BeautifulSoup(raw_text, 'lxml').get_text()
    data = re.sub(r'http\S+|www\S+', '', raw_text)
    sentences = sent_tokenize(data)
  #  sentences = sentences[2:-1]
    clean_data = []
    for sent in sentences:
        sent = re.sub("[^\w]", " ", sent)
        sent = re.sub(r"\d+", " ", sent)
        sent = re.sub("_", " ", sent)
        # sent = re.sub(r"https|http", "", sent)
        sent = ' '.join([t.lower() for t in sent.split(' ') if t])
        sent = ' '.join( [w for w in sent.split() if len(w)>2] )
        # sent = ' '.join([w for w in sent.split(' ') if w not in ext_stop])
        sent = ' '.join([wordnet_lemmatizer.lemmatize(t, wordnet.VERB) for t in sent.split()])
#         sent = ' '.join([w for w in sent.split(' ') if w not in stop_words])
        clean_data.append(sent)
    clean_data = '. '.join(clean_data)
    return clean_data
X_ = train_data.question_text
X_ = [Clean_data(d) for d in X_]
train_data['question_text_clean_with_stop'] = X_
# train_data.to_csv('../input/train_data_new.csv', index = False)
X_ = train_data.question_text_clean_with_stop
Y = train_data.target
# Y = [1 if (y==0) else 0 for y in Y]
# le = preprocessing.LabelEncoder()
# Y= le.fit_transform(Y)
test_data = pd.read_csv('../input/test.csv')
test_X = test_data.question_text
test_X = [Clean_data(d) for d in test_X]
test_data['question_text_clean_with_stop'] = test_X
test_X = test_data.question_text_clean_with_stop
# Y = train_data.target
# rus = RandomUnderSampler()
# X, Y = rus.fit_sample(np.array(X).reshape(-1, 1),Y)
# X = pd.Series([x[0] for x in X])
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
VOCAB_SIZE = 13500
counter = CountVectorizer(max_features=VOCAB_SIZE)
Xc = counter.fit_transform(X_)
tfidf_vect = TfidfTransformer()
X_ = tfidf_vect.fit_transform(Xc)
# clf = LogisticRegression(solver='lbfgs',max_iter=500, multi_class='multinomial')
# sent_lens =np.sum(Xc, axis=1).astype("float")
# sent_lens[sent_lens == 0] = 1e-14
# print(sent_lens.shape)
# sent_ = np.divide(1,sent_lens)
# Xc = Xc.multiply(sparse.csr_matrix(sent_))
# del X

'''
freqs = np.sum(Xc, axis=0).astype("float")
probs = freqs / np.sum(freqs)
ALPHA = 1e-3
coeff = ALPHA /(ALPHA + probs)
Xw = Xc.multiply(sparse.csr_matrix(coeff))
# Xw = np.multiply(Xc, coeff)
del Xc
'''
'''
import os
GLOVE_EMBEDDINGS = '/kaggle/input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
E = np.zeros((VOCAB_SIZE, 300))
fglove = open(GLOVE_EMBEDDINGS, "r")
for line in fglove:
    cols = line.strip().split(" ")
    word = cols[0]
    try:
        i = counter.vocabulary_[word]
        E[i] = np.array([float(x) for x in cols[1:]])
    except KeyError:
        pass
fglove.close()
'''
# E.shape
'''
# # Xc = Xc.toarray()
# # compute word probabilities from corpus
Xs =Xw.dot(sparse.csr_matrix(E))
# # Xs = np.divide(np.dot(Xw, E), sent_lens)
del Xw, E
# from sklearn.decomposition import TruncatedSVD

# svd = TruncatedSVD(n_components=1, n_iter=20, random_state=0)
# svd.fit(Xs)
# # svd.fit(X)
# pc = svd.components_
# pc_t = pc.T
# pc = sparse.csr_matrix(pc)
# pc_t = sparse.csr_matrix(pc_t)
# print(type(pc))
# Xr = Xs - Xs.dot(pc.T).dot(pc) 
# # X_pc = (pc_t).dot(pc)
# # del pc
# # Xr_ = X.dot(X_pc)
# # Xr = X - Xr_
# del  X,X_pc #Xs,
'''
'''
pc = sparse.csr_matrix(pc)
Xr = Xs - Xs.dot(pc.T).dot(pc)
del Xs, pc
'''
'''
skf = StratifiedKFold(n_splits = 10, shuffle = True)
# X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer(ngram_range = (1,3),max_features=13500)),
    ('tfidf_transformer',  TfidfTransformer()),
#     ('SVD', TruncatedSVD(n_components=1, n_iter=10)),
#     ('PCA', SparsePCA(n_components=1,normalize_components=True)),
#     ('qda', QuadraticDiscriminantAnalysis(store_covariances=True)),
#    ('classifier',  svm.SVC())]) #solver='lbfgs',class_weight={0:0.6,1:1.5},max_iter=500, multi_class='multinomial'))])
    ('classifier', LogisticRegression(solver='lbfgs',max_iter=500, multi_class='multinomial'))])
#     ('classifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=10, random_state=0))])
'''
'''
accuracy2 = []
mcc_value2 =[]
f1_scores = []
test_Y = []
predicted = []
cm =[]
model = None
tfidf_model = None
cv_model = None
start_time = time.time()
for train_ix, test_ix in skf.split(X,Y):

    X_train, X_test = X[train_ix], X[test_ix]
    Y_train, Y_test = Y[train_ix], Y[test_ix]
   # CV.fit_transform(X_train.values,Y_train)
#     pipeline.fit(X_train,Y_train)
    pipeline.fit(X_train.values.astype('U'),Y_train)
#     cv_model_ =  pipeline.steps[0][1]
#     tfidf_model_ =  pipeline.steps[1][1]
#     PCA_model_ = pipeline.steps[2][1]
#     model_ =  pipeline.steps[3][1]
    prediction = pipeline.predict(X_test.values.astype('U'))
    ac = np.mean(prediction == Y_test)
    mcc = matthews_corrcoef(Y_test, prediction)
    f1 = f1_score(Y_test, prediction)
#     if not accuracy2:
#         model = model_
#         cv_model = cv_model_
#         tfidf_model = tfidf_model_
# #         PCA_model = PCA_model_
#     elif ac>max(accuracy2) and f1>max(f1_scores):
#         model = model_
#         tfidf_model = tfidf_model_
#         cv_model = cv_model_
#         PCA_model = PCA_model_
    test_Y.append(Y_test)
    predicted.append(prediction)
    accuracy2.append(ac)
    mcc_value2.append(mcc)
    f1_scores.append(f1)
    print(ac,mcc,f1)
    cm.append(confusion_matrix(Y_test, prediction))
'''
# pipelines = {}
predicted = []
for i in range(len(train_data_list)):
    data = train_data_list[i]
    X = data.question_text
    X = [Clean_data(d) for d in X]
    data['question_text_clean_with_stop'] = X
    # train_data.to_csv('../input/train_data_new.csv', index = False)
    X = data.question_text_clean_with_stop
    Y = data.target
    accuracy2 = []
    mcc_value2 =[]
    f1_scores = []
#     test_Y = []
#     predicted = []
    start_time = time.time()
#     X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)
    # CV.fit_transform(X_train.values,Y_train)
    #     pipeline.fit(X_train,Y_train)
    X = counter.transform(X)
    X = tfidf_vect.transform(X)
    clf = LogisticRegression(solver='lbfgs',max_iter=450,class_weight={0:0.8,1:1}, multi_class='multinomial')
    clf.fit(X,Y)
    #     cv_model_ =  pipeline.steps[0][1]
    #     tfidf_model_ =  pipeline.steps[1][1]
    #     PCA_model_ = pipeline.steps[2][1]
    #     model_ =  pipeline.steps[3][1]
#     prediction = pipeline.predict(X_test.values.astype('U'))
#     ac = np.mean(prediction == Y_test)
#     mcc = matthews_corrcoef(Y_test, prediction)
#     f1 = f1_score(Y_test, prediction)
    #     if not accuracy2:
    #         model = model_
    #         cv_model = cv_model_
    #         tfidf_model = tfidf_model_
    # #         PCA_model = PCA_model_
    #     elif ac>max(accuracy2) and f1>max(f1_scores):
    #         model = model_
    #         tfidf_model = tfidf_model_
    #         cv_model = cv_model_
    #         PCA_model = PCA_model_
#     model =  pipeline.steps[2][1]
#     tfidf_model =  pipeline.steps[1][1]
#     cv_model =  pipeline.steps[0][1]
#     test_Y.append(Y_test)
#     predicted.append(prediction)
#     accuracy2.append(ac)
#     mcc_value2.append(mcc)
#     f1_scores.append(f1)
#     print(ac,mcc,f1)
#     cm.append(confusion_matrix(Y_test, prediction))
    test_x = counter.transform(test_X)
    test_x = tfidf_vect.transform(test_x)
    predicted_= clf.predict(test_x)
    predicted.append(predicted_)
#     pipelines[i]=pipeline
from collections import Counter
prediction = []
for i in range(len(predicted[0])):
    prediction.append([predicted[j][i] for j in range(len(predicted))])
# max_key = max(stats, key=lambda k: stats[k])
prediction_ = [max(x, key=lambda k: x[k]) for x in prediction]
# model =  pipeline.steps[2][1]
# tfidf_model =  pipeline.steps[1][1]
# cv_model =  pipeline.steps[0][1]
# # for i in model.classes_:
# feat_ind = np.argsort(model.coef_[0])[::-1][:1000]
# feat_ = [cv_model.get_feature_names()[idx] for idx in feat_ind]
# important_features[1]= feat_
# del model, tfidf_model, cv_model
# import json
# with open('important_features.json', 'wb') as f:
#     f.write(important_features)
test_data.shape
import pandas as pd
# prediction = pipeline.predict(test_X.values.astype('U'))
sub = pd.DataFrame()
sub['qid'] = test_data.qid
sub['prediction'] = prediction_
sub.to_csv('submission.csv', index = False)
