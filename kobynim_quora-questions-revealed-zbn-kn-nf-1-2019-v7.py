################# GENERAL IMPORTS
import os
import string
from pprint import pprint
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
##################################### NLP SPECIFIC IMPORTS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.stem import PorterStemmer
from nltk.corpus import reuters
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from collections import Counter
reuters.fileids()
stopwords.words('english')

import numpy as np
import pandas as pd
import warnings
from sys import modules

warnings.filterwarnings('ignore')
%matplotlib inline

from gensim.models import word2vec
import logging

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Dropout

import seaborn as sns
sns.set(style = 'darkgrid')
print(os.listdir("../input"))
import re
pd.set_option('max_colwidth', 800)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

np.random.seed(1234)
%matplotlib inline

print('all set')
quora_train=pd.read_csv("../input/train.csv")
quora_test=pd.read_csv("../input/test.csv")
print("Train size =" ,quora_train.shape)
print("Test size =" ,quora_test.shape)
# quora_train=quora_train[0:5000]
# quora_test=quora_test[0:1000]
quora_train.head()
quora_train.info()
sincere = quora_train[quora_train.target==0]
insincere = quora_train[quora_train.target==1]
[print(q,'\n') for q in sincere['question_text'][[1,6,40,300,120]]]
quora_train['target'].value_counts().plot(kind='bar', title='Target distribution')
round(quora_train['target'].value_counts(normalize =True),3)*100
quora_train['words'] = quora_train.question_text.apply(lambda x: len(x.split()))
quora_train['characters'] = quora_train.question_text.apply(lambda x: len(x))
quora_test['words'] = quora_test.question_text.apply(lambda x: len(x.split()))
quora_test['characters'] = quora_test.question_text.apply(lambda x: len(x))
quora_train.head()
fig = plt.figure(figsize=(18, 7))

plt.subplot(1, 2, 1)
quora_train.groupby('target')['words'].mean().plot(kind='bar', ylim=(0,20), title= 'Average word count by target')

plt.subplot(1, 2, 2)
quora_train.groupby('target')['characters'].mean().plot(kind='bar', ylim=(0,105), title= 'Average character count by target')
fig = plt.figure(figsize=(18, 8))
font = {'size': 16, 'weight': 'bold'}

plt.subplot(2, 1, 1)
plt.title('Word count - by outcome',fontdict=font)
plt.xlim(0,70)
ax = sns.boxplot(x="words", y="target", data=quora_train, orient="h")

plt.subplot(2, 1, 2)
plt.title('character count - by outcome',fontdict=font)
plt.xlim(0,350)
ax = sns.boxplot(x="characters", y="target", data=quora_train, orient="h")
from nltk import pos_tag

def verb_count(text):
    token_text= word_tokenize(text)
    tagged_text = pos_tag(token_text)
    counter=0
    for w,t in tagged_text:
        t = t[:2]
        if t in ['VB']:
            counter+=1
    return counter

def noun_count(text):
    token_text= word_tokenize(text)
    tagged_text = pos_tag(token_text)
    counter=0
    for w,t in tagged_text:
        t = t[:2]
        if t in ['NN']:
            counter+=1
    return counter

quora_train.head()
quora_train['question_text_prep'] = quora_train['question_text'].apply(lambda x: x.lower())
quora_test['question_text_prep'] = quora_test['question_text'].apply(lambda x: x.lower())
def pad_punctuation_w_space(string):
    s = re.sub('([:;"*.,!?()/\=-])', r' \1 ', string)
    s=re.sub('[^a-zA-Z]',' ',s)
    s = re.sub('\s{2,}', ' ', s)
    s =  re.sub(r"\b[a-zA-Z]\b", "", s) #code for removing single characters
    return s
quora_train['question_text_prep'] = quora_train['question_text_prep'].apply(lambda x: pad_punctuation_w_space(x))
quora_test['question_text_prep'] = quora_test['question_text_prep'].apply(lambda x: pad_punctuation_w_space(x))

quora_train['question_text_prep'] = quora_train['question_text_prep'].apply(lambda x: x.split())
quora_test['question_text_prep'] = quora_test['question_text_prep'].apply(lambda x: x.split())
stop_list = stopwords.words('english') + list(string.punctuation)
quora_train['question_text_prep'] = quora_train['question_text_prep'].apply(lambda x: [i for i in x if i not in stop_list])
quora_test['question_text_prep'] = quora_test['question_text_prep'].apply(lambda x: [i for i in x if i not in stop_list]) 

quora_train['question_text_prep_string'] = quora_train['question_text_prep'].str.join(" ")
quora_test['question_text_prep_string'] = quora_test['question_text_prep'].str.join(" ")
# quora_insincere =  quora_train[quora_train.target==1]
# quora_sincere =  quora_train[quora_train.target==0]

# # make "all in one" corpuses for the 2 classes in the target
# insincere_all_in_one = ' '.join([q for q in quora_insincere.question_text_prep_string])
# sincere_all_in_one = ' '.join([q for q in quora_sincere.question_text_prep_string]) 
# fig = plt.figure(figsize=(30, 12))
# font = {'size': 20, 'weight': 'bold'}

# plt.subplot(1, 2, 1)
# plt.title('Insincere questions',fontdict=font)
# cloud1 = WordCloud(max_words=100,width=480, height=480, background_color='grey')
# cloud1.generate_from_text(insincere_all_in_one)
# plt.imshow(cloud1)
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title('Sincere questions',fontdict=font)
# cloud = WordCloud(max_words=100,width=480, height=480, background_color='skyblue')
# cloud.generate_from_text(sincere_all_in_one)
# plt.imshow(cloud)
# plt.axis('off')
# insincere_tokens = [t for t in word_tokenize(insincere_all_in_one)]
# sincere_tokens = [t for t in word_tokenize(sincere_all_in_one)]
# from collections import Counter
# from nltk import ngrams
# bi_gram_insincere = Counter(ngrams(insincere_tokens, 2))
# tri_gram_insincere = Counter(ngrams(insincere_tokens, 3))
# bi_gram_sincere = Counter(ngrams(sincere_tokens, 2))
# tri_gram_sincere = Counter(ngrams(sincere_tokens, 3))
# bi_gram_ins = pd.DataFrame(bi_gram_insincere.most_common(20), columns=['bi_gram_ins','frequency'])
# bi_gram_sin = pd.DataFrame(bi_gram_sincere.most_common(20), columns=['bi_gram_sin','frequency'])
# tri_gram_ins = pd.DataFrame(tri_gram_insincere.most_common(20), columns=['tri_gram_ins','frequency'])
# tri_gram_sin = pd.DataFrame(tri_gram_sincere.most_common(20), columns=['tri_gram_sin','frequency'])
# import matplotlib.pyplot as plt
# fig, (ax, ax2) = plt.subplots(ncols=2, sharex=False)
# fig.subplots_adjust(wspace =0.6)
# ax.invert_xaxis()

# bi_gram_ins.sort_values(by='frequency').plot(kind='barh', x='bi_gram_ins', legend=True, ax=ax, figsize=(18,9), fontsize =16)
# bi_gram_sin.sort_values(by='frequency').plot(kind='barh', x='bi_gram_sin',ax=ax2, figsize=(18,9),fontsize =16)
# plt.show()
# fig, (ax, ax2) = plt.subplots(ncols=2, sharex=False)
# fig.subplots_adjust(wspace =0.8)
# ax.invert_xaxis()

# tri_gram_ins.sort_values(by='frequency').plot(kind='barh', x='tri_gram_ins', legend=True, ax=ax, figsize=(18,9), fontsize =16)
# tri_gram_sin.sort_values(by='frequency').plot(kind='barh', x='tri_gram_sin',ax=ax2, figsize=(18,9), fontsize =16)
# plt.show()
sents = list(quora_train.question_text_prep.values) 
sents[0]
min_num = 3 # minimum number of occurrences in text
EMBEDDING_FILE= "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
import numpy as np
def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r', encoding='utf8')
    model = {}
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model
word_model= loadGloveModel(EMBEDDING_FILE)   
# print (word_model['hello']) # if we want to see an example for a vector
print('Loaded %s word vectors.' % len(word_model))
unknown_words = []
for question in quora_train.question_text_prep:
    for word in question:
        if word not in word_model:
            unknown_words.append(word)
        else: pass
len(unknown_words)
unknown_words[:10]
total_term_frequency = Counter(unknown_words)

for word, freq in total_term_frequency.most_common(20):
    print("{}\t{}".format(word, freq))
def get_vector(DataFrame):
    vec_X = []
    i = 0
    for item in DataFrame.question_text_prep_string: 
        
        sentence = pad_punctuation_w_space(item)
        s = np.array([])
        s = []
        if len(sentence)==0:
            s = np.array(word_model['UNK'])
            vec_X.append(s) 
            i += 1
        else:
                for word in sentence.split():
                    if len(s) == 0:
                        try:
                            s = np.array(word_model[word])
                        except: 
                            s = np.array(word_model['UNK'])
                    else:
                        try:
                            s += np.array(word_model[word])
                        except: 
                            s += np.array(word_model['UNK'])         
                vec_X.append(s) 
                i += 1

    return vec_X
vec_X_train=get_vector(quora_train)
vec_X_test=get_vector(quora_test)
quora_train["vector"]=vec_X_train
quora_test["vector"]=vec_X_test

from imblearn.under_sampling import RandomUnderSampler
X = quora_train[['words','characters','vector']] #,'noun_count'
y = quora_train['target']
rus = RandomUnderSampler(return_indices=True, ratio = 0.42)
X_rus, y_rus, id_rus = rus.fit_sample(X, y)

print('indexes:', id_rus)
print(len(id_rus))
print(quora_train.target.value_counts())
quora_undr=quora_train.loc[id_rus]
quora_undr['target'].value_counts(ascending=True).plot(kind='bar')
quora_undr['target'].value_counts(normalize=True)
quora_under_prep = quora_undr
quora_under_prep["characters"].head()
quora_under_prep['noun_count'] = quora_under_prep.question_text.apply(lambda x: noun_count(x))
quora_test['noun_count'] = quora_test.question_text.apply(lambda x: noun_count(x))
quora_under_prep['vector_length']= quora_under_prep['vector'].apply(lambda x: len(x))
quora_test['vector_length']= quora_test['vector'].apply(lambda x: len(x))
quora_test['vector_length'].describe()
quora_best=quora_under_prep
import numpy as np
quora_best["joinvector"]=[np.concatenate((np.array([quora_best["characters"].iloc[i]]),quora_best["vector"].iloc[i]), axis=None) for i in range(len(quora_best))]
quora_best["joinvector_2"]=[np.concatenate((np.array([quora_best["words"].iloc[i]]),quora_best["joinvector"].iloc[i]), axis=None) for i in range(len(quora_best))]
quora_best["joinvector_all"]=[np.concatenate((np.array([quora_best["noun_count"].iloc[i]]),quora_best["joinvector_2"].iloc[i]), axis=None) for i in range(len(quora_best))]
quora_test["joinvector"]=[np.concatenate((np.array([quora_test["characters"].iloc[i]]),quora_test["vector"].iloc[i]), axis=None) for i in range(len(quora_test))]
quora_test["joinvector_2"]=[np.concatenate((np.array([quora_test["words"].iloc[i]]),quora_test["joinvector"].iloc[i]), axis=None) for i in range(len(quora_test))]
quora_test["joinvector_all"]=[np.concatenate((np.array([quora_test["noun_count"].iloc[i]]),quora_test["joinvector_2"].iloc[i]), axis=None) for i in range(len(quora_test))]
# quora_test.head(1)
X_joinvec=quora_best["joinvector_all"].tolist()
# from sklearn.cluster import DBSCAN
# fit_model_joinvec=DBSCAN(eps=0.25,min_samples=10).fit(X_joinvec)
# fitted_model_joinvec=fit_model_joinvec.labels_
# quora_best['DBSCAN_Cluster_joinvec']=fitted_model_joinvec

# quora_best.head()
#quora_best['DBSCAN_Cluster_joinvec'].value_counts()
# from sklearn.cluster import KMeans, MiniBatchKMeans
# def calc_inertia(k):
#         model_kmeans= KMeans(n_clusters=k, init='k-means++',verbose=1,random_state=42).fit(X_joinvec)
#         return model_kmeans.inertia_,model_kmeans.labels_

# inertias,labels_kmeans = [(k, calc_inertia(k)) for k in range(1, 21)]
# from sklearn.cluster import KMeans, MiniBatchKMeans
# n_clusters=50
# model_MiniBatch = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=1,
#                          init_size=500,
#                          batch_size=500, verbose=1)

# print ("Clustering sparse data with %s" % model_MiniBatch)
# model_MiniBatch.fit(X_joinvec)
# labels_MiniBatch = model_MiniBatch.labels_
# print("done")
# # print("labels", labels_MiniBatch)
# # print("intertia:", model_MiniBatch.inertia_)
# quora_best['MiniBatch_Cluster_joinvec']=labels_MiniBatch
# # quora_best.head()
# quora_best['MiniBatch_Cluster_joinvec'].value_counts().head(10)
# import matplotlib.pylab as plt
# prob_groups = quora_best.groupby("MiniBatch_Cluster_joinvec").MiniBatch_Cluster_joinvec.count()
# prob_groups.plot(kind='hist')
# plt.figure(figsize=(30,100))
# plt.show()
# pd.set_option('max_colwidth', 800)
# cluster_columns=quora_best[["question_text","target","MiniBatch_Cluster_joinvec"]]
# random_cluster=cluster_columns[cluster_columns["MiniBatch_Cluster_joinvec"]==38].sort_values(by='target',ascending=False)
# random_cluster.head(10)
# random_cluster.target.value_counts(normalize=True).plot(kind='bar', title='target', figsize=(10,5))
# plt.show()
Features = quora_best['joinvector_all']
# Features2=quora_best['vector']
# FF=Features.tolist()
# Features.shape
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


X_train, X_val, y_train, y_val = train_test_split(Features,quora_best['target'],
                                                    train_size=0.7, random_state = 143, stratify=quora_best['target'])
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

#evaluators:
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
# # Lr_clf = LogisticRegression()
# # X =  X_train.tolist()
# # y = y_train

# # Lr_clf.fit(X,y)
# GB_clf=GradientBoostingClassifier()
# X =  X_train.tolist()
# y = y_train

# GB_clf.fit(X,y)
# f1 = cross_val_score(GB_clf, X, y, scoring='f1', cv=5)
# accuracy = cross_val_score(GB_clf, X, y, cv=5, scoring='accuracy')
print('f1 score:{}\nacurracy: {}'.format(round(f1.mean(),2),round(accuracy.mean(),2)))
#weights = [2,3]
# param_grid = {'C': [0.5,10],
#               'class_weight':[{0:1, 1:w} for w in weights]}
# param_grid ={"learning_rate":(0.1,0.5),
#                             'max_depth' : range(2,5,10),
#                             'min_samples_split': range(2,5,10),
#                             'min_samples_leaf' : range(2,5,10),
#                             #'max_features':range(4,8),
#                             "subsample":(0.5,0.8)} 
param_grid ={"learning_rate":(0.5),
                            'max_depth' : range(2),
                            'min_samples_split': range(2),
                            'min_samples_leaf' : range(2),
                            #'max_features':range(4,8),
                            "subsample":(0.8)} 
X_grid5 = X_train.tolist()
y_grid5 = y_train
#Lr_clf = LogisticRegression()
GB_clf=GradientBoostingClassifier()
# gs= GridSearchCV(estimator=Lr_clf, param_grid=param_grid, cv=2,scoring='f1') # verbose=15, n_jobs=-1
gs= GridSearchCV(estimator=GB_clf, param_grid=param_grid, cv=2,scoring='f1') # verbose=15, n_jobs=-1
gs.fit(X_grid5, y_grid5)
best_model=gs.best_estimator_ 
best_model
y_pred= best_model.predict(X_grid5)
f1 = cross_val_score(best_model, X_grid5, y_grid5, scoring='f1', cv=2)
accuracy = cross_val_score(best_model, X_grid5, y_grid5, cv=2, scoring='accuracy')
cm = confusion_matrix(y_true=y_grid5, y_pred=y_pred)

print('f1-score:',round(f1.mean(),2),'\naccuracy:',round(accuracy.mean(),2),'\n______________\n')
cm = cm
print (classification_report(y_grid5, y_pred))

print('confusion matrix:')
pd.DataFrame(cm, 
             index=best_model.classes_, 
             columns=best_model.classes_)
X1_val=X_val.tolist()
y1_val=y_val
best_model.fit(X1_val,y1_val)
f1 = cross_val_score(best_model, X1_val,y1_val, scoring='f1', cv=3)
print('f1 score of "val":', round(f1.mean(),2))
# X_NN =  X_train.tolist()
# y_NN = y_train

# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense, Embedding
# from keras.layers import LSTM
# max_features=50000
# MAX_SEQUENCE_LENGTH=303
from keras import backend as K
# def f1(y_true, y_pred):
#     '''
#     metric from here 
#     https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
#     '''
#     def recall(y_true, y_pred):
#         """Recall metric.

#         Only computes a batch-wise average of recall.

#         Computes the recall, a metric for multi-label classification of
#         how many relevant items are selected.
#         """
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#         recall = true_positives / (possible_positives + K.epsilon())
#         return recall

#     def precision(y_true, y_pred):
#         """Precision metric.

#         Only computes a batch-wise average of precision.

#         Computes the precision, a metric for multi-label classification of
#         how many selected items are relevant.
#         """
#         true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#         predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + K.epsilon())
#         return precision
#     precision = precision(y_true, y_pred)
#     recall = recall(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
# TT=np.array(FF)
# # num_words = min(max_features, len(TT))
# num_words=len(TT)
# from keras.preprocessing.text import Tokenizer

# t = Tokenizer()
# t.fit_on_texts(quora_best["question_text_prep"])

# vocab_size = len(t.word_index) + 1
# #vocab_size

# encoded_docs = t.texts_to_sequences(quora_best["question_text_prep"])
# from numpy import zeros
# embedding_matrix = zeros((vocab_size, 300))
# for word, i in t.word_index.items():
#     embedding_vector = word_model.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector
# from keras.preprocessing.sequence import pad_sequences
# max_length = 300
# padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# # print(padded_docs)
# embedding_matrix.shape
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


# X_train, X_val, y_train, y_val = train_test_split(padded_docs,quora_best['target'],
#                                                     train_size=0.7, random_state = 143, stratify=quora_best['target'])
# model = Sequential()
# #model.add(Embedding(num_words,input_length=MAX_SEQUENCE_LENGTH,weights=[TT],trainable=False,output_dim=MAX_SEQUENCE_LENGTH))

# model.add(Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_length, trainable=False))
# model.add(Bidirectional(LSTM(128,dropout=0.2, recurrent_dropout=0.2, return_sequences=True))) #
# model.add(Bidirectional(LSTM(64,return_sequences=True))) # dropout=0.2, recurrent_dropout=0.2,
# #model.add(Attention(MAX_SEQUENCE_LENGTH))
# model.add(GlobalMaxPool1D())
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=[f1])

# print(model.summary())
# batch_size = 512 #32
# X_train_list=X_train.tolist()
# X_val_list=X_val.tolist()
# observation_train = np.asarray(X_train_list)
# observation_val = np.asarray(X_val_list)
# #weight={0:0.4,1:1}
# weight={0:1,1:2}
# model.fit(X_train, y_train,
#           batch_size=batch_size,
#           epochs=2,
#           validation_data=(X_val, y_val),
#           class_weight=weight)

# model.fit(observation_train, y_train,
#           batch_size=batch_size,
#           epochs=10,
#           validation_data=(observation_val, y_val),
#           class_weight=weight)

# score, f1_calc = model.evaluate(X_val, y_val,
#                             batch_size=batch_size)
# print('Test score:', score)
# print('Test f1:', f1_calc)
# #X_train, X_val, y_train, y_val
# y_pred_train = model.predict(X_train)
# y_pred_val = model.predict(observation_val)
from tqdm.auto import tqdm
# def bestThresshold(y_train,y_pred_train):
#     tmp = [0,0,0] # idx, cur, max
#     delta = 0
#     for tmp[0] in tqdm(np.arange(0.1, 0.501, 0.01)):
#         tmp[1] = f1_score(y_train, np.array(y_pred_train)>tmp[0])
#         if tmp[1] > tmp[2]:
#             delta = tmp[0]
#             tmp[2] = tmp[1]
#     print('best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, tmp[2]))
#     return delta
# delta = bestThresshold(y_train,y_pred_train)
# t_test = Tokenizer()
# t_test.fit_on_texts(quora_test["question_text_prep"])

# #vocab_size = len(t.word_index) + 1
# #vocab_size

# encoded_docs_test = t_test.texts_to_sequences(quora_test["question_text_prep"])
# max_length = 300
# padded_docs_test = pad_sequences(encoded_docs_test, maxlen=max_length, padding='post')
# #print(padded_docs)
# # test_Features = quora_test['vector'] #quora_test['joinvector_all']
# # X_test_original=test_Features.tolist()
# # observation_test = np.asarray(X_test_original)
# # y_test_pred= model.predict(observation_test)
# y_test_pred= model.predict(padded_docs_test)
test_Features = quora_test['joinvector_all']
X_test_original=test_Features.tolist()
y_test_pred= best_model.predict(X_test_original)
# y_test_pred
quora_test_tmp=quora_test
# delta=0.4
quora_test_tmp["pred"]=y_test_pred #(y_test_pred > delta).astype(int) 
quora_test_tmp1 = quora_test_tmp[['qid','question_text','pred']]
quora_test_tmp1[quora_test_tmp1['pred']==1].sample(10)

sub = pd.read_csv('../input/sample_submission.csv')
out_df = pd.DataFrame({"qid":sub["qid"].values})
out_df['prediction'] = y_test_pred
out_df.to_csv("submission.csv", index=False)
# sub = pd.read_csv('../input/sample_submission.csv')
# out_df = pd.DataFrame({"qid":sub["qid"].values})
# out_df['prediction'] =y_test_pred# (y_test_pred > delta).astype(int) #y_test_pred
# out_df.to_csv("submission.csv", index=False)
round(out_df['prediction'].value_counts(normalize =True),3)*100
