!pip install swifter
!pip install wordninja
!pip install pandarallel
import logging
##_______________________________________________________________________________________________________________________________________


## pandas, numpy
import pandas as pd
import numpy as np
from numpy import random
from tqdm import tqdm
tqdm.pandas()
# import swifter
##_______________________________________________________________________________________________________________________________________


## gensim
import gensim
from gensim.models import KeyedVectors,Word2Vec
##_______________________________________________________________________________________________________________________________________


## sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import f1_score,precision_score,recall_score
##_______________________________________________________________________________________________________________________________________


## nltk, nlp, spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.chunk import conlltags2tree
from nltk.tree import Tree
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
# import wordninja
import re
from textblob import TextBlob
import spacy
from spacy import displacy
import en_core_web_sm
##_______________________________________________________________________________________________________________________________________


## plotting
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
##_______________________________________________________________________________________________________________________________________


## imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
##_______________________________________________________________________________________________________________________________________


## tensorflow, keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Convolution1D,GlobalMaxPool1D,GlobalMaxPooling1D,Attention,Dropout,Dense,Embedding,Bidirectional,GRU,GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
##_______________________________________________________________________________________________________________________________________


import pickle
import heapq
from collections import OrderedDict
from collections import Counter
from multiprocessing import Pool
from string import punctuation
## basic cleaning

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
#     text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text
    
# df['consumer_complaint_narrative'] = df.consumer_complaint_narrative.astype(str)
# df['clean_narrative'] = df['consumer_complaint_narrative'].swifter.apply(clean_text)
## advanced cleaning
stop_words = set(stopwords.words('english'))
lem = WordNetLemmatizer()

def removal(text):
    text = re.sub('XXXX',' UNKNOWN ',text)
    text = re.sub('XX/XX/','',text)
    text = re.sub('UNKNOWN   UNKNOWN','UNKNOWN',text)
    text = re.sub('\n',' ',text)
    text = re.sub('  ',' ',text)
    return text

def cleaning(text):
    text = removal(text)
    #text = text.lower()
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    #words = [w for w in words if len(w)>2]
    words = [lem.lemmatize(w,'v') for w in words]
    return ' '.join(words)

def total_clean(text):
    
    text = text.lower()
    text = re.sub('[^A-Za-z0-9]',' ',text)
    text = cleaning(text)
    text = wordninja.split(text)
    text = ' '.join(text)
#     text = remove3ConsecutiveDuplicates(text)
    #blob = TextBlob(text)
    #text = blob.correct()
    text = re.sub(r'\b\w{1,1}\b', '', text)
    #text = word_tokenize(str(text))
    #text = ' '.join([w for w in text if len(w)>1])
    #text = re.sub('sap','asap',str(text))
    return text


# df['consumer_complaint_narrative'] = df.consumer_complaint_narrative.astype(str)
# df['clean_narrative'] = df['consumer_complaint_narrative'].swifter.apply(total_clean)
# sampling_dict = {'Closed with explanation': 8000, 'Closed with non-monetary relief': 8361, 'Closed with monetary relief': 4969,
# 'Closed': 1722, 'Untimely response':520}
# undersample = RandomUnderSampler(sampling_strategy=sampling_dict)
# X_under, y_under = undersample.fit_resample(df[['consumer_complaint_narrative', 'company_public_response',
#        'clean_narrative']], df.company_response_to_consumer)
# y_under[y_under =='Untimely response'] = 'Closed with explanation'
# y_under[y_under =='Closed'] = 'Closed with explanation'
# y_under.value_counts()
# sampling_dict = {'Closed with explanation': 10242, 'Closed with non-monetary relief': 10000, 'Closed with monetary relief': 10000}
# oversample = RandomOverSampler(sampling_strategy=sampling_dict)
# X_over, y_over = oversample.fit_resample(X_under, y_under)
# y_over.value_counts()
# tokenizer = Tokenizer()
# x = X_over.clean_narrative
# y = pd.get_dummies(y_over).values

# tokenizer.fit_on_texts(x)
# seq = tokenizer.texts_to_sequences(x)
# pad_seq = pad_sequences(seq,maxlen = 500,padding='post',truncating='pre')
# vocab_size = len(tokenizer.word_index)+1
# vocab_size
# word2vec = KeyedVectors.load_word2vec_format('/kaggle/input/nlpword2vecembeddingspretrained/GoogleNews-vectors-negative300.bin', \
#         binary=True)
# print('Found %s word vectors of word2vec' % len(word2vec.vocab))
# embedding_matrix = np.zeros((vocab_size, 300))
# for word, i in tokenizer.word_index.items():
#     if word in word2vec.vocab:
#         embedding_matrix[i] = word2vec.word_vec(word)
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# model = Sequential()
# model.add(Embedding(vocab_size,300,input_length=500,weights = [embedding_matrix],trainable = False))
# model.add(Bidirectional(GRU(32,return_sequences=True)))
# #model.add(Convolution1D(32,2,activation='relu'))
# #model.add(Convolution1D(64,3,activation = 'relu'))
# model.add(GlobalMaxPooling1D())
# model.add(Dense(32,activation = 'relu'))
# #model.add(Dropout(0.2))
# model.add(Dense(3,activation = 'softmax'))

# model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy',f1_m,precision_m,recall_m])

# filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]

# model.fit(pad_seq,y,batch_size = 128,epochs = 8,validation_split = 0.10,callbacks=callbacks_list)


 
# from sklearn.feature_extraction.text import CountVectorizer
# import re

# stop_words = set(stopwords.words('english'))
# #get the text column 
# docs=clean_df['Total Clean Text'].to_list()
 

# cv=CountVectorizer(max_df=0.85,stop_words=stop_words)
# word_count_vector=cv.fit_transform(docs)
# from sklearn.feature_extraction.text import CountVectorizer
# import re

# stop_words = set(stopwords.words('english'))
# #get the text column 
# docs=clean_df['Total Clean Text'].to_list()
 

# cv=CountVectorizer(max_df=0.85,stop_words=stop_words)
# word_count_vector=cv.fit_transform(docs)
# from sklearn.feature_extraction.text import TfidfTransformer

# feature_names=cv.get_feature_names()
# tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
# tfidf_transformer.fit(word_count_vector) 

# def extract_topn_from_vector(doc,topn=500):
    
#     tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
#     coo_matrix = tf_idf_vector.tocoo()
    
#     top500=heapq.nlargest(topn, coo_matrix.data)
    
#     dictionary = OrderedDict(dict())
    
#     for idx,score in list(zip(coo_matrix.col,coo_matrix.data)):
#         dictionary[feature_names[idx]]=score
    
#     words = [w for w in doc.split() if w not in stop_words]
#     try:
        
#         results = OrderedDict({x:dictionary[x] for x in words if dictionary[x] in top500})
#         return ' '.join(results.keys())
#     except:
#         print('not converted')
#         return doc

# # urgent_complaints['top_500'] = urgent_complaints['Total Clean'].swifter.apply(extract_topn_from_vector)


# def extract_topn_from_vector(doc,topn=500):
    
#     tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
#     coo_matrix = tf_idf_vector.tocoo()
    
#     top500=heapq.nlargest(topn, coo_matrix.data)
    
#     dictionary = OrderedDict(dict())
    
#     for idx,score in list(zip(coo_matrix.col,coo_matrix.data)):
#         dictionary[feature_names[idx]]=score
    
#     words = [w for w in doc.split() if w not in stop_words]
#     try:
        
#         results = OrderedDict({x:dictionary[x] for x in words if dictionary[x] in top500})
#         return ' '.join(results.keys())
#     except:
#         print('not converted')
#         return doc
    
# feature_names=cv.get_feature_names()
 
# # get the document that we want to extract keywords from
# doc=clean_df['Total Clean Text'][1]

# keywords=extract_topn_from_vector(doc)
# urgent_complaints['top_500'] = urgent_complaints['Total Clean'].swifter.apply(extract_topn_from_vector)
# import pickle
# with open('tfidf_nostopw.pk', 'wb') as fin:
#     pickle.dump(tfidf_transformer, fin)
# fin.close()

# le = LabelEncoder()
# tokenizer = Tokenizer()
# x = urgent_complaints['top_500']
# y = le.fit_transform(urgent_complaints['Target'])

# tokenizer.fit_on_texts(x)
# seq = tokenizer.texts_to_sequences(x)
# pad_seq = pad_sequences(seq,maxlen = 500,padding='post',truncating='pre')
# vocab_size = len(tokenizer.word_index)+1
# vocab_size
# word2vec = KeyedVectors.load_word2vec_format('/kaggle/input/nlpword2vecembeddingspretrained/GoogleNews-vectors-negative300.bin', \
#         binary=True)
# print('Found %s word vectors of word2vec' % len(word2vec.vocab))
# embedding_matrix = np.zeros((vocab_size, 300))
# for word, i in tokenizer.word_index.items():
#     if word in word2vec.vocab:
#         embedding_matrix[i] = word2vec.word_vec(word)
# from keras import backend as K

# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))
# def LSTM_model():
    
#     model = Sequential()
#     # input layer
#     model.add(Embedding(vocab_size,300,input_length=500,weights = [embedding_matrix],trainable = False))
 
#     # LSTM layer
#     model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    
# #     # hidden layer with dropout
# #     model.add(Dense(32,activation = 'relu'))
# #     model.add(Dropout(0.2))

#     # output layer
#     model.add(Dense(1,activation = 'sigmoid'))

#     model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy',f1_m,precision_m,recall_m])

#     filepath="f1_score-{epoch:02d}-{val_f1_m:.2f}.h5"

#     return model
# def CNN_model():
#     model = Sequential()
    
#     # input layer
#     model.add(Embedding(vocab_size,300,input_length=500,weights = [embedding_matrix],trainable = False))
    
#     # CNN layer
#     # model.add(Convolution1D(32,2,activation='relu'))
#     model.add(Convolution1D(64,3,activation = 'relu'))
    
#     # pooling layer
#     model.add(GlobalMaxPooling1D())
    
#     # hidden layer
#     model.add(Dense(32,activation = 'relu'))
#     model.add(Dropout(0.2))

#     # output layer
#     model.add(Dense(1,activation = 'sigmoid'))

#     model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy',f1_m,precision_m,recall_m])

#     return model
# ## Bi directional CRU for binary classification

# def BiDirGRU_model():
#     model = Sequential()
    
#     #input layer
#     model.add(Embedding(vocab_size,300,input_length=500,weights = [embedding_matrix],trainable = False))
    
#     # Bi-Directional GRU layer
#     model.add(Bidirectional(GRU(64,return_sequences=True)))

#     # pooling layer
#     model.add(GlobalMaxPooling1D())
#     model.add(Dense(32,activation = 'relu'))
    
# #     # hidden layer
# #     model.add(Dense(32,activation = 'relu'))
# #     model.add(Dropout(0.2))

#     # output layer
#     model.add(Dense(1,activation = 'sigmoid'))

#     model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy',f1_m,precision_m,recall_m])

#     return model
# filepath="f1_score-{epoch:02d}-{val_f1_m:.2f}.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# es = EarlyStopping(monitor='val_loss',patience = 5)
# callbacks_list = [es,checkpoint]
# estimator1 = KerasClassifier(build_fn = LSTM_model, epochs=20, batch_size=64 ,  callbacks=callbacks_list)
# estimator2 = KerasClassifier(build_fn = CNN_model, epochs=20, batch_size=64 ,  callbacks=callbacks_list)
# estimator3 = KerasClassifier(build_fn = BiDirGRU_model, epochs=20, batch_size=64 ,  callbacks=callbacks_list)

# kfold = StratifiedKFold(n_splits=10, shuffle=True)
# results = cross_val_score(pipeline, pad_seq, y, cv=kfold, fit_params = {'mlp__callbacks': callbacks_list})
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# filepath="LSTM_f1_score-{epoch:02d}-{val_f1_m:.2f}.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# es = EarlyStopping(monitor='val_loss',patience = 5)
# callbacks_list = [es,checkpoint]

# model1 = LSTM_model()
# model1.fit(pad_seq,y,batch_size = 64,epochs = 30, validation_split = 0.20, callbacks=callbacks_list)
# filepath="Bi-Dir-GRUf1_score-{epoch:02d}-{val_f1_m:.2f}.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# es = EarlyStopping(monitor='val_loss',patience = 5)
# callbacks_list = [es,checkpoint]

# model2 = BiDirGRU_model()
# model2.fit(pad_seq,y,batch_size = 64,epochs = 30, validation_split = 0.25, callbacks=callbacks_list)
# filepath="CNN-f1_score-{epoch:02d}-{val_f1_m:.2f}.h5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# es = EarlyStopping(monitor='val_loss',patience = 5)
# callbacks_list = [es,checkpoint]

# model3 = CNN_model()
# model3.fit(pad_seq,y,batch_size = 64,epochs = 30, validation_split = 0.20, callbacks=callbacks_list)


# st = StanfordNERTagger('/kaggle/input/stanfordenglish3class/english.all.3class.distsim.crf.ser',\
#                         '/kaggle/input/stanfordnerjar/stanford-ner.jar',\
#                         encoding='utf-8')

# def stanfordNE2BIO(tagged_sent):
#     bio_tagged_sent = []
#     prev_tag = "O"
#     for token, tag in tagged_sent:
#         if tag == "O": #O
#             bio_tagged_sent.append((token, tag))
#             prev_tag = tag
#             continue
#         if tag != "O" and prev_tag == "O": # Begin NE
#             bio_tagged_sent.append((token, "B-"+tag))
#             prev_tag = tag
#         elif prev_tag != "O" and prev_tag == tag: # Inside NE
#             bio_tagged_sent.append((token, "I-"+tag))
#             prev_tag = tag
#         elif prev_tag != "O" and prev_tag != tag: # Adjacent NE
#             bio_tagged_sent.append((token, "B-"+tag))
#             prev_tag = tag

#     return bio_tagged_sent

# def stanfordNE2tree(text):
    
#     tokenized_text = word_tokenize(text)
#     ne_tagged_sent = st.tag(tokenized_text)
    
#     bio_tagged_sent = stanfordNE2BIO(ne_tagged_sent)
#     sent_tokens, sent_ne_tags = zip(*bio_tagged_sent)
#     sent_pos_tags = [pos for token, pos in pos_tag(sent_tokens)]

#     sent_conlltags = [(token, pos, ne) for token, pos, ne in zip(sent_tokens, sent_pos_tags, sent_ne_tags)]
#     ne_tree = conlltags2tree(sent_conlltags)
    
#     ne_in_sent=[]
#     for subtree in ne_tree:
#         if type(subtree) == Tree: # If subtree is a noun chunk, i.e. NE != "O"
#             ne_label = subtree.label()
#             ne_string = " ".join([token for token, pos in subtree.leaves()])
#             ne_in_sent.append((ne_string, ne_label))
#     return ne_in_sent

# clean_df['entities'] = clean_df['Total Clean Text'].swifter.apply(entities)
# stanfordNE2tree(clean_df['Total Clean Text'][1])
# nlp = en_core_web_sm.load()
# def spacy_ner(text):
#     doc = nlp(text)
#     organisation = [X.text for X in doc.ents if X.label_=='ORG']
#     return ', '.join(organisation)
# clean_df['entities'] = clean_df['Total Clean Text'].swifter.apply(spacy_ner)
# num_partitions = 4 #number of partitions to split dataframe
# num_cores = 4 #number of cores on your machine

# def parallelize_dataframe(df, func):
#     df_split = np.array_split(df, num_partitions)
#     pool = Pool(num_cores)
#     df = pd.concat(pool.map(func, df_split))
#     pool.close()
#     pool.join()
#     return df
# def multiply_columns(data):
#     data['organisation'] = data['Total Clean Text'].swifter.apply(spacy_ner)
#     return data
# clean_df = parallelize_dataframe(clean_df, multiply_columns)
# import pandarallel
# pandarallel.pandarallel.initialize(progress_bar= True)
# clean_df['Organisation'] = clean_df['Total Clean Text'].parallel_apply(spacy_ner)
# companies = pd.read_csv('/kaggle/working/All_organisations.csv', names= ['organisations'])
# companies.organisations = companies.organisations.str.strip()
companies = pd.Dataframe(companies.organisations.unique(),column=['organisations'])
# org = pd.read_csv('/kaggle/input/company-corpus/All_organisations.csv')

# embedding_data = pd.read_csv('/kaggle/input/cfpbmarch2020/complaints-2020-05-04_21_58.csv')
# embedding_data['Consumer complaint narrative'] = embedding_data['Consumer complaint narrative'].astype(str)
# embedding_data = embedding_data.drop_duplicates(subset='Consumer complaint narrative')
## basic cleaning

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
lem = WordNetLemmatizer()

def corpus_related_cleaning(text):
    text = re.sub('XXXX',' UNKNOWN ',text)
    text = re.sub('XX/XX/','',text)
    text = re.sub('UNKNOWN   UNKNOWN','UNKNOWN',text)
    text = re.sub('\n',' ',text)
    text = re.sub('  ',' ',text)

    return text

def clean_text(text):
    
    
    """
        text: a string
        
        return: modified initial string
    """
#     text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = corpus_related_cleaning(text)
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    words = word_tokenize(text)
#     text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    words = [lem.lemmatize(w,'v') for w in words if w not in STOPWORDS]
#     words = [lem.lemmatize(w,'v') for w in words]
    return ' '.join(words)
    
# df['consumer_complaint_narrative'] = df.consumer_complaint_narrative.astype(str)


# df['clean_narrative'] = df['consumer_complaint_narrative'].swifter.apply(clean_text)
# %%time
# ## multiprocessing method 1


# import pandas as pd
# import numpy as np
# import seaborn as sns
# from multiprocessing import Pool

# num_partitions = 4 #number of partitions to split dataframe
# num_cores = 4 #number of cores on your machine

# def parallelize_dataframe(df, func):
#     df_split = np.array_split(df, num_partitions)
#     pool = Pool(num_cores)
#     df = pd.concat(pool.map(func, df_split))
#     pool.close()
#     pool.join()
#     return df

# def multiply_columns(data):
#     data['Clean_narrative'] = data['Consumer complaint narrative'].swifter.apply(clean_text)
#     return data
# %%time
# embedding_data = parallelize_dataframe(embedding_data, multiply_columns)
# %%time
# import tensorflow as tf
# # detect and init the TPU
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)

# # instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
# %%time
# import pandarallel
# pandarallel.pandarallel.initialize(progress_bar= True)

# test_data['Clean_narrative'] = test_data['Consumer complaint narrative'].parallel_apply(clean_text)
# embedding_data.to_csv('cleaned_mar20.csv')
# %%time

# ## multiprocessing method 2
# import multiprocessing as mp
# from tqdm import tqdm

# pool = mp.Pool(mp.cpu_count())

# Clean_narratives = pool.map(clean_text, embedding_data['Consumer complaint narrative'])
# pool.terminate()
# pool.join()
embedding_data = pd.read_csv('/kaggle/input/cleanedmar20/cleaned_mar20.csv')
embedding_data['Company response to consumer'].value_counts()
embedding_data = embedding_data[(embedding_data['Company response to consumer'].str.contains('Closed with non-monetary relief')) | (embedding_data['Company response to consumer'].str.contains('Closed with monetary relief'))]
embedding_data.shape

# %%time
# train_sentences = list(embedding_data.Clean_narrative.swifter.apply(str.split).values)
# %%time

# model = gensim.models.Word2Vec(sentences=train_sentences, size=300, workers=4)
# model.save('custom_corpus_500k.model')
from gensim.models import Word2Vec
# custom_embed = Word2Vec.load("/kaggle/working/custom_corpus_500k.model")
custom_embed = Word2Vec.load('/kaggle/input/customembed/custom_corpus_500k.model')
custom_embed.most_similar('fraud')
# len(custom_embed.wv.vocab.keys())
# embedding_data['Company response to consumer'].value_counts()
embedding_data = embedding_data.fillna('')

from sklearn.feature_extraction.text import TfidfVectorizer
from yellowbrick.text import TSNEVisualizer
from yellowbrick.datasets import load_hobbies

# Load the data and create document vectors
corpus = load_hobbies()
tfidf = TfidfVectorizer()

X = tfidf.fit_transform(embedding_data['Clean_narrative'][:3000])
y = embedding_data['Company response to consumer']

# Create the visualizer and draw the vectors
tsne = TSNEVisualizer()
tsne.fit(X, y)
tsne.show()
# from sklearn.feature_extraction.text import CountVectorizer
# import re

# stop_words = set(stopwords.words('english'))
# #get the text column 
# docs=embedding_data['Clean_narrative'].to_list()

# cv=CountVectorizer(max_df=0.85,stop_words=stop_words)
# word_count_vector=cv.fit_transform(docs)
# from sklearn.feature_extraction.text import TfidfTransformer
 
# tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
# tfidf_transformer.fit(word_count_vector)

# feature_names=cv.get_feature_names()

# def extract_topn_from_vector(doc,topn=500):
    
#     tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
#     coo_matrix = tf_idf_vector.tocoo()
    
#     top500=heapq.nlargest(topn, coo_matrix.data)
    
#     dictionary = OrderedDict(dict())
    
#     for idx,score in list(zip(coo_matrix.col,coo_matrix.data)):
#         dictionary[feature_names[idx]]=score
    
#     words = [w for w in doc.split() if w not in stop_words]
#     try:
        
#         results = OrderedDict({x:dictionary[x] for x in words if dictionary[x] in top500})
#         return ' '.join(results.keys())
#     except:
#         print('not converted')
#         return doc

# embedding_data['top_500'] = embedding_data['Clean_narrative'].parallel_apply(extract_topn_from_vector)
le = LabelEncoder()
tokenizer = Tokenizer()
x = embedding_data['Clean_narrative']
y = le.fit_transform(embedding_data['Company response to consumer'])
# y=pd.get_dummies(embedding_data['Company response to consumer']).values

tokenizer.fit_on_texts(x)
seq = tokenizer.texts_to_sequences(x)
pad_seq = pad_sequences(seq,maxlen = 500,padding='post',truncating='pre')
# sampling_dict = {0: 27480, 1: 27480}
# undersample = RandomUnderSampler(sampling_strategy=sampling_dict)

# X_under, y_under = undersample.fit_resample(pad_seq, y)
with open('tokenizer_bin.pkl','wb') as fin:
    pickle.dumps(tokenizer)
fin.close()
# import pandas as pd
# ids = embedding_data["Consumer complaint narrative"]
# embedding_data[ids.isin(ids[ids.duplicated()])].sort_values('Consumer complaint narrative').to_csv('duplicates.csv')
vocab_size = len(tokenizer.word_index)+1
vocab_size
%%time
embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tokenizer.word_index.items():
    if word in custom_embed.wv.vocab:
        embedding_matrix[i] = custom_embed.wv.word_vec(word)
model = Sequential()

#input layer
model.add(Embedding(vocab_size,300,input_length=500,weights = [embedding_matrix],trainable = True))

# Bi-Directional GRU layer
model.add(Bidirectional(GRU(64,return_sequences=True)))

# pooling layer
model.add(GlobalMaxPooling1D())
model.add(Dense(32,activation = 'relu'))

#     # hidden layer
#     model.add(Dense(32,activation = 'relu'))
#     model.add(Dropout(0.2))

# output layer
model.add(Dense(1,activation = 'sigmoid'))

model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy',f1_m,precision_m,recall_m])

filepath="Bi-Dir-GRUf1_score-{epoch:02d}-{val_f1_m:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
es = EarlyStopping(monitor='val_loss',patience = 4)
callbacks_list = [es,checkpoint]
# model2.fit(pad_seq,y,batch_size = 32,epochs = 15, validation_split = 0.25, callbacks=callbacks_list)
## load model

loaded_model = tf.keras.models.load_model('/kaggle/input/binmodelmonetarygru/Bi-Dir-GRUf1_score-03-0.90.h5',custom_objects={'f1_m':f1_m,'precision_m':precision_m,'recall_m':recall_m})

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(pad_seq,y,test_size = 0.1,random_state= 42)
model.fit(x_train,y_train,batch_size = 32,epochs = 3, validation_split = 0.1, callbacks=callbacks_list)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_under,y_under,test_size = 0.1,random_state= 42)

model.fit(x_train,y_train,batch_size = 32,epochs = 5, validation_split = 0.1, callbacks=callbacks_list)
predictions = loaded_model.predict(x_test)
predictions
from sklearn.metrics import f1_score,precision_score,recall_score
recall = []
for thresh in np.arange(0,1,0.01):
    thresh = np.round(thresh,2)
    print('Recall Score at threshold {0} is {1}'.format(thresh,recall_score(y_test,(predictions>thresh).astype(int))))
    recall.append(recall_score(y_test,(predictions>thresh).astype(int)))
precision = []
for thresh in np.arange(0,1,0.01):
    thresh = np.round(thresh,2)
    print('Precision Score at threshold {0} is {1}'.format(thresh,precision_score(y_test,(predictions>thresh).astype(int))))
    precision.append(precision_score(y_test,(predictions>thresh).astype(int)))
f1 = []
for thresh in np.arange(0,1,0.01):
    thresh = np.round(thresh,2)
    print('F1 Score at threshold {0} is {1}'.format(thresh,f1_score(y_test,(predictions>thresh).astype(int))))
    f1.append(f1_score(y_test,(predictions>thresh).astype(int)))
# x_axis = range(0,100)
# plt.plot(x_axis,recall,label = 'Recall')
# plt.plot(x_axis,precision,label = 'Precision')
# plt.plot(x_axis,f1,label = 'F1 Score')

# plt.legend()

# plt.show()

x_axis = range(0,100)
plt.figure(figsize=(20,12))
plt.plot(x_axis,recall,label = 'Recall')
plt.plot(x_axis,precision,label = 'Precision')
plt.plot(x_axis,f1,label = 'F1 Score')
plt.legend()


plt.xticks(range(100))
plt.grid()
plt.show()
x_test_prediction = []
for i in tqdm(range(len(x_test))):
    if predictions[i]>0.5:
        x_test_prediction.append(1)
    else:
        x_test_prediction.append(0)

# prediction = np.array(april_prediction_2_classes)
# encoded = np.array(df2['Encoded'].values)
#encoded
print(classification_report(y_test,x_test_prediction))


import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.heatmap(confusion_matrix(y_test,x_test_prediction),annot = True,fmt = 'g')
x_test_prediction = []
for i in tqdm(range(len(x_test))):
    if predictions[i]>0.43:
        x_test_prediction.append(1)
    else:
        x_test_prediction.append(0)

# prediction = np.array(april_prediction_2_classes)
# encoded = np.array(df2['Encoded'].values)
#encoded
print(classification_report(y_test,x_test_prediction))


import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.heatmap(confusion_matrix(y_test,x_test_prediction),annot = True,fmt = 'g')
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, thresholds = precision_recall_curve(y_test, predictions)
from sklearn.metrics import auc
auc = auc(recall, precision)
auc
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(recall, precision, marker='.', label='Logistic')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()
from sklearn.metrics import f1_score,precision_score,recall_score
test_data = pd.read_csv('/kaggle/input/testapril/complaints-2020-05-07_09_29.csv')
# import tensorflow as tf
loaded_model = tf.keras.models.load_model('/kaggle/input/binmodelmonetarygru/Bi-Dir-GRUf1_score-03-0.90.h5',custom_objects={'f1_m':f1_m,'precision_m':precision_m,'recall_m':recall_m})

import pandarallel
pandarallel.pandarallel.initialize(progress_bar= True)
test_data['Clean_narrative'] = test_data['Consumer complaint narrative'].progress_apply(clean_text)
test_df1 = test_data[test_data['Company response to consumer']=='Closed with explanation']
test_df1.reset_index(drop = True,inplace = True)
test_df1.shape
test_df1.shape

b = tokenizer.texts_to_sequences(test_df1['Clean_narrative'])
b_pad = pad_sequences(b,maxlen=500)
results = loaded_model.predict(b_pad)
pd.DataFrame(results).shape
april_prediction = []
for i in tqdm(range(len(test_df1))):
    if results[i]>0.5:
        april_prediction.append(1)
    else:
        april_prediction.append(0)
from collections import Counter
Counter(april_prediction)

df2 = test_data[test_data['Company response to consumer']!='Closed with explanation']
df2.reset_index(drop = True,inplace = True)
df2.shape
df2['Clean_narrative'] = df2['Consumer complaint narrative'].progress_apply(clean_text)
df2

b = tokenizer.texts_to_sequences(df2['Clean_narrative'])
b_pad = pad_sequences(b,maxlen=500)
results = loaded_model.predict(b_pad)

april_prediction_2_classes = []

for i in tqdm(range(len(df2))):
  
    if results[i]>0.29:
        april_prediction_2_classes.append(1)
    else:
        april_prediction_2_classes.append(0)
df2['threshold 0.5'] = april_prediction_2_classes
df2['threshold 0.43'] = april_prediction_2_classes
df2['threshold 0.29'] = april_prediction_2_classes
df2.to_csv('cross_tab.csv')
from collections import Counter
Counter(april_prediction_2_classes)
def encoding(text):
    if text == 'Closed with non-monetary relief':
        return int(1)
    elif text == "Closed with monetary relief":
        return int(0)
    
df2['Encoded'] = df2['Company response to consumer'].apply(encoding)
df2['Encoded'] = df2['Encoded'].astype("int64")


prediction = np.array(april_prediction_2_classes)
encoded = np.array(df2['Encoded'].values)
#encoded
prediction.shape,encoded.shape
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
print(classification_report(encoded,prediction))
sns.heatmap(confusion_matrix(encoded,prediction),annot = True,fmt = 'g')
print(classification_report(encoded,prediction))
sns.heatmap(confusion_matrix(encoded,prediction),annot = True,fmt = 'g')
print(classification_report(encoded,prediction))
sns.heatmap(confusion_matrix(encoded,prediction),annot = True,fmt = 'g')
recall = []
for thresh in np.arange(0,1,0.01):
    thresh = np.round(thresh,2)
    print('Recall Score at threshold {0} is {1}'.format(thresh,recall_score(df2['Encoded'],(results>thresh).astype(int))))
    recall.append(recall_score(df2['Encoded'],(results>thresh).astype(int)))
precision = []
for thresh in np.arange(0,1,0.01):
    thresh = np.round(thresh,2)
    print('Precision Score at threshold {0} is {1}'.format(thresh,precision_score(df2['Encoded'],(results>thresh).astype(int))))
    precision.append(precision_score(df2['Encoded'],(results>thresh).astype(int)))
f1 = []
for thresh in np.arange(0,1,0.01):
    thresh = np.round(thresh,2)
    print('F1 Score at threshold {0} is {1}'.format(thresh,f1_score(df2['Encoded'],(results>thresh).astype(int))))
    f1.append(f1_score(df2['Encoded'],(results>thresh).astype(int)))
x_axis = range(0,100)
plt.figure(figsize=(20,12))
plt.plot(x_axis,recall,label = 'Recall')
plt.plot(x_axis,precision,label = 'Precision')
plt.plot(x_axis,f1,label = 'F1 Score')
plt.legend()


plt.xticks(range(100))
plt.grid()
plt.show()
4+3
april_prediction_2_classes = []
for i in tqdm(range(len(df2))):
  
    if results[i]>0.38:
        april_prediction_2_classes.append(1)
    else:
        april_prediction_2_classes.append(0)
prediction = np.array(april_prediction_2_classes)
encoded = np.array(df2['Encoded'].values)
#encoded
print(classification_report(encoded,prediction))
import seaborn as sns
sns.heatmap(confusion_matrix(encoded,prediction),annot = True,fmt = 'g')
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

prediction = []
for i in tqdm(range(len(x_test))):
    if predictions[i]>0.5:
        prediction.append(1)
    else:
        prediction.append(0)
encoded = y_test

print(classification_report(encoded,prediction))
import seaborn as sns
sns.heatmap(confusion_matrix(encoded,prediction),annot = True,fmt = 'g')
34/(34+484)
pd.DataFrame([df2['Clean_narrative'],df2['Encoded'],prediction], columns=['narrative','encoded','pred']).to_csv('swap_analysis.csv')
df2['Encoded'].shape,results.shape




















