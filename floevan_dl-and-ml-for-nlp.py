import pickle

import time
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Sequential

from keras.layers.recurrent import LSTM, GRU,SimpleRNN

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.embeddings import Embedding

from keras.layers.normalization import BatchNormalization

from keras.utils import np_utils

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D

from keras.preprocessing import sequence, text

from keras.callbacks import EarlyStopping





import matplotlib.pyplot as plt

import seaborn as sns





from wordcloud import WordCloud, STOPWORDS



stopword=set(STOPWORDS)



import warnings

warnings.filterwarnings("ignore")
val = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/validation.csv")

test = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/test.csv')

train = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train.info(memory_usage='deep')
for dtype in ['int64','object']:

    selected_dtype = train.select_dtypes(include=[dtype])

    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()

    mean_usage_mb = mean_usage_b / 1024 ** 2

    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))
train.head()
val.head()
test.head()
wc = WordCloud(stopwords=stopword)

plt.figure(figsize=(18,10))

wc.generate(str(train['comment_text']))

plt.imshow(wc)

plt.title('Common words in comments');
def new_len(x):

    if type(x) is str:

        return len(x.split())

    else:

        return 0



train["comment_words"] = train["comment_text"].apply(new_len)
Toxic = train[train['toxic'] == 1]

NoToxic = train[train['toxic'] == 0]
wc = WordCloud(stopwords= stopword)

plt.figure(figsize = (18,12))

wc.generate(str(Toxic['comment_text']))

plt.imshow(wc)

plt.title('Words frequented in Toxic Comments');
wc = WordCloud(stopwords= stopword)

plt.figure(figsize = (18,12))

wc.generate(str(NoToxic['comment_text']))

plt.imshow(wc)

plt.title('Words frequented in Clean Comments')
obscene = train[train['obscene'] == 1]

severe = train[train['severe_toxic'] == 1]

threat = train[train.threat == 1]

insult = train[train.insult == 1]
wc = WordCloud(stopwords= stopword)

plt.figure(figsize = (18,12))

wc.generate(str(obscene['comment_text']))

plt.imshow(wc)

plt.title('Words frequented in obscene Comments');
wc = WordCloud(stopwords= stopword)

plt.figure(figsize = (18,12))

wc.generate(str(severe['comment_text']))

plt.imshow(wc)

plt.title('Words frequented in severe Comments');
wc = WordCloud(stopwords= stopword)

plt.figure(figsize = (18,12))

wc.generate(str(threat['comment_text']))

plt.imshow(wc)

plt.title('Words frequented in threat Comments');
wc = WordCloud(stopwords= stopword)

plt.figure(figsize = (18,12))

wc.generate(str(insult['comment_text']))

plt.imshow(wc)

plt.title('Words frequented in insult Comments');
train.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)
train = train.loc[:30000,:]

train.shape
#We will check the maximum number of words that can be present in a comment , this will help us in padding later



train['comment_text'].apply(lambda x:len(str(x).split())).max()
#Writing a function for getting auc score for validation



def roc_auc(predictions,target):

    '''

    This methods returns the AUC Score when given the Predictions

    and Labels

    '''

    

    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)

    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc
from nltk.corpus import stopwords

import nltk

import re

import string, collections

from nltk.stem.snowball import EnglishStemmer

from nltk.stem import WordNetLemmatizer



stop_words = set(stopwords.words('english'))

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

stemmer = EnglishStemmer()

lemmatizer = WordNetLemmatizer()



def clean_text(text):

    text = text.lower() #make text lowercase and fill na

    text = re.sub('\[.*?\]', '', text) 

    text = re.sub('\\n', '',str(text))

    text = re.sub("\[\[User.*",'',str(text))

    text = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(text))

    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text) #remove hyperlinks

    text = re.sub(r'\:(.*?)\:', '', text) #remove emoticones

    text = re.sub(r'[\w\.-]+@[\w\.-]+', '', str(text)) #remove email

    text = re.sub(r'(?<=@)\w+', '', text) #remove @

    text = re.sub(r'[0-9]+', '', text) #remove numbers

    text = re.sub("[^A-Za-z0-9 ]", '', text) #remove non alphanumeric like ['@', '#', '.', '(', ')']

    text = re.sub(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*', '', text) #remove punctuations from sentences

    text = re.sub('<.*?>+', '', str(text))

    text = re.sub('[%s]' % re.escape(string.punctuation), '', str(text))

    text = re.sub('\w*\d\w*', '', str(text))

    text = tokenizer.tokenize(text)

    text = [word for word in text if not word in stop_words]

    #text = [lemmatizer.lemmatize(word) for word in text]

    text = [stemmer.stem(word) for word in text]

    final_text = ' '.join( [w for w in text if len(w)>1] ) #remove word with one letter

    return final_text











#val["comment_text"] = clean_text(str(val["comment_text"]))

#test_data["content"] = clean_text(str(test_data["content"]))

#train["comment_text"] = clean_text(str(train["comment_text"]))



                  

val['comment_text'] = val['comment_text'].apply(lambda x : clean_text(x))



test['content'] = test['content'].apply(lambda x : clean_text(x))



train['comment_text'] = train['comment_text'].apply(lambda x : clean_text(x))  
train.head()
xtrain, xval, ytrain, yval = train_test_split(train['comment_text'].values, train['toxic'].values, 

                                                  stratify=train.toxic.values, 

                                                  random_state=42, 

                                                  test_size=0.2, shuffle=True)
train.head()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegressionCV

from sklearn.pipeline import make_pipeline

from imblearn.over_sampling import  SMOTE





smt = SMOTE(random_state=777, k_neighbors=1)



vec = TfidfVectorizer(min_df=3,max_features=10000,strip_accents='unicode',

                     analyzer='word',ngram_range=(1,2),token_pattern=r'\w{1,}',use_idf=1,smooth_idf=1,sublinear_tf=1,

                     stop_words='english')



vec_fit=vec.fit_transform(xtrain)



clf = LogisticRegressionCV()





# Over Sampling

X_SMOTE, y_SMOTE = smt.fit_sample(vec_fit, ytrain)
from collections import Counter

#we over sampled it 

print(Counter(y_SMOTE))
#dealed with imbalanced
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=0.1, solver='sag')

scores = cross_val_score(clf, X_SMOTE,y_SMOTE, cv=5,scoring='f1_weighted')
scores.mean()
clf.fit(X_SMOTE,y_SMOTE)
#Writing a function for getting auc score for validation



def roc_auc(predictions,target):

    '''

    This methods returns the AUC Score when given the Predictions

    and Labels

    '''

    

    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)

    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc





from sklearn import metrics



def print_report1(data, y):

    y_test =  y

    test_features=vec.transform(data)

    y_pred = clf.predict(test_features)

    report = metrics.classification_report(y_test, y_pred, target_names=['Toxic', 'Clean'])

    print(report)

    print("accuracy: {:0.3f}".format(metrics.accuracy_score(y_test, y_pred)))

    print()

    print("Auc: %.2f%%" % (roc_auc(y_pred,y_test)))



print_report1(xval, yval)
import eli5

# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)

eli5.show_weights(clf, vec=vec, top=15,

                  target_names=['clean','toxic'])
print(xval[2])

print('\n')

print(yval[2])
import eli5

eli5.show_prediction(clf, xval[2], vec=vec,

                     target_names=['clean','toxic'],top=15)

val = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/validation.csv")

test = pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/test.csv')

train = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train = train.loc[:12000,:]

train.shape
val['comment_text'] = val['comment_text'].apply(lambda x : clean_text(x))



test['content'] = test['content'].apply(lambda x : clean_text(x))



train['comment_text'] = train['comment_text'].apply(lambda x : clean_text(x)) 
xtrain, xval, ytrain, yval = train_test_split(train['comment_text'].values, train['toxic'].values, 

                                                  stratify=train.toxic.values, 

                                                  random_state=42, 

                                                  test_size=0.2, shuffle=True)
# using keras tokenizer here

token = text.Tokenizer(num_words=None)

max_len = 1500



token.fit_on_texts(list(xtrain) + list(xval))

xtrain_seq = token.texts_to_sequences(xtrain)

xval_seq = token.texts_to_sequences(xval)



#zero pad the sequences

xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)

xval_pad = sequence.pad_sequences(xval_seq, maxlen=max_len)



word_index = token.word_index
model = Sequential()

model.add(Embedding(len(word_index) + 1,

                     300,

                     input_length=max_len))

model.add(SimpleRNN(100))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(xtrain_pad, ytrain, epochs = 5, batch_size = 64, verbose = 2)
pred = model.predict(xval_pad)
print("Auc: %.2f%%" % (roc_auc(pred,yval)))
file_name = 'simpleRNN.sav'

pickle.dump(model , open(file_name, 'wb'))
%%time

model = pickle.load(open('simpleRNN.sav', 'rb'))
pred = model.predict(xval_pad)

print("Auc: %.2f%%" % (roc_auc(pred,yval)))
del model



import gc; gc.collect()

time.sleep(10)
# load the GloVe vectors in a dictionary:



embeddings_index = {}

f = open('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt','r',encoding='utf-8')

for line in tqdm(f):

    values = line.split(' ')

    word = values[0]

    coefs = np.asarray([float(val) for val in values[1:]])

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
# create an embedding matrix for the words we have in the dataset

embedding_matrix = np.zeros((len(word_index) + 1, 200))

for word, i in tqdm(word_index.items()):

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
model = Sequential()

model.add(Embedding(len(word_index) + 1,

                     200,

                     weights=[embedding_matrix],

                     input_length=max_len,

                     trainable=False))



model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

print(model.summary())
model.fit(xtrain_pad, ytrain, epochs=5, batch_size = 64)
pred = model.predict(xval_pad)
print("Auc: %.2f%%" % (roc_auc(pred,yval)))
file_name = 'LSTM.sav'

pickle.dump(model , open(file_name, 'wb'))
del model

import gc; gc.collect()

time.sleep(10)
model = Sequential()

model.add(Embedding(len(word_index) + 1,

                     200,

                     weights=[embedding_matrix],

                     input_length=max_len,

                     trainable=False))

model.add(SpatialDropout1D(0.3))

model.add(GRU(300))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

print(model.summary())
model.fit(xtrain_pad, ytrain, epochs = 5, batch_size = 64)
pred = model.predict(xval_pad)

print("Auc: %.2f%%" % (roc_auc(pred,yval)))
file_name = 'GRU.sav'

pickle.dump(model , open(file_name, 'wb'))



del model

import gc; gc.collect()

time.sleep(10)
model = Sequential()

model.add(Embedding(len(word_index) + 1,

                     200,

                     weights=[embedding_matrix],

                     input_length=max_len,

                     trainable=False))

model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))



model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

print(model.summary())
model.fit(xtrain_pad, ytrain, epochs=5, batch_size=64)
scores = model.predict(xval_pad)

print("Auc: %.2f%%" % (roc_auc(scores,yval)))
file_name = 'Bidirectional.sav'

pickle.dump(model , open(file_name, 'wb'))



del model

import gc; gc.collect()

time.sleep(10)
def roc_auc(predictions,target):

    '''

    This methods returns the AUC Score when given the Predictions

    and Labels

    '''

    

    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)

    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc



from nltk.corpus import stopwords

import nltk

import re

import string, collections



stop_words = set(stopwords.words('english'))

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')



def clean_text(text):

    text = text.lower() #make text lowercase and fill na

    text = re.sub('\[.*?\]', '', text) 

    text = re.sub('\\n', '',str(text))

    text = re.sub("\[\[User.*",'',str(text))

    text = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(text))

    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text) #remove hyperlinks

    text = re.sub(r'\:(.*?)\:', '', text) #remove emoticones

    text = re.sub(r'[\w\.-]+@[\w\.-]+', '', str(text)) #remove email

    text = re.sub(r'(?<=@)\w+', '', text) #remove @

    text = re.sub(r'[0-9]+', '', text) #remove numbers

    text = re.sub("[^A-Za-z0-9 ]", '', text) #remove non alphanumeric like ['@', '#', '.', '(', ')']

    text = re.sub(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*', '', text) #remove punctuations from sentences

    text = re.sub('<.*?>+', '', str(text))

    text = re.sub('[%s]' % re.escape(string.punctuation), '', str(text))

    text = re.sub('\w*\d\w*', '', str(text))

    text = tokenizer.tokenize(text)

    text = [word for word in text if not word in stop_words]

    final_text = ' '.join( [w for w in text if len(w)>1] ) #remove word with one letter

    return final_text

# load the GloVe vectors in a dictionary:



embeddings_index = {}

f = open('glove.840B.300d.txt','r',encoding='utf-8')

for line in tqdm(f):

    values = line.split(' ')

    word = values[0]

    coefs = np.asarray([float(val) for val in values[1:]])

    embeddings_index[word] = coefs

f.close()



print('Found %s word vectors.' % len(embeddings_index))
# create an embedding matrix for the words we have in the dataset

embedding_matrix = np.zeros((len(word_index) + 1, 300))

for word, i in tqdm(word_index.items()):

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
import math

import sklearn.metrics as sklm





def function(model, model_name):

    

    val = pd.read_csv("validation.csv")

    test = pd.read_csv('test.csv')

    train = pd.read_csv("jigsaw-toxic-comment-train.csv")



    train.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)



    train = train.loc[:12000,:]

    train.shape









    val['comment_text'] = val['comment_text'].apply(lambda x : clean_text(x))



    test['content'] = test['content'].apply(lambda x : clean_text(x))



    train['comment_text'] = train['comment_text'].apply(lambda x : clean_text(x))  

    

    xtrain, xval, ytrain, yval = train_test_split(train['comment_text'].values, train['toxic'].values, 

                                                  stratify=train.toxic.values, 

                                                  random_state=42, 

                                                  test_size=0.2, shuffle=True)

    

    # using keras tokenizer here

    token = text.Tokenizer(num_words=None)

    max_len = 1500



    token.fit_on_texts(list(xtrain) + list(xval))

    xtrain_seq = token.texts_to_sequences(xtrain)

    xval_seq = token.texts_to_sequences(xval)



    #zero pad the sequences

    xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)

    xval_pad = sequence.pad_sequences(xval_seq, maxlen=max_len)



    word_index = token.word_index





    #modeling

    if model == SimpleRNN :

        



        model = Sequential()

        model.add(Embedding(len(word_index) + 1,

                             300,

                             input_length=max_len))

        model.add(SimpleRNN(100))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(model.summary())

        

        model.fit(xtrain_pad, ytrain, epochs = 5, batch_size = 64, verbose = 0)

        

        pred = model.predict(xval_pad)

        

        print("The AUC with {} is: {}".format(model_name,(roc_auc(pred,yval))))



        #save our model 

        

        file_name = 'simpleRNN.sav'

        pickle.dump(model , open(file_name, 'wb'))

        print('Model saved !')

    

    elif model == LSTM:

        

        model = Sequential()

        model.add(Embedding(len(word_index) + 1,

                             300,

                             weights=[embedding_matrix],

                             input_length=max_len,

                             trainable=False))



        model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

        print(model.summary())

        

        model.fit(xtrain_pad, ytrain, epochs = 5, batch_size = 64, verbose = 0)

        

        pred = model.predict(xval_pad)

        

        print("The AUC with {} is: {}".format(model_name,(roc_auc(pred,yval))))



        #save our model 

        

        file_name = 'LSTM.sav'

        pickle.dump(model , open(file_name, 'wb'))

        print('Model saved !')

        

    elif model == GRU:

        

        model = Sequential()

        model.add(Embedding(len(word_index) + 1,

                             300,

                             weights=[embedding_matrix],

                             input_length=max_len,

                             trainable=False))

        model.add(SpatialDropout1D(0.3))

        model.add(GRU(300))

        model.add(Dense(1, activation='sigmoid'))



        model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

        print(model.summary())

        

        model.fit(xtrain_pad, ytrain, epochs = 5, batch_size = 64, verbose = 0)

        

        pred = model.predict(xval_pad)

        

        print("The AUC with {} is: {}".format(model_name,(roc_auc(pred,yval))))



        #save our model 

        

        file_name = 'GRU.sav'

        pickle.dump(model , open(file_name, 'wb'))

        print('Model saved !')

    

    elif model == BiRNN:



        model = Sequential()

        model.add(Embedding(len(word_index) + 1,

                             300,

                             weights=[embedding_matrix],

                             input_length=max_len,

                             trainable=False))

        model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))



        model.add(Dense(1,activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

        print(model.summary())



        

        model.fit(xtrain_pad, ytrain, epochs = 5, batch_size = 64, verbose = 0)

        

        pred = model.predict(xval_pad)

        

        print("The AUC with {} is: {}".format(model_name,(roc_auc(pred,yval))))



        #save our model 

        

        file_name = 'BiRNN.sav'

        pickle.dump(model , open(file_name, 'wb'))

        print('Model saved !')