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
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Flatten, CuDNNLSTM, CuDNNGRU, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, SpatialDropout1D, concatenate
from keras.models import Model, Sequential, load_model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D, GlobalAveragePooling1D
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from nltk.stem import WordNetLemmatizer
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
train_f = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/train.tsv",sep="\t")
train_f.head(10)
train_f.groupby('Sentiment').agg({'PhraseId': 'count'})
test_f = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/test.tsv",sep="\t")
test_f.head()
corpus_sentences = list(map(str,train_f["Phrase"])) + list(map(str,test_f["Phrase"]))
corpus_sentences[66292]

len(corpus_sentences)
#stemmer = PorterStemmer()
#stemmed_words = [stemmer.stem(word.lower()) for word in corpus_sentences]
#stops = set(stopwords.words("english"))

def clean_text(c):
    lemmatizer = WordNetLemmatizer()
    lemmed_words = c.copy()
    i = 0
    for sentences in c:
        temp = [lemmatizer.lemmatize(j) for j in sentences.lower().split()]
        lemmed_words[i] = " ".join(temp)
        i+=1
    text = lemmed_words.copy()
    #text = [re.sub(r'[^\w\s]','',s) for s in lemmed_words]
    return(text)

text = clean_text(corpus_sentences)
len(text)
for i in range(2000):
    if corpus_sentences[i].lower() != text[i] and len(corpus_sentences[i]) > 200:
        print(i)
i=563

print(corpus_sentences[i])
#print(lemmed_words[i])
print(text[i])


#X_train, X_val, y_train, y_val = train_test_split(train_f['Phrase'],train_f['Sentiment'],test_size=0.1)

#print(len(X_train))
#print(len(X_val))

#print(len(y_train))
#print(len(y_val))

X_train = train_f['Phrase']
y_train = train_f['Sentiment']
Xy_train = pd.concat([X_train, y_train], axis=1)
Xy_train.shape
#Xy_val = pd.concat([X_val, y_val], axis=1)
#Xy_val.groupby('Sentiment').agg({'Phrase': 'count'})
Xy_train.groupby('Sentiment').agg({'Phrase': 'count'})
Xy_train = Xy_train.reset_index(drop=True)


i_class0 = np.where(Xy_train['Sentiment'] == 0)[0]
i_class1 = np.where(Xy_train['Sentiment'] == 1)[0]
i_class2 = np.where(Xy_train['Sentiment'] == 2)[0]
i_class3 = np.where(Xy_train['Sentiment'] == 3)[0]
i_class4 = np.where(Xy_train['Sentiment'] == 4)[0]

m = max(len(i_class0), len(i_class1), len(i_class2), len(i_class3), len(i_class4))


print(m)
print(len(i_class0))
print(len(i_class1))
print(len(i_class2))
print(len(i_class3))
print(len(i_class4))

i_class0_upsampled = np.random.choice(i_class0, size=m, replace=True)
i_class1_upsampled = np.random.choice(i_class1, size=m, replace=True)
i_class2_upsampled = i_class2 #max
i_class3_upsampled = np.random.choice(i_class3, size=m, replace=True)
i_class4_upsampled = np.random.choice(i_class4, size=m, replace=True)


t0 = Xy_train.loc[i_class0_upsampled, ]
t1 = Xy_train.loc[i_class1_upsampled, ]
t2 = Xy_train.loc[i_class2_upsampled, ]
t3 = Xy_train.loc[i_class3_upsampled, ]
t4 = Xy_train.loc[i_class4_upsampled, ]

train_fu = t0.append(t1).append(t2).append(t3).append(t4)
train_fu.groupby('Sentiment').agg({'Phrase': 'count'})

train_fu['Phrase'] = clean_text(list(map(str,train_fu["Phrase"])))
Xy_train['Phrase'] = clean_text(list(map(str,Xy_train["Phrase"])))
#Xy_val['Phrase'] = clean_text(list(map(str,Xy_val["Phrase"])))
test_f['Phrase'] = clean_text(list(map(str,test_f["Phrase"])))
max_words = 15000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(list(text))

#list_tokenized_train = tokenizer.texts_to_sequences(train_fu["Phrase"])
list_tokenized_train = tokenizer.texts_to_sequences(Xy_train["Phrase"])
#list_tokenized_val = tokenizer.texts_to_sequences(Xy_val["Phrase"])
list_tokenized_test = tokenizer.texts_to_sequences(test_f["Phrase"])
len(list_tokenized_train)
#len(list_tokenized_val)
len(list_tokenized_test)
num_words = [len(i) for i in text]
plt.hist(num_words)
np.mean(num_words)
max_len = 80
X_train_final = pad_sequences(list_tokenized_train,maxlen=max_len)
#X_val_final = pad_sequences(list_tokenized_val,maxlen=max_len)
X_test_final = pad_sequences(list_tokenized_test,maxlen=max_len)
X_train_final.shape
#train_dummies = pd.get_dummies(train_fu['Sentiment'])
train_dummies = pd.get_dummies(Xy_train['Sentiment'])
y_train_final = train_dummies.values
y_train_final[:10,:]
#train_dummies = pd.get_dummies(Xy_val['Sentiment'])
#y_val_final = train_dummies.values
#y_val_final[:10,:]
np.random.seed(226)
shuffle_indices = np.random.permutation(np.arange(len(X_train_final)))
X_trains = X_train_final[shuffle_indices]
y_trains = y_train_final[shuffle_indices]

print(X_train_final[1])
print(X_trains[1])

print(y_train_final[1])
print(y_trains[1])
phs = Xy_train['Phrase'][shuffle_indices]
phs[0:2]
td = 100

vec = TfidfVectorizer(max_features=td, ngram_range=(1,2))
x_tfidf = vec.fit_transform(phs).toarray()
np.count_nonzero(x_tfidf)/len(phs)
test_tfidf = vec.transform(test_f['Phrase']).toarray()
test_tfidf
def cal_score(c):
    sid = SentimentIntensityAnalyzer()
    i = 0
    a = np.zeros(shape=(len(c),5))
    for sentences in c:
        temp1 = sum([sid.polarity_scores(j)['compound'] for j in sentences.split()])
        temp2 = sum([sid.polarity_scores(j)['compound'] > 0.5 for j in sentences.split()])
        temp3 = sum([sid.polarity_scores(j)['compound'] < -0.5 for j in sentences.split()])
        temp4 = sid.polarity_scores(sentences)['compound']
        temp5 = TextBlob(sentences).sentiment.polarity
        a[i][0] = temp1
        a[i][1] = temp2 / (1+len(sentences.split()))
        a[i][2] = temp3 / (1+len(sentences.split()))
        a[i][3] = temp4
        a[i][4] = temp5
        #a[i][5] = temp4 / (1+len(sentences.split()))
        #a[i][6] = temp5 / (1+len(sentences.split()))
        i+=1
    return(a)

phs2 = cal_score(phs)
phs2
test_phs2 = cal_score(test_f['Phrase'])
test_phs2
#X_train_t, X_train_dev, y_train_t, y_train_dev = train_test_split(np.array(X_trains),np.array(y_trains),test_size=0.3)
#print(len(X_train_dev))

#remove_i = []

#for i in range(len(X_train_dev)):
#    if (X_train_dev[i] in X_train_t):
#        remove_i.append(i)
        
#print(len(remove_i))
#print(remove_i)
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

def create_eb(path, s):
    embedding_path = path
    embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))
    embed_size = s

    word_index = tokenizer.word_index
    nb_words = min(max_words, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_words: continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix


fast_text_eb = create_eb("../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec", 300)
fast_text_eb.shape
glove_eb = create_eb("../input/glove840b300dtxt/glove.840B.300d.txt", 300)
glove_eb.shape
def keras_dl(model, eb, embed_size, batch_size, epochs):   
    inp = Input(shape = (max_len,), name = 'lstm')
    #x = Embedding(max_words,embed_size,input_length=max_len)(inp)
    x = Embedding(max_words,embed_size,weights = [eb], trainable = False)(inp)
    #model.add(Embedding(max_words,embed_size,weights = [embedding_matrix], trainable = False))
    x1 = SpatialDropout1D(0.5)(x)
    
    x_lstm = CuDNNLSTM(128, return_sequences = True)(x1)
    x_lstm_c1d = Conv1D(64,kernel_size=3,padding='valid',activation='relu')(x_lstm)
    x_lstm_c1d_gp = GlobalMaxPooling1D()(x_lstm_c1d)
    #x_lstm_c1d_gp = Flatten()(x_lstm_c1d)
    
    #x_c1d = Conv1D(128,kernel_size=3,padding='same',activation='tanh')(x1)
    #x_c1d = MaxPooling1D()(x_c1d)
    #x_c1d_lstm = CuDNNLSTM(64)(x_c1d)
    #x_c1d_gru = CuDNNGRU(64)(x_c1d)
    
    x_gru = CuDNNGRU(128, return_sequences = True)(x1)
    x_gru_c1d = Conv1D(64,kernel_size=2,padding='valid',activation='relu')(x_gru)
    x_gru_c1d_gp = GlobalMaxPooling1D()(x_gru_c1d)
    #x_gru_c1d_gp = Flatten()(x_gru_c1d)
    
    inp2 = Input(shape = (td,), name = 'tfidf')
    x2 = BatchNormalization()(inp2)
    x2 = Dense(16, activation='relu')(x2)
    
    inp3 = Input(shape = (5,), name = 'score')
    x3 = BatchNormalization()(inp3)
    x3 = Dense(3, activation='tanh')(x3)
    
    x_f = concatenate([x_lstm_c1d_gp, x_gru_c1d_gp])#, x_c1d_lstm, x_c1d_gru])
    x_f = BatchNormalization()(x_f)
    x_f = Dropout(0.5)(Dense(128, activation='tanh') (x_f))    
    x_f = BatchNormalization()(x_f)
    x_f = concatenate([x_f, x2, x3])
    x_f = Dropout(0.5)(Dense(32, activation='tanh') (x_f))
    x_f = BatchNormalization()(x_f)
    x_f = Dropout(0.5)(Dense(16, activation='tanh') (x_f))
    #x_f = Dense(5, activation = "sigmoid")(x_f)
    x_f = Dense(5, activation = "softmax")(x_f)
    model = Model(inputs = [inp, inp2, inp3], outputs = x_f)
    
    #model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    #model.compile(loss='hinge',optimizer='adadelta',metrics=['accuracy'])
    print(model.summary())
    return (model)
embed_size = 300
batch_size = 256
epochs = 30
model = Sequential()

file_path1 = "best_model1.hdf5"
check_point = ModelCheckpoint(file_path1, monitor = "val_loss", verbose = 1,
                              save_best_only = True, mode = "min")
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 4)

firstmodel1 = keras_dl(model, fast_text_eb, embed_size, batch_size, epochs)
text_model1 = firstmodel1.fit({'lstm': X_trains, 'tfidf': x_tfidf, 'score': phs2}, y_trains, batch_size=batch_size,epochs=epochs,verbose=0,
                            validation_split = 0.1, #validation_data=(X_val_final,y_val_final), 
                            callbacks = [check_point, early_stop])
firstmodel1 = load_model(file_path1)
embed_size = 300
batch_size = 256
epochs = 30
model = Sequential()

file_path2 = "best_model2.hdf5"
check_point = ModelCheckpoint(file_path2, monitor = "val_loss", verbose = 1,
                              save_best_only = True, mode = "min")
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 4)

firstmodel2 = keras_dl(model, glove_eb, embed_size, batch_size, epochs)
text_model2 = firstmodel2.fit({'lstm': X_trains, 'tfidf': x_tfidf, 'score': phs2}, y_trains, batch_size=batch_size,epochs=epochs,verbose=0,
                            validation_split = 0.1, #validation_data=(X_val_final,y_val_final), 
                            callbacks = [check_point, early_stop])
firstmodel2 = load_model(file_path2)
pred = firstmodel1.predict([np.array(X_test_final), test_tfidf, test_phs2], verbose = 1)
pred_new = firstmodel2.predict([np.array(X_test_final), test_tfidf, test_phs2], verbose = 1)

print(pred.shape)
print(pred_new.shape)


pred_combi = (np.array(pred) + np.array(pred_new)) / 2
pred2 = np.round(np.argmax(pred_combi, axis=1)).astype(int)

print(pred_combi.shape)
print(pred2.shape)

pred2
# Pseudo-labeling
PL_X = np.vstack((X_trains, X_test_final))
PL_tfidf = np.vstack((x_tfidf, test_tfidf))
PL_s = np.vstack((phs2, test_phs2))

y_test = pd.get_dummies(pred2)
PL_y = np.vstack((y_trains, y_test))

print(PL_X.shape)
print(PL_tfidf.shape)
print(PL_s.shape)
print(PL_y.shape)
model = Sequential()
file_path1 = "best_model1.hdf5"
check_point = ModelCheckpoint(file_path1, monitor = "val_loss", verbose = 1,
                              save_best_only = True, mode = "min")
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 4)

finalmodel1 = keras_dl(model, fast_text_eb, embed_size, batch_size, epochs)

final_text_model1 = finalmodel1.fit({'lstm': PL_X, 'tfidf': PL_tfidf, 'score': PL_s}, PL_y, batch_size=batch_size,epochs=epochs,verbose=0,
                                validation_split = 0.1, #validation_data=(X_val_final,y_val_final), 
                                callbacks = [check_point, early_stop])
finalmodel1 = load_model(file_path1)
model = Sequential()

file_path2 = "best_model2.hdf5"
check_point = ModelCheckpoint(file_path2, monitor = "val_loss", verbose = 1,
                              save_best_only = True, mode = "min")
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 4)

finalmodel2 = keras_dl(model, glove_eb, embed_size, batch_size, epochs)

final_text_model2 = finalmodel2.fit({'lstm': PL_X, 'tfidf': PL_tfidf, 'score': PL_s}, PL_y, batch_size=batch_size,epochs=epochs,verbose=0,
                                validation_split = 0.1, #validation_data=(X_val_final,y_val_final), 
                                callbacks = [check_point, early_stop])
finalmodel2 = load_model(file_path2)
final_pred = finalmodel1.predict([np.array(X_test_final), test_tfidf, test_phs2], verbose = 1)
final_pred_new = finalmodel2.predict([np.array(X_test_final), test_tfidf, test_phs2], verbose = 1)

final_pred_combi = (final_pred + final_pred_new) / 2
final_pred2 = np.round(np.argmax(final_pred_combi, axis=1)).astype(int)
sub = pd.DataFrame({'PhraseId': test_f['PhraseId'],
                   'Sentiment': final_pred2})

sub.to_csv("DL.csv", index = False, header = True)
sub.groupby('Sentiment').agg({'PhraseId': 'count'})