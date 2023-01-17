import datetime
import pandas as pd
import numpy as np
import random
import time

# Data Viz
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
from yellowbrick.text import TSNEVisualizer

# Hide Warnings
Warning = True
if Warning is False:
    import warnings
    warnings.filterwarnings(action='ignore')
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    warnings.filterwarnings(action='ignore', category=FutureWarning)

#Modeling 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from keras.preprocessing import text, sequence
    

np.random.seed(2018)
train = pd.read_csv("../input/train.csv", index_col= 'qid')#.sample(50000)
test = pd.read_csv("../input/test.csv", index_col= 'qid')#.sample(5000)
testdex = test.index

target_names = ["Sincere","Insincere"]
y = train['target'].copy()
print(train.shape)
train.head()
print("Distribution of Classes:")
train.target.value_counts(normalize=False)
all_text = pd.concat([train['question_text'],test['question_text']], axis =0)

word_vect = TfidfVectorizer(
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(1, 2),
            max_features=20000)
word_vect.fit(all_text)
X  = word_vect.transform(train['question_text'])
testing  = word_vect.transform(test['question_text'])
X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=123, stratify=y)
# Create the visualizer and draw the vectors
plt.figure(figsize = [15,9])
tsne = TSNEVisualizer()
n = 20000
tsne.fit(X_train[:n], train.target[:n].map({1: target_names[1],0:target_names[0]}))
tsne.poof()
def model_fit(model,X_train,X_val,y_train,y_val):
    model.fit(X_train, y_train)
    
    # Predict
    valid_logistic_pred = model.predict(X_val)
    train_logistic_pred = model.predict(X_train)
    
    print("Train Set F1 Score: {:.3f}".format(metrics.f1_score(train_logistic_pred, y_train)))
    print("Validation Set F1 Score: {:.3f}".format(metrics.f1_score(valid_logistic_pred, y_val)))

    # Confusion Matrix
    C = metrics.confusion_matrix(valid_logistic_pred, y_val)/len(y_val)
    sns.heatmap(C, annot=True)
# Fit Model
model_fit(LogisticRegression(solver = 'sag'),X_train,X_valid,y_train,y_valid)
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def model_val(model,X_valid,y_valid):
    preds = model.predict([X_valid],batch_size=1024,verbose = True)
    for i in np.arange(.1,.6,.025):
        i = np.round(i, 3)
        score = metrics.f1_score(y_valid,(preds > i))
        print("F1 score at threshold {0} is {1}".format(i, score))
from keras.models import Model, Sequential
from keras.layers import CuDNNGRU,CuDNNLSTM,Input, Dense, Embedding,Dropout, concatenate, Bidirectional,Flatten,GlobalAveragePooling1D, GlobalMaxPool1D
maxlen = 100
max_features = 20000

inp = Input((max_features,))
hidden1 = Dense(units = maxlen)(inp)
hidden2 = Dense(units = maxlen)(hidden1)

final = Dense(units = 1,activation = 'sigmoid')(hidden2)

model = Model(inp,final)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[f1])
model.summary()
model.fit(X_train,y_train,batch_size = 1024,epochs = 2,validation_data = (X_valid,y_valid),verbose = True)
model_val(model,X_valid,y_valid)
## some config values 
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use

## split to train and val
train_df, val_df = train_test_split(train, test_size=0.1, random_state=2018)

## fill up the missing values
train_X = train_df["question_text"].fillna("_na_").values
val_X = val_df["question_text"].fillna("_na_").values
test_X = test["question_text"].fillna("_na_").values

## Tokenize the sentences
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
X_train = tokenizer.texts_to_sequences(train_X)
X_valid = tokenizer.texts_to_sequences(val_X)
X_test = tokenizer.texts_to_sequences(test_X)

## Pad the sentences 
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_valid = sequence.pad_sequences(X_valid, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

## Get the target values
y_train = train_df['target'].values
y_valid = val_df['target'].values
inp = Input(shape=(maxlen,))
emb = Embedding(max_features, embed_size)(inp)

hidden1 = Dense(units = 64)(emb)
flat = Flatten()(hidden1)
hidden2 = Dense(units = 16,activation = 'relu')(flat)

final = Dense(units = 1,activation = 'sigmoid')(hidden2)

model = Model(inp,final)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[f1])
model.fit(X_train, y_train, batch_size=1024, epochs=2, validation_data=(X_valid, y_valid),verbose = True)
model_val(model,X_valid,y_valid)
inp = Input(shape=(maxlen,))
emb = Embedding(max_features, embed_size)(inp)

hidden1 = Dense(units = 64)(emb)
max_pool = GlobalMaxPool1D()(hidden1)
avg_pool = GlobalAveragePooling1D()(hidden1)
conc = concatenate([max_pool,avg_pool])

hidden2 = Dense(units = 16,activation = 'relu')(conc)

final = Dense(units = 1,activation = 'sigmoid')(hidden2)

model = Model(inp,final)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[f1])
model.fit(X_train, y_train, batch_size=1024, epochs=2, validation_data=(X_valid, y_valid),verbose = True)
model_val(model,X_valid,y_valid)
inp = Input(shape=(maxlen,))
emb = Embedding(max_features, embed_size)(inp)

hidden1 = Bidirectional(CuDNNLSTM(units = 64,return_sequences = True))(emb)
max_pool = GlobalMaxPool1D()(hidden1)
avg_pool = GlobalAveragePooling1D()(hidden1)
conc = concatenate([max_pool,avg_pool])

hidden2 = Dense(units = 16,activation = 'relu')(conc)

final = Dense(units = 1,activation = 'sigmoid')(hidden2)

model = Model(inp,final)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[f1])
model.summary()
model.fit(X_train, y_train, batch_size=1024, epochs=2, validation_data=(X_valid, y_valid),verbose = True)
model_val(model,X_valid,y_valid)
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
inp = Input(shape=(maxlen,))
emb = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)

hidden1 = Bidirectional(CuDNNLSTM(units = 64,return_sequences = True))(emb)
max_pool = GlobalMaxPool1D()(hidden1)
avg_pool = GlobalAveragePooling1D()(hidden1)
conc = concatenate([max_pool,avg_pool])

hidden2 = Dense(units = 16,activation = 'relu')(conc)
drop = Dropout(0.1)(hidden2)
final = Dense(units = 1,activation = 'sigmoid')(drop)

model = Model(inp,final)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[f1])
model.summary()
model.fit(X_train, y_train, batch_size=1024, epochs=2, validation_data=(X_valid, y_valid),verbose = True)
model_val(model,X_valid,y_valid)
pred_test_y = model.predict(X_test,batch_size = 1024,verbose = True)
pred_test_y = (pred_test_y>0.35).astype(int)
out_df = pd.DataFrame({"qid":test.index.values})
out_df['prediction'] = pred_test_y
out_df.to_csv("submission.csv", index=False)