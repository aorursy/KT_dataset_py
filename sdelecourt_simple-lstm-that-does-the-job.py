import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
                    
import os
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score, precision_score, recall_score


from keras.layers import Input
from keras import Model
from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import Callback

import nltk
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer,WordNetLemmatizer
stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()
from string import punctuation

import re
import os
import gc

import matplotlib.pyplot as plt
print(os.listdir("../input"))
df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
print("Columns :", df.columns)
print("Row 0 :")
print("qid :", df.iloc[0]['qid'])
print("question_text : ", df.iloc[0]['question_text'])
print("target :", df.iloc[0]['target'])
n_rows = len(df)
n_insincere = sum(df['target'])
print(n_rows)
print(n_insincere)

label_repart = pd.DataFrame(data={"" :[n_rows - n_insincere, n_insincere]}, index = [str(n_rows - n_insincere) + ' sincere questions', str(n_insincere) + ' insincere question'])
label_repart.plot(kind='pie', title='Labels repartition, ratio ' + str(round(n_insincere / n_rows,2)*100) + "%", subplots=True, figsize=(8,8))
insincere_question = df[df['target'] == 1]['question_text'].values
for i in range(10):
    print(insincere_question[i])
def clean_review(review_col):
    review_corpus=[]
    stops = set(stopwords.words("english"))
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z]',' ',review)
        word_token = word_tokenize(str(review).lower())
        #review = [word for word in word_token if word not in stops]
        #review=' '.join(review)
        review=[lemma.lemmatize(w) for w in word_token if w not in stops]
        review=' '.join(review)
        review_corpus.append(review)
        #if i % 1000 == 0:
           #print(i/len(review_col)) 
    return review_corpus
df['clean_question']=clean_review(df['question_text'].values)
df_test['clean_question']=clean_review(df_test['question_text'].values)
#all_words=' '.join(df['clean_question'].values)
#all_words=word_tokenize(all_words)
#dist=FreqDist(all_words)
#num_unique_word=len(dist)
num_unique_word = 166289
df.head()
#r_len=[]
#for text in df['clean_question'].values:
#    word=word_tokenize(text)
#    l=len(word)
#    r_len.append(l)    
#MAX_QUESTION_LEN=np.max(r_len)
MAX_QUESTION_LEN=125
MAX_FEATURES = num_unique_word
MAX_WORDS = MAX_QUESTION_LEN
y_train = df['target'].values
X_train_text = df['clean_question'].values
X_test_text = df_test['clean_question'].values
print(X_train_text.shape,y_train.shape)
print(X_test_text.shape)
X_train_text, X_val_text, y_train, y_val = train_test_split(X_train_text, y_train, test_size=0.1)
tokenizer = Tokenizer(num_words=MAX_FEATURES)
tokenizer.fit_on_texts(list(X_train_text))
X_train = tokenizer.texts_to_sequences(X_train_text)
X_val = tokenizer.texts_to_sequences(X_val_text)
X_test = tokenizer.texts_to_sequences(X_test_text)
X_train = sequence.pad_sequences(X_train, maxlen=MAX_WORDS)
X_val = sequence.pad_sequences(X_val, maxlen=MAX_WORDS)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_WORDS)
print(X_train.shape,X_val.shape)
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
    
def get_embed_mat(EMBEDDING_FILE, max_features,embed_dim):
    # word vectors
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding='utf8'))
    print('Found %s word vectors.' % len(embeddings_index))

    # embedding matrix
    word_index = tokenizer.word_index
    num_words = min(max_features, len(word_index) + 1)
    all_embs = np.stack(embeddings_index.values()) #for random init
    embedding_matrix = np.zeros((len(word_index) + 1, embed_dim))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    max_features = embedding_matrix.shape[0]
    
    return embedding_matrix

gloveEmbed = get_embed_mat('../input/embeddings/glove.840B.300d/glove.840B.300d.txt', MAX_FEATURES, 300)
word_index = tokenizer.word_index
embedding_layer = Embedding(len(word_index) + 1,
                            300,
                            weights=[gloveEmbed],
                            input_length=MAX_WORDS,
                            trainable=False)
def line_search_f1_score(y_score, y_test):
    max_f1_score = 0
    opt_threshold = 0
    for threshold in [i*0.01 for i in range(100)]:
        y_preds = y_score > threshold
        score = f1_score(y_preds, y_test)
        if max_f1_score < score:
            max_f1_score = score
            opt_threshold = threshold
    return max_f1_score, opt_threshold
class Metrics(Callback):
    def __init__(self):
        self.best_threshold = 0.5
        self.best_f1_score = 0
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.best_f1_score = 0
    def on_epoch_end(self, epoch, logs={}):
         idx = np.random.randint(0,self.validation_data[0].shape[0],1000)
         val_predict = (np.asarray(self.model.predict(self.validation_data[0][idx], verbose=1))).round()
         val_targ = self.validation_data[1][idx]
         #_val_f1 = f1_score(val_targ, val_predict)
         _val_f1, threshold = line_search_f1_score(val_targ, val_predict)
         if _val_f1 > self.best_f1_score:
                self.best_f1_score = _val_f1
         self.best_threshold = threshold
         _val_recall = recall_score(val_targ, val_predict)
         _val_precision = precision_score(val_targ, val_predict)
         self.val_f1s.append(_val_f1)
         self.val_recalls.append(_val_recall)
         self.val_precisions.append(_val_precision)
         print(" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
         return
 
metric = Metrics()
lstm_out = 200

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=2, batch_size=1024, verbose=1,callbacks=[metric])
y_score_test = model.predict(X_val, verbose=1)
max_f1_score, threshold = line_search_f1_score(y_score_test, y_val)
y_sub = model.predict(X_test, verbose = 1)
sub = pd.read_csv('../input/sample_submission.csv')
sub.prediction = np.array(y_sub > threshold, dtype=int) 
sub.to_csv("submission.csv", index=False)
# Best f1_score on validation dataset :
print(threshold)
print(max_f1_score)