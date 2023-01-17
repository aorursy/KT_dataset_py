import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn import metrics
import nltk
import os
import gc
from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Input, Dense,Dropout,Embedding,LSTM, CuDNNGRU, Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,GlobalMaxPool1D,SpatialDropout1D,Bidirectional
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
from nltk.corpus import stopwords
import string
punctuations = string.punctuation
stopword = stopwords.words("english")
def clean(text):
    
    lower_text = text.lower()
    
    text = "".join(w for w in lower_text if w not in punctuations)
    
    words = text.split()
    words = [w for w in words if w not in stopword]
    res = " ".join(words)
    return res
clean("this is a test!")
train['cleaned'] = train['question_text'].apply(clean)
test['cleaned'] = test['question_text'].apply(clean)
target = to_categorical(train['target']) 
#target = train['label']
x_train, x_val, y_train, y_val  = train_test_split(train['cleaned'], target, test_size=0.2, random_state=1)
words = ' '.join(x_train)
words = nltk.word_tokenize(words)
dist = nltk.FreqDist(words)
num_unique_words = len(dist)
r_len = []
for w in x_train:
    word=nltk.word_tokenize(w)
    l=len(word)
    r_len.append(l)
max_len = np.max(r_len)
max_len
max_features = num_unique_words
max_words = max_len
batch_size = 128
embed_dim = 300

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train))
x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
x_test = tokenizer.texts_to_sequences(test['cleaned'])
x_train = sequence.pad_sequences(x_train, maxlen=max_words)
x_val = sequence.pad_sequences(x_val, maxlen=max_words)
x_test = sequence.pad_sequences(x_test, maxlen=max_words)
print(x_train.shape,x_val.shape,x_test.shape)
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
ros = RandomUnderSampler(random_state=777)
X_ROS, y_ROS = ros.fit_sample(x_train, y_train)
#x_train = X_ROS
#y_train = y_ROS
EMBEDDING_FILE =open("../input/embeddings/glove.840B.300d/glove.840B.300d.txt", encoding="utf8") 

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in EMBEDDING_FILE)

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
inp = Input(shape=(max_words,))
x = Embedding(max_features, embed_dim, weights=[embedding_matrix])(inp)
x = Bidirectional(GRU(64,return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(2, activation="softmax")(x)
model_2 = Model(inputs=inp, outputs=x)
model_2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_2.summary())
model_2.fit(x_train, y_train, batch_size=512, epochs=1, validation_data=(x_val, y_val))
pred=np.round(np.clip(model_2.predict(x_val), 0, 1))
print(f1_score(y_val, pred, average = None))
pred_2=np.round(np.clip(model_2.predict(x_test), 0, 1)).astype(int)
pred_2 = pd.DataFrame(pred_2)
pred_2 = pred_2.idxmax(axis=1)
submission = pd.DataFrame({'qid':test['qid'], 'prediction':pred_2})
submission.to_csv("submission.csv", index=False)

