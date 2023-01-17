# imports 

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 



import nltk

from nltk.corpus import stopwords



import spacy



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import MultinomialNB



# word2vec

import gensim



# 

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import  Dense, Dropout, Embedding, LSTM

from tensorflow.keras import utils 

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping



# utility

import re 

import os 

import pickle 

from tqdm import tqdm



%matplotlib inline 
nltk.download('stopwords')
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]



# TEXT CLENAING

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"



df = pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv',

                 encoding='latin',names=DATASET_COLUMNS)

df.head()
# SENTIMENT

POSITIVE = "Positive"

NEGATIVE = "Negative"

NEUTRAL = "Neutral"

decode_label = {0:"Negative",2:"Neutral",4:"Positive"}



def decode_sentiment(label):

    return decode_label[int(label)]
df.target = df.target.map(decode_sentiment)

df.head()
sns.catplot(kind='count',data=df,x='target',aspect=2)

plt.show()
stop_words = stopwords.words('english')

nlp = spacy.load('en_core_web_sm',disable=['tagger','parser','ner','textcat'])
# TEXT CLENAING

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"



def preprocess(text):

    text = re.sub(TEXT_CLEANING_RE,' ',str(text).lower()).strip()

    txt = nlp(text)

    rel_tok = " ".join([tok.lemma_.lower() for tok in txt if tok.lemma_.lower() not in stop_words])

    return rel_tok

        

            
tqdm.pandas()

processed_text = df.text.progress_apply(preprocess)

df.processed_text = processed_text
df['processed_text'] = processed_text

df.head()
# split 

df.drop('text',inplace=True,axis=1)

y = df['target']

X = df.drop('target',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
# ml models 

tfidf = TfidfVectorizer()

X_tr = tfidf.fit_transform(X_train['processed_text'])

X_te = tfidf.transform(X_test['processed_text'])
# logistic regression

clf = LogisticRegression(max_iter=200,n_jobs=-1)

clf.fit(X_tr,y_train)

preds = clf.predict(X_te)

print('accuracy score: {:.4f}'.format(accuracy_score(y_test,preds)))

confusion_matrix(y_test,preds)
# Linear svm classifier 

svc = LinearSVC()

svc.fit(X_tr,y_train)

preds = svc.predict(X_te)

print('accuracy score: {:.4f}'.format(accuracy_score(y_test,preds)))

confusion_matrix(y_test,preds)
# Multinomial classifier 

nbc = MultinomialNB()

nbc.fit(X_tr,y_train)

preds = nbc.predict(X_te)

print('accuracy score: {:.4f}'.format(accuracy_score(y_test,preds)))

confusion_matrix(y_test,preds,)
%%time 

# Word2 vec

docs = [txt.split() for txt in X_train.processed_text]
W2V_SIZE = 300

W2V_WINDOW = 7

W2V_MIN_COUNT = 10

W2V_EPOCH = 32

w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE,

                                          window=W2V_WINDOW,

                                          min_count=W2V_MIN_COUNT,

                                          workers=8)
w2v_model.build_vocab(docs)
words = w2v_model.wv.vocab.keys()

vocab_size = len(words)

print('vocab size',vocab_size)
%%time 

w2v_model.train(docs,total_examples=len(docs),epochs=W2V_EPOCH)
w2v_model.most_similar("love")
%%time 

# Tokenization 

tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train.processed_text)



vocab_size = len(tokenizer.word_index)+1

print('total words',vocab_size)
%%time 

# padding

SEQUENCE_LENGTH = 300

x_train = pad_sequences(tokenizer.texts_to_sequences(X_train.processed_text),maxlen=SEQUENCE_LENGTH)

x_test = pad_sequences(tokenizer.texts_to_sequences(X_test.processed_text),maxlen=SEQUENCE_LENGTH)
# label encoder 

labels = y_train.unique().tolist()

labels.append(NEUTRAL)

labels

encoder = LabelEncoder()

encoder.fit(y_train.tolist())



y_train = encoder.transform(y_train.tolist())

y_test = encoder.transform(y_test.tolist())



y_train = y_train.reshape(-1,1)

y_test = y_test.reshape(-1,1)



print("y_train",y_train.shape)

print("y_test",y_test.shape)
## Embedding layer 

embedding_matrix = np.zeros((vocab_size,W2V_SIZE))

for word, i in tokenizer.word_index.items():

    if word in w2v_model.wv:

        embedding_matrix[i] = w2v_model.wv[word]

print(embedding_matrix.shape)
embedding_layer = Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], 

                            input_length=SEQUENCE_LENGTH, trainable=False)
## Model 

def create_model():

    return Sequential([

        embedding_layer,

        Dropout(0.5),

        LSTM(60,dropout=0.2,recurrent_dropout=0.2),

        Dense(1,activation='sigmoid')

    ])
model = create_model()



# compile model 

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])



# callbacks 

callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),

              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]
%%time 

EPOCHS = 4

BATCH_SIZE = 1024



history = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs=EPOCHS,

                   validation_split=.1,verbose=1,callbacks=callbacks)
def plot(measure):

    train_measure = history.history[measure]

    val_measure = history.history[f'val_{measure}']

    epochs = np.arange(len(train_measure))

    plt.plot(epochs,train_measure,'b')

    plt.plot(epochs,val_measure,'r')

    plt.title(f'Training & Validation {measure}')

    plt.xlabel('Epoch')

    plt.ylabel(f'{measure}')

    plt.legend(['Training','Validation'])

    plt.show()

    
plot('accuracy')
plot('loss')
# evaluation 

model.evaluate(x_test,y_test,batch_size=BATCH_SIZE)
SENTIMENT_THRESHOLDS = (0.4, 0.7)

def decode_sentiment(score, include_neutral=True):

    if include_neutral:        

        label = NEUTRAL

        if score <= SENTIMENT_THRESHOLDS[0]:

            label = NEGATIVE

        elif score >= SENTIMENT_THRESHOLDS[1]:

            label = POSITIVE



        return label

    else:

        return NEGATIVE if score < 0.5 else POSITIVE
def predict(text, include_neutral=True):

    # Tokenize text

    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)

    # Predict

    score = model.predict([x_test])[0]

    # Decode sentiment

    label = decode_sentiment(score, include_neutral=include_neutral)



    return {"label": label, "score": float(score)} 
predict("I love the music")
predict("i don't know what i'm doing")