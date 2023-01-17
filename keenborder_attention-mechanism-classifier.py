import pandas as pd

import nltk

import numpy as np

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize 

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.naive_bayes import GaussianNB,MultinomialNB

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from sklearn.metrics import roc_curve,classification_report

from matplotlib import style

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import LSTM,Input,TimeDistributed,Reshape

from tensorflow.keras.layers import Embedding,Dropout,Bidirectional,Dot

import tensorflow

import string

style.use("ggplot")
df_train = pd.read_csv('../input/nlp-getting-started/train.csv', dtype={'id': np.int16, 'target': np.int8})

df_test = pd.read_csv('../input/nlp-getting-started/test.csv', dtype={'id': np.int16})
df_train.isnull().sum()
df_train.head()
train_tweets=df_train["text"]
train_tweets.shape
train_tweets_lower=[sentence.lower() for sentence in train_tweets]
plt.bar(["Real Disaster","Not Real Disaster"],df_train['target'].value_counts().tolist())
stop_words = set(stopwords.words('english'))

filter_corpus=[]

ps =PorterStemmer()



for sentence in train_tweets_lower:

    filter_sentence=[]

    word_sentence=word_tokenize(sentence)

    for word in word_sentence:

        if word not in stop_words and word.isalpha():

            table = str.maketrans('', '', string.punctuation)

            word=word.translate(table)

            rootWord=ps.stem(word)

            filter_sentence.append(rootWord)

    filter_corpus.append(filter_sentence)



filter_corpus_text=[]

for sentence in filter_corpus:

    filter_corpus_text.append(' '.join(sentence))

filter_corpus_text[:2]            
def Naive_Bayes(feature_extraction,grid_search=False,roc=False):

    NB=MultinomialNB()

    if feature_extraction=="TD-IDF":

        tdidf=TfidfVectorizer()

        X_td=tdidf.fit_transform(filter_corpus_text).toarray()

        y_td=np.array(df_train["target"])

        X_train,X_test,y_train,y_test=train_test_split(X_td,y_td,test_size=0.1)

    else:

        CV=CountVectorizer()

        X_cv=CV.fit_transform(filter_corpus_text).toarray()

        y_cv=df_train["target"]

        X_train,X_test,y_train,y_test=train_test_split(X_cv,y_cv,test_size=0.1)

    if grid_search:

            grid_search=GridSearchCV(estimator=NB,param_grid=({'alpha':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0

                                                                             ],'fit_prior':[True,False]}))

            grid_search.fit(X_train,y_train)

            print(grid_search.best_estimator_)

    if roc:

        NB=MultinomialNB(alpha=0.9, class_prior=None, fit_prior=True)

        NB.fit(X_train,y_train)

        y_test_prob=NB.predict_proba(X_test)

        y_test_pred=NB.predict(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:,1])

        ax=plt.subplot(111)

        ax.plot(fpr,tpr)

        ax.set_ylabel("True Positive Rate")

        ax.set_xlabel("False Positive Rate")

        ax.set_title("Naive Bayes(Multinomial) ROC with {}".format(feature_extraction))

        print("The Classification Report for Naive--Bayes with {}".format(feature_extraction))

        print(classification_report(y_test,y_test_pred))

        

        ax.plot([0,1],[0,1],color="black")
Naive_Bayes("TD-IDF",grid_search=False,roc=True)  
Naive_Bayes("CountVectorizer",grid_search=False,roc=True)
# integer encode sequences of words

tokenizer = Tokenizer()

tokenizer.fit_on_texts(filter_corpus_text)

sequences = tokenizer.texts_to_sequences(filter_corpus_text)

# vocabulary size

vocab_size = len(tokenizer.word_index) + 1

sequences = np.array(sequences)

sequences=pad_sequences(sequences, dtype='int32', padding='pre')
sequences.shape
y=df_train["target"]

y_one_hot=to_categorical(y)

X_train,X_test,y_train,y_test=train_test_split(sequences,y_one_hot,test_size=0.1)
def Attention_Mechanism():##This function is the Main Neural Network made through Keras

    """"

    Main LSTM Network Created through Keras

    """



    inputs = Input(shape=(X_train.shape[1],))

    embedding = Embedding(vocab_size, 300, input_length=X_train.shape[1], trainable=True)(inputs)

    

    # Apply dropout to prevent overfitting

    embedded_inputs = Dropout(0.2)(embedding)

    

    # Apply Bidirectional LSTM over embedded inputs

    lstm_outs =Bidirectional(

        LSTM(300, return_sequences=True)

    )(embedded_inputs)

    

    # Apply dropout to LSTM outputs to prevent overfitting

    lstm_outs = Dropout(0.2)(lstm_outs)

    

    # Attention Mechanism - Generate attention vectors

    attention_vector = TimeDistributed(Dense(1))(lstm_outs)

    attention_vector = Reshape((X_train.shape[1],))(attention_vector)

    attention_vector = tensorflow.keras.layers.Activation('softmax', name='attention_vec')(attention_vector)

    attention_output = Dot(axes=1)([lstm_outs, attention_vector])

    

    # Last layer: fully connected with softmax activation

    fc = Dense(300, activation='relu')(attention_output)

    output = tensorflow.keras.layers.Dense(2, activation='softmax')(fc)

    

    model=tensorflow.keras.Model(inputs,output)

    

    model.compile(loss='categorical_crossentropy',

              optimizer='adam',metrics=['acc'])

        

    print(model.summary())

    early_stopping_cb = tensorflow.keras.callbacks.EarlyStopping(patience=4,restore_best_weights=True)#early stopping

    

    checkpoint_cb = tensorflow.keras.callbacks.ModelCheckpoint("Attention_Mechanism.h5",

                                                    save_best_only=True) #checkpoint for saving the model

    

    

    history=model.fit(X_train,y_train,validation_data=(X_test,y_test),callbacks=[checkpoint_cb,

                      early_stopping_cb],epochs=10,verbose=2)

    

    return history

history=Attention_Mechanism()
pd.DataFrame(history.history)[["loss",'val_loss']].plot(figsize=(8, 5))

plt.grid(True)

plt.title(f"Loss of Deep LSTM Model with GloveVec Embedding")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.gca().set_ylim(0, 1) # set the vertical range
attention_mechanism=load_model("Attention_Mechanism.h5")

y_test_prob=attention_mechanism.predict(X_test)

y_test_pred=np.argmax(attention_mechanism.predict(X_test),axis=1)

fpr, tpr, thresholds = roc_curve(np.argmax(y_test,axis=1), y_test_prob[:,1])

ax=plt.subplot(111)

ax.plot(fpr,tpr)

ax.set_ylabel("True Positive Rate")

ax.set_xlabel("False Positive Rate")

ax.set_title("Attention Mechanim ROC")

print("The Classification Report for Attention-Mechanism")

print(classification_report(np.argmax(y_test,axis=1),y_test_pred))

ax.plot([0,1],[0,1],color="black")
test_tweets=df_test["text"]

test_tweets_lower=[sentence.lower() for sentence in test_tweets]

stop_words = set(stopwords.words('english'))

filter_corpus=[]

ps =PorterStemmer()



for sentence in test_tweets_lower:

    filter_sentence=[]

    word_sentence=word_tokenize(sentence)

    for word in word_sentence:

        if word not in stop_words and word.isalpha():

            table = str.maketrans('', '', string.punctuation)

            word=word.translate(table)

            rootWord=ps.stem(word)

            filter_sentence.append(rootWord)

    filter_corpus.append(filter_sentence)



filter_corpus_text=[]

for sentence in filter_corpus:

    filter_corpus_text.append(' '.join(sentence))

sequences_test = tokenizer.texts_to_sequences(filter_corpus_text)    

sequences_test=pad_sequences(sequences_test, maxlen=22,dtype='int32', padding='pre')

final_prediction=np.argmax(attention_mechanism.predict(sequences_test),axis=1)

submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

submission["target"]=final_prediction

submission.to_csv('submission.csv', index=False)