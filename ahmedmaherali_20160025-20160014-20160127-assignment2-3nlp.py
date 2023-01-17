import gensim
import gensim.models.keyedvectors as word2vec
word2vecDict = word2vec.KeyedVectors.load_word2vec_format("../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary=True)
#embedmodel = gensim.models.Word2Vec(tweets)
import pandas as pd
def read_data():
    df=pd.read_csv('/kaggle/input/twitterdata/Tweets.csv')
    df.sort_values(by=['airline_sentiment'],inplace=True)
    df.dropna(subset=["text"],axis=0,inplace=True)
    df.dropna(subset=["airline_sentiment"],axis=0,inplace=True)
    
    df.reset_index(drop=True,inplace=True)
    
    df = df.iloc[6400:,]
    
    df.reset_index(inplace=True)
    print(len(df))
    tweets=df['text']
    labels=[]
    for l in df.airline_sentiment:
        if(l=='neutral'):
            labels.append(1)
        elif (l=='negative'):
            labels.append(0)
        else:
            labels.append(2)
    
    return tweets,labels
tweets,labels=read_data()
tweets
import re
import nltk
import string
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lem=WordNetLemmatizer()

def process_data(tweets,labels):
    processedTweets=[]
    newData=[]
    lbls=[]
    i=0
    for tweet in tweets:
        s=""
        s+=(tweet.lower()+' ')
        s = re.sub("(@\w* )", ' ', s)
        s = re.sub("\\bhttps://(.*) \\b",' ',s)
        s = re.sub("[^a-z0-9\ ]+", ' ', s)
        s = re.sub(' \d+', ' ', s)
        s = re.sub('\#(.*) ',' ',s)
        s = re.sub(" +",' ',s)
        
        word_tokens = word_tokenize(s)
        s=""
        for w in word_tokens :
            if not w in stop_words:
                s+=(lem.lemmatize(w,"v")+' ')
                
        if(len(s)>1):
            lbls.append(labels[i])
            newData.append(s)
        i+=1
        
    #lbls=np.array(lbls)
    return newData,lbls
def avgSentenceEmbedding(text, size=300):
    text=text.split()
    
    vec = np.zeros(size).reshape( size)
    
    count = 0
    
    for word in text:
        try:
            vec += word2vecDict[word].reshape( size)
        except KeyError:
            continue
        count += 1
    if count != 0:
        vec /= count
    return vec
from sklearn.model_selection import train_test_split

tweets,labels=read_data()
tweets,labels=process_data(tweets,labels)

x_train,x_test,y_train,y_test=train_test_split(tweets, labels, test_size=0.2,random_state=42)

x_train_recsords=[]

for r in x_train:
    x_train_recsords.append(avgSentenceEmbedding(r))
    
x_test_recsords=[]
for r in x_test:
    x_test_recsords.append(avgSentenceEmbedding(r))
    
"""SVM Model"""
from sklearn import svm,metrics
from sklearn.metrics import accuracy_score

clf = svm.SVC(gamma='scale')


fittedModel = clf.fit(x_train_recsords, y_train)


predictions = fittedModel.predict(x_test_recsords)

print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))
import pickle
filename = 'MostACsntimentSvmModel.sav'
pickle.dump(clf, open(filename, 'wb'))

# some time later...

# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))

"""Logistic Regression Model"""
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


model = LogisticRegression(solver='newton-cg', C=1e7)


model.fit(x_train_recsords, y_train)
predictions = model.predict(x_test_recsords)
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))


import pickle
filename='sntimentLogisticModel.sav'

#loaded_model = pickle.load(open(filename, 'rb'))

def process_review(rev):
    s=""
    s+=(rev.lower()+' ')
    s = re.sub("(@\w* )", ' ', s)
    s = re.sub("\\bhttps://(.*) \\b",' ',s)
    s = re.sub("[^a-z0-9\ ]+", ' ', s)
    s = re.sub(' \d+', ' ', s)
    s = re.sub(" +",' ',s)
    word_tokens = word_tokenize(s)
    s=""
    for w in word_tokens :
        if not w in stop_words:
            s+=(lem.lemmatize(w,"v")+' ')

    return s
def predict(review):
    smbls=dict()
    smbls[0]="Negative"
    smbls[1]="Neutral"
    smbls[2]="Positive"
    review=process_review(review)
    review=(avgSentenceEmbedding(review))
    data=[]
    data.append(review)

    predictions=clf.predict(data)

    return smbls[predictions[0]]

print(predict("fucking bad journey"))
print(predict("I had a very good trip "))
print(predict("I am so happy with this result "))
print(predict("the result was really sucks"))
print(predict("i am so sad for this bad services we got"))
print(predict("amazing journy thanks for it"))
print(predict("could you do it again please"))
print(predict("let's play the match"))
print(predict("I Don't know why nothing works"))
review=input("Entire your review")
print(predict(review))




import gensim
import gensim.models.keyedvectors as word2vec

import pandas as pd
#word2vecDict = word2vec.KeyedVectors.load_word2vec_format("../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary=True)
#embedmodel = gensim.models.Word2Vec(tweets)


def read_data():
    df=pd.read_csv('/kaggle/input/twitterdata/Tweets.csv')
    df.sort_values(by=['airline_sentiment'],inplace=True)
    df.dropna(subset=["text"],axis=0,inplace=True)
    df.dropna(subset=["airline_sentiment"],axis=0,inplace=True)
    df.reset_index(drop=True,inplace=True)
    
    df = df.iloc[6400:,]
    df.reset_index(inplace=True)
    print(len(df))
    tweets=df['text']
    labels=[]
    for l in df.airline_sentiment:
        if(l=='neutral'):
            labels.append(1)
        elif (l=='negative'):
            labels.append(0)
        else:
            labels.append(2)
    
    return tweets,labels

import re
import nltk
import string
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lem=WordNetLemmatizer()

def process_data(tweets,labels):
    processedTweets=[]
    newData=[]
    lbls=[]
    i=0
    for tweet in tweets:
        s=""
        s+=(tweet.lower()+' ')
        s = re.sub("(@\w* )", ' ', s)
        s = re.sub("\\bhttps://(.*) \\b",' ',s)
        s = re.sub("[^a-z0-9\ ]+", ' ', s)
        s = re.sub(' \d+', ' ', s)
        s = re.sub('\#(.*) ',' ',s)
        s = re.sub(" +",' ',s)
        
        word_tokens = word_tokenize(s)
        s=""
        for w in word_tokens :
            if not w in stop_words:
                s+=(lem.lemmatize(w,"v")+' ')
        if(len(s)>1):
            lbls.append(labels[i])
            newData.append(s)
        i+=1
        
    #lbls=np.array(lbls)
    return newData,lbls

def avgSentenceEmbedding(text, size=300):
    text=text.split()
    vec = np.zeros(size).reshape(size)
    count = 0
    for word in text:
        try:
            vec += word2vecDict[word].reshape( size)
        except KeyError:
            continue
        count += 1
    if count != 0:
        vec /= count
    return vec

def sumSentenceEmbedding(text, size=300):
    text=text.split()
    vec = np.zeros(size).reshape(size)
    for word in text:
        try:
            vec += word2vecDict[word].reshape( size)
        except KeyError:
            continue
    return vec

import tensorflow
from tensorflow import keras
from sklearn.model_selection import train_test_split


tweets,labels=read_data()
tweets,labels=process_data(tweets,labels)
x_train,x_test,y_train,y_test=train_test_split(tweets, labels, test_size=0.1,random_state=42)

x_train_recsords=[]    
x_test_recsords=[]
choice="2"#input("Enter 2 for sumSentenceEmbedding or anything else for avgSentenceEmbedding")
if choice == "2":
    for r in x_train:
        x_train_recsords.append(sumSentenceEmbedding(r))
    for r in x_test:
        x_test_recsords.append(sumSentenceEmbedding(r))
else :
    for r in x_train:
        x_train_recsords.append(avgSentenceEmbedding(r))
    for r in x_test:
        x_test_recsords.append(avgSentenceEmbedding(r))
print("Sentences processing and embbeding has finished..")
x_train_recsords=np.array(x_train_recsords)
x_test_recsords =np.array(x_test_recsords)
print(x_train_recsords.shape)
x_train_recsords=np.reshape(x_train_recsords,(x_train_recsords.shape[0],1,x_train_recsords.shape[1]))
x_test_recsords=np.reshape(x_test_recsords,(x_test_recsords.shape[0],1,x_test_recsords.shape[1]))
print(x_train_recsords.shape)
from keras.utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
print(y_train.shape)
print(y_train.shape)
print("Train and test data has prepeared..")


from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, LSTM, Embedding, Dropout, Activation, Bidirectional
from keras.models import Sequential

print(x_train_recsords.shape,y_train.shape)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train_recsords.shape[1],x_train_recsords.shape[2])))
model.add(Dropout(0.5))
model.add(LSTM(units=15))
model.add(Dropout(0.3))
model.add(Dense(units=3,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(x_train_recsords,y_train,
                    epochs=15,
                    verbose=1,
                    )
model.summary()
loss,acc=model.evaluate(x_test_recsords,y_test)
print("loss: ",loss)
print("acc : ",acc)
while review != "e":
    review=input("Enter your review: ")
    print(lstmPrediction(review))
def lstmPrediction(review):
    smbls=dict()
    smbls[0]="Negative"
    smbls[1]="Neutral"
    smbls[2]="Positive"
    if choice == "2":
        review=sumSentenceEmbedding(process_review(review))
    else:
        review=avgSentenceEmbedding(process_review(review))
    data=np.array([review])
    data=np.reshape(data,(data.shape[0],1,300))
    ps=model.predict(data)
    print(ps)
    return smbls[np.argmax(ps)]
def process_review(rev):
    s=""
    s+=(rev.lower()+' ')
    s = re.sub("(@\w* )", ' ', s)
    s = re.sub("\\bhttps://(.*) \\b",' ',s)
    s = re.sub("[^a-z0-9\ ]+", ' ', s)
    s = re.sub(' \d+', ' ', s)
    s = re.sub(" +",' ',s)
    word_tokens = word_tokenize(s)
    s=""
    for w in word_tokens :
        if not w in stop_words:
            s+=(lem.lemmatize(w,"v")+' ')

    return s
print(lstmPrediction("fucking bad journey"))
print(lstmPrediction("I had a very good trip "))
print(lstmPrediction("I am so happy with this result "))
print(lstmPrediction("the result was really sucks"))
print(lstmPrediction("i am so sad for this bad services we got"))
print(lstmPrediction("amazing journy thanks for it"))
print(lstmPrediction("could you do it again please"))
print(lstmPrediction("let's play the match"))
print(lstmPrediction("I Don't know why nothing works"))



import gensim
import gensim.models.keyedvectors as word2vec

import pandas as pd
#word2vecDict = word2vec.KeyedVectors.load_word2vec_format("../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary=True)
#embedmodel = gensim.models.Word2Vec(tweets)
def read_data():
    df=pd.read_csv('/kaggle/input/twitterdata/Tweets.csv')
    df.sort_values(by=['airline_sentiment'])
    df.dropna(subset=["text"],axis=0,inplace=True)
    df.dropna(subset=["airline_sentiment"],axis=0,inplace=True)
    df.reset_index(drop=True,inplace=True)
    
    df = df.iloc[6400:,]
    df.reset_index(inplace=True)
    print(len(df))
    tweets=df['text']
    labels=[]
    for l in df.airline_sentiment:
        if(l=='neutral'):
            labels.append(1)
        elif (l=='negative'):
            labels.append(0)
        else:
            labels.append(2)
    
    return tweets,labels

import re
import nltk
import string
import numpy as np
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lem=WordNetLemmatizer()

def process_data(tweets,labels):
    processedTweets=[]
    newData=[]
    lbls=[]
    i=0
    for tweet in tweets:
        s=""
        s+=(tweet.lower()+' ')
        s = re.sub("(@\w* )", ' ', s)
        s = re.sub("\\bhttps://(.*) \\b",' ',s)
        s = re.sub("[^a-z0-9\ ]+", ' ', s)
        s = re.sub(' \d+', ' ', s)
        s = re.sub('\#(.*) ',' ',s)
        s = re.sub(" +",' ',s)
        
        word_tokens = word_tokenize(s)
        s=""
        for w in word_tokens :
            if not w in stop_words:
                s+=(lem.lemmatize(w,"v")+' ')
        if(len(s)>1):
            lbls.append(labels[i])
            newData.append(s)
        i+=1
        
    #lbls=np.array(lbls)
    return newData,lbls

def avgSentenceEmbedding(text, size=300):
    text=text.split()
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in text:
        try:
            vec += word2vecDict[word].reshape((1, size))
        except KeyError:
            continue
        count += 1
    if count != 0:
        vec /= count
    return vec

def sumSentenceEmbedding(text, size=300):
    text=text.split()
    vec = np.zeros(size).reshape((1, size))
    for word in text:
        try:
            vec += word2vecDict[word].reshape((1, size))
        except KeyError:
            continue
    return vec

import tensorflow
from tensorflow import keras
from sklearn.model_selection import train_test_split


tweets,labels=read_data()
tweets,labels=process_data(tweets,labels)
x_train,x_test,y_train,y_test=train_test_split(tweets, labels, test_size=0.2,random_state=42)

x_train_recsords=[]    
x_test_recsords=[]
choice=input("Enter 2 for sumSentenceEmbedding or anything else for avgSentenceEmbedding")
if choice == "2":
    for r in x_train:
        x_train_recsords.append(sumSentenceEmbedding(r))
    for r in x_test:
        x_test_recsords.append(sumSentenceEmbedding(r))
else :
    for r in x_train:
        x_train_recsords.append(avgSentenceEmbedding(r))
    for r in x_test:
        x_test_recsords.append(avgSentenceEmbedding(r))
print("Sentences processing and embbeding has finished..")
x_train_recsords=np.array(x_train_recsords)
x_test_recsords =np.array(x_test_recsords)

x_train_recsords=np.reshape(x_train_recsords,(x_train_recsords.shape[0],x_train_recsords.shape[1],300))
x_test_recsords=np.reshape(x_test_recsords,(x_test_recsords.shape[0],x_test_recsords.shape[1],300))

from keras.utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

print("Train and test data has prepeared..")
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, LSTM, Embedding, Dropout, Activation, Bidirectional
from keras.models import Sequential

y_train=np.array(y_train)
y_test=np.array(y_test)

model = Sequential()
model.add(Bidirectional(LSTM(units=100, return_sequences=True,input_shape=(x_train_recsords.shape[1],300))))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units=50)))
model.add(Dropout(0.3))
model.add(Dense(units=3,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(x_train_recsords,y_train,
                    epochs=25,
                    verbose=1,
                    validation_split=0.1
                    )
model.summary()
loss,acc=model.evaluate(x_test_recsords,y_test)
print("loss: ",loss)
print("acc : ",acc)
