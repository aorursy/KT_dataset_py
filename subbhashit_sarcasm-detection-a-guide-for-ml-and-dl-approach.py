# Basic Libraries

import numpy as np
import pandas as pd

#Visuals

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.figure_factory as ff

#NLP tasks

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer  
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS


#DL Tasks

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,GlobalAvgPool1D,Dense,Dropout,Bidirectional,LSTM,GRU
import tensorflow as tf

#ML Tasks

from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor,XGBRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression,SGDRegressor
from sklearn.svm import SVC,SVR
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from lightgbm import LGBMRegressor,LGBMClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
data=pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json",lines=True)
data.head()
data.describe()
plt.figure(figsize=(15,10))
sns.countplot(data.is_sarcastic)
sen=np.array(data.headline)
wc=WordCloud(width=500,height=500)
wc.generate(' '.join(sen))
wc.to_image()
import re
def process(x):
    processed_tweet = re.sub(r'\W', ' ', str(x))
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 
    processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
    processed_tweet = processed_tweet.lower()
    return processed_tweet
data.headline=data.headline.apply(process)
data.head()
tfidfconverter = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
data.headline= tfidfconverter.fit_transform(data.headline).toarray()
data.columns
from sklearn.model_selection import train_test_split as split,cross_val_score,KFold,GridSearchCV
x=data.drop(['is_sarcastic','article_link'],axis=1)
y=data['is_sarcastic']
xr,xt,yr,yt=split(x,y,test_size=0.9)

model=LGBMClassifier()
parameters={"n_estimators":[1000]}
grid=GridSearchCV(model,parameters,cv=2)
grid.fit(x,y)
print(grid.best_score_)
print(grid.best_params_)
yp=grid.predict(xt)
print(accuracy_score(yt,yp))
print(classification_report(yt,yp))
sns.heatmap(confusion_matrix(yt,yp),annot=True,cmap='rainbow')
labels=np.array(data.is_sarcastic)
len(np.unique(labels))
tf=Tokenizer(num_words=1000,oov_token="oov<>")
tf.fit_on_texts(data.headline)
a=tf.word_index
a["oov<>"]
seq=tf.texts_to_sequences(data.headline)
seq[:5]
pad=pad_sequences(seq,maxlen=100,padding="post")
pad=np.array(pad)
pad
num_class=len(np.unique(labels))
xr,xt,yr,yt=train_test_split(pad,labels,test_size=0.1)
yr=keras.utils.to_categorical(yr,num_class)
yt=keras.utils.to_categorical(yt,num_class)
model=Sequential()
model.add(Embedding(10000,10,input_length=100))
model.add(Bidirectional(GRU(32,return_sequences=True)))
model.add(GlobalAvgPool1D())
model.add(Dense(500,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_class,activation='softmax'))
model.summary()
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
history=model.fit(xr,yr,epochs=10,batch_size=128,validation_split=0.1)
score=model.evaluate(xt,yt,verbose=1)
print("Loss : {}".format(score[0]))
print("Accuracy : {}".format(score[1]))
import matplotlib.pyplot as plt
figure=plt.figure(figsize=(15,15))
ax=figure.add_subplot(121)
ax.plot(history.history['accuracy'])
ax.plot(history.history['val_accuracy'])
ax.legend(['Training Accuracy','Val Accuracy'])
bx=figure.add_subplot(122)
bx.plot(history.history['loss'])
bx.plot(history.history['val_loss'])
bx.legend(['Training Loss','Val Loss'])
plt.show()