# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import spacy

from spacy.lang.en.stop_words import STOP_WORDS
df=pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv',encoding='latin',header=None)
df.head()
df=df[[5,0]]

df.columns=['tweets','sentiments']

df.head()
df['sentiments'].value_counts()
sent_map={0:'negative',4:'positive'}
df['word_counts']=df['tweets'].apply(lambda x:len(str(x).split()))
df.head()
df['char_counts']=df['tweets'].apply(lambda x:len(x))
df.head()
def get_avg_word_len(x):

    words=x.split()

    word_len=0

    for word in words:

        word_len+=len(word)

    return word_len/len(words)
df['avg_word_len']=df['tweets'].apply(lambda x:get_avg_word_len(x))
df.head()
def get_stop_word_count(x):

    words=x.split()

    count=0

    for word in words:

        if word in STOP_WORDS:

            count+=1

    return count
df['stop_word_count']=df['tweets'].apply(lambda x:get_stop_word_count(x))
df.head()
df['hashtag_count']=df['tweets'].apply(lambda x:len([t for t in x.split() if t.startswith('#')]))

df['mention_count']=df['tweets'].apply(lambda x:len([t for t in x.split() if t.startswith('@')]))
df.head()
df['digit_count']=df['tweets'].apply(lambda x:len([t for t in x.split() if t.isdigit()]))

df['upper_count']=df['tweets'].apply(lambda x:len([t for t in x.split() if t.isupper() and len(t)>3]))

df.head()
df.loc[45]['tweets']
df['tweets']=df['tweets'].apply(lambda x:x.lower())
contractions = { 

"ain't": "am not / are not / is not / has not / have not",

"aren't": "are not / am not",

"can't": "cannot",

"can't've": "cannot have",

"'cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he had / he would",

"he'd've": "he would have",

"he'll": "he shall / he will",

"he'll've": "he shall have / he will have",

"he's": "he has / he is",

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how has / how is / how does",

"i'd": "I had / I would",

"i'd've": "I would have",

"i'll": "I shall / I will",

"i'll've": "I shall have / I will have",

"i'm": "I am",

"i've": "I have",

"isn't": "is not",

"it'd": "it had / it would",

"it'd've": "it would have",

"it'll": "it shall / it will",

"it'll've": "it shall have / it will have",

"it's": "it has / it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"might've": "might have",

"mightn't": "might not",

"mightn't've": "might not have",

"must've": "must have",

"mustn't": "must not",

"mustn't've": "must not have",

"needn't": "need not",

"needn't've": "need not have",

"o'clock": "of the clock",

"oughtn't": "ought not",

"oughtn't've": "ought not have",

"shan't": "shall not",

"sha'n't": "shall not",

"shan't've": "shall not have",

"she'd": "she had / she would",

"she'd've": "she would have",

"she'll": "she shall / she will",

"she'll've": "she shall have / she will have",

"she's": "she has / she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"so's": "so as / so is",

"that'd": "that would / that had",

"that'd've": "that would have",

"that's": "that has / that is",

"there'd": "there had / there would",

"there'd've": "there would have",

"there's": "there has / there is",

"they'd": "they had / they would",

"they'd've": "they would have",

"they'll": "they shall / they will",

"they'll've": "they shall have / they will have",

"they're": "they are",

"they've": "they have",

"to've": "to have",

"wasn't": "was not",

"we'd": "we had / we would",

"we'd've": "we would have",

"we'll": "we will",

"we'll've": "we will have",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what shall / what will",

"what'll've": "what shall have / what will have",

"what're": "what are",

"what's": "what has / what is",

"what've": "what have",

"when's": "when has / when is",

"when've": "when have",

"where'd": "where did",

"where's": "where has / where is",

"where've": "where have",

"who'll": "who shall / who will",

"who'll've": "who shall have / who will have",

"who's": "who has / who is",

"who've": "who have",

"why's": "why has / why is",

"why've": "why have",

"will've": "will have",

"won't": "will not",

"won't've": "will not have",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y'all'd": "you all would",

"y'all'd've": "you all would have",

"y'all're": "you all are",

"y'all've": "you all have",

"you'd": "you had / you would",

"you'd've": "you would have",

"you'll": "you shall / you will",

"you'll've": "you shall have / you will have",

"you're": "you are",

"you've": "you have"

}
def cont_to_exp(x):

    if type(x)==str:

        for key in contractions:

            value=contractions[key]

            x=x.replace(key,value)

    return x
df['tweets']=df['tweets'].apply(lambda x:cont_to_exp(x))
df.head()
import re
x='hi mail is abc@gmail.com and second is xyz@gmail.com'
re.findall(r'([A-Za-z0-9._-]+@[A-Za-z0-9._-]+\.[A-Za-z0-9_-]+)',x)
def get_email_count(x):

    return len(re.findall(r'([A-Za-z0-9._-]+@[A-Za-z0-9._-]+\.[A-Za-z0-9_-]+)',x))
df['email_count']=df['tweets'].apply(lambda x:get_email_count(x))
df.head()
df[df['email_count']>0]
df['tweets']=df['tweets'].apply(lambda x:re.sub(r'([A-Za-z0-9._-]+@[A-Za-z0-9._-]+\.[A-Za-z0-9_-]+)','',x))
x='url is https://abc.com/xyz thank you'
re.findall(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',x)
re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?','',x)
df['tweets']=df['tweets'].apply(lambda x:re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?','',x))
df['tweets']=df['tweets'].apply(lambda x:re.sub('RT',"",x))
df.head()
df['tweets']=df['tweets'].apply(lambda x:re.sub('[^A-Z a-z 0-9-]+','',x))
df.head()
df.loc[35]['tweets']
x='hello   xyz   '
x=" ".join(x.split())
df['tweets']=df['tweets'].apply(lambda x:" ".join(x.split()))
from bs4 import BeautifulSoup
x='<html><h1>hello world<h1>'
BeautifulSoup(x,'lxml').get_text()
#df['tweets']=df['tweets'].apply(lambda x:BeautifulSoup(x,'lxml').get_text())
df.head()
import unicodedata
x='noção hello noção'
def remove_accented_chars(x):

    x=unicodedata.normalize('NFKD',x).encode('ascii','ignore').decode('ascii','ignore')

    return x
remove_accented_chars(x)
df['tweets']=df['tweets'].apply(lambda x:remove_accented_chars(x))
import spacy
x='this is stop word'
" ".join([t for t in x.split() if t not in STOP_WORDS])
df['tweets']=df['tweets'].apply(lambda x:" ".join([t for t in x.split() if t not in STOP_WORDS]))
text=" ".join(df['tweets'])

text=text.split()

freq=pd.Series(text).value_counts()

f20=freq[:20]

r20=freq[-20:]
f20
r20
df['tweets']=df['tweets'].apply(lambda x:" ".join([t for t in x.split() if t not in f20]))
df['tweets']=df['tweets'].apply(lambda x:" ".join([t for t in x.split() if t not in r20]))
df.head()
from wordcloud import WordCloud

import matplotlib.pyplot as plt

%matplotlib inline
x=" ".join(text[:20000])

wc=WordCloud(width=800,height=400).generate(x)
plt.imshow(wc)

plt.axis('off')

plt.show()
nlp=spacy.load('en_core_web_lg')
x='hello world I am shubham dog cat Ayush'

doc=nlp(x)

for token in doc:

    print(token.text,token.has_vector)
token.vector.shape
x='one two three dog cat lion'

doc=nlp(x)
for token1 in doc:

    for token2 in doc:

        print(token1.text,token2.text,token1.similarity(token2))

    print()
df.shape
df.columns
df0=df[df['sentiments']==0].sample(3000)

df4=df[df['sentiments']==4].sample(3000)
df_red=df0.append(df4)
df_red.shape
df_red.head()
df_red_feat=df_red.drop(labels=['tweets','sentiments'],axis=1)

df_red_feat.head()
df_red_y=df_red['sentiments']

df_red_y.head()
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

text_counts=cv.fit_transform(df_red['tweets'])
text_counts.toarray().shape
df_red_bow=pd.DataFrame(text_counts.toarray(),columns=cv.get_feature_names())
df_red_bow.head()
from sklearn.linear_model import SGDClassifier , LogisticRegression, LogisticRegressionCV

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.preprocessing import MinMaxScaler
sgd=SGDClassifier(n_jobs=-1,random_state=2,max_iter=200)

lr=LogisticRegression(random_state=2,max_iter=200)

lrcv=LogisticRegressionCV(cv=2,random_state=2,max_iter=1000)

svm=LinearSVC(random_state=2,max_iter=200)

rfc=RandomForestClassifier(n_jobs=-1,random_state=2,n_estimators=200)
clf={'SGD':sgd ,'LR':lr , 'LRCV':lrcv,'SVM':svm,'RFC':rfc}

clf.keys()
def classify(X,y):

    scaler=MinMaxScaler(feature_range=(0,1))

    X=scaler.fit_transform(X)

    

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

    

    for key in clf.keys():

        model=clf[key]

        model.fit(X_train,y_train)

        y_pred=model.predict(X_test)

        acc=accuracy_score(y_test,y_pred)

        print(key,"---->",acc)
%%time

classify(df_red_bow,df_red_y)
%%time

classify(df_red_feat,df_red_y)
def get_vector(x):

    doc=nlp(x)

    return doc.vector.reshape(1,-1)
%%time

df_red['vector']=df_red['tweets'].apply(lambda x:get_vector(x))
X=np.concatenate(df_red['vector'].to_numpy(),axis=0)
X.shape
classify(pd.DataFrame(X),df_red_y)
df.head()
df.shape
dfr=df.sample(20000)
%%time

dfr['vector']=dfr['tweets'].apply(lambda x:get_vector(x))
dfr.shape
X=np.concatenate(dfr['vector'].to_numpy(),axis=0)

y=(dfr['sentiments']>1).astype(int)
y.head()
%%time

classify(pd.DataFrame(X),y)
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,LSTM

from tensorflow.keras.models import Sequential
model=Sequential([

    Dense(128,activation='relu'),

    Dropout(0.25),

    BatchNormalization(),

    Dense(64,activation='relu'),

    Dropout(0.25),

    BatchNormalization(),

    Dense(2,activation='sigmoid')

])
import tensorflow as tf

y_oh=tf.keras.utils.to_categorical(y,num_classes=2)
X_train,X_test,y_train,y_test=train_test_split(X,y_oh,random_state=2,test_size=0.2)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history=model.fit(X_train,y_train,epochs=10,batch_size=32,validation_data=[X_test,y_test])
y_test=np.array(y_test)

y_test.reshape(10000,1)
from sklearn.metrics import confusion_matrix

y_pred=model.predict(X_test)

y_pred=np.argmax(y_pred,axis=1)

y_pred

cm=confusion_matrix(y_test,y_pred)

cm