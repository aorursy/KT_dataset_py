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
import pandas as pd
frame1=pd.read_csv('../input/train.tsv',sep='\t')
import nltk
import string
import os

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

path = '/opt/datacourse/data/parts'
token_dict = {}
stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(wordnet_lemmatizer.lemmatize(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems
i=0
for line in frame1['Phrase']:
    
    lowers = line.lower()
    no_punctuation = lowers.translate(str.maketrans('','',string.punctuation))
    token_dict[i] = no_punctuation
    i=i+1    
#this can take some time

senti=list(frame1['Sentiment'])
values=list(token_dict.values())
frame2=pd.DataFrame({'Phrases':values,'Sentiment':senti})
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(token_dict.values())
feature_names = tfidf.get_feature_names()
px=[]
b=dict()
for x in tfs:
    a=[]
    for col in x.nonzero()[1]:
        if(x[0,col]>0.12):
            #print(col)
            #print (feature_names[col], ' - ', x[0, col])
            a.append(feature_names[col])
        else:
            if feature_names[col] not in b:
                b[feature_names[col]]=x[0,col]
            b[feature_names[col]]=min(x[0,col],b[feature_names[col]])
    px.append(a)
b
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))

for i in range(len(px)):
    temp=px[i]
    px[i]=[]
    for w in temp:
        if w not in stop_words:
            px[i].append(w)

for i in range(len(px)):
    px[i]=' '.join(px[i])

frame3=pd.DataFrame(columns=['phrase','sentiment'])
frame2=pd.DataFrame({'phrase':px,'sentiment':frame2['Sentiment']})
k=0

l1=len(frame2)
for i in range(l1):
    if frame2.loc[i,'phrase']!='':
        frame3.loc[k]=[str(frame2.loc[i,'phrase']),frame2.loc[i,'sentiment']]
        k=k+1
        print(k)
vocab_size=len(feature_names)   
max_words=196
from keras.preprocessing.text import Tokenizer
t=Tokenizer()
t.fit_on_texts(px)
train_X=t.texts_to_sequences(px)
from keras.preprocessing.sequence import pad_sequences
train_X= np.array(pad_sequences(train_X,maxlen=max_words,padding='post'))
y_train=frame1['Sentiment']
import numpy as np
train_X=np.array(train_X)
y_train=np.array(y_train)
from sklearn.utils import shuffle
batch_size=1000
train_X=np.array(train_X)
train_X,y_train=shuffle(train_X,y_train)
#y_train=to_categorical(y_train)
X_valid, y_valid = train_X[:batch_size], y_train[:batch_size]
X_train2, y_train2 = train_X[batch_size:], y_train[batch_size:]
print(type(X_train2),type(y_train2),type(X_valid),type(y_valid))


ratio={0:12000,1:30000,2:79087,3:32927,4:15000}
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state=42,ratio=ratio)
train_X,y_train = sm.fit_sample(X_train2,y_train2)
import numpy as np

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(train_X,y_train)

a=clf.predict(X_valid)
frame_test=pd.read_csv('../input/test.tsv',sep='\t')
i=0
token_dict1={}
for line in frame_test['Phrase']:
    
    lowers = line.lower()
    no_punctuation = lowers.translate(str.maketrans('','',string.punctuation))
    token_dict1[i] = no_punctuation
    i=i+1
values=list(token_dict1.values())

frame_test=pd.DataFrame({'phrase':values})
len(frame_test)

px=values
px1=[]
tfs = tfidf.fit_transform(px)
for x in tfs:
    a=[]
    for col in x.nonzero()[1]:
        a.append(feature_names[col])
    px1.append(' '.join(a))

test_X=t.texts_to_sequences(px1)
test_X= np.array(pad_sequences(test_X,maxlen=max_words,padding='post'))
test_X

a=clf.predict(test_X)

frame=pd.read_csv('../input/test.tsv',sep='\t')
frame_final=pd.DataFrame({'PhraseId':frame['PhraseId'],'Sentiment':a})

frame_final.to_csv('../input/final_submission1.csv',index=False)

