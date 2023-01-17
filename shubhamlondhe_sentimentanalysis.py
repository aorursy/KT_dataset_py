# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from keras.preprocessing.text import Tokenizer,text_to_word_sequence

from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Embedding,LSTM,Dense

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from collections import Counter

from sklearn.manifold import TSNE

from gensim.models import word2vec

from nltk import word_tokenize

from nltk.corpus import stopwords

# Any results you write to the current directory are saved as output.
filesList=os.listdir('../input/sentiment labelled sentences/sentiment labelled sentences')

os.listdir('../input/sentiment labelled sentences/sentiment labelled sentences')
imdb_labelFile='../input/sentiment labelled sentences/sentiment labelled sentences/yelp_labelled.txt'

amazon_labelFile='../input/sentiment labelled sentences/sentiment labelled sentences/amazon_cells_labelled.txt'

yelp_labelFile='../input/sentiment labelled sentences/sentiment labelled sentences/yelp_labelled.txt'
def getReviewSentimentFromFile(file):

    fr=open(file)

    lines=fr.readlines()

    fr.close()

    reviewsentimentList=[]

    for l in lines:

        x=l.split('\t')

        reviewsentimentList.append([str.lstrip(str.rstrip(x[0])),str.lstrip(str.rstrip(x[1]))])

    return reviewsentimentList
rsList=getReviewSentimentFromFile(imdb_labelFile)+getReviewSentimentFromFile(amazon_labelFile)+getReviewSentimentFromFile(yelp_labelFile)

len(rsList[:])
rsList[0]
rsDF=pd.DataFrame(rsList,columns=['REVIEW','SENTIMENT'])
rsDF.head(5)
X=rsDF['REVIEW']

y=rsDF['SENTIMENT']

y=to_categorical(num_classes=2,y=y)
np.shape(y)
tok=Tokenizer(lower=True,num_words=10000)
tok.fit_on_texts(X)

seqs=tok.texts_to_sequences(X)

padded_seqs=pad_sequences(seqs,maxlen=100)
def createLSTM():

    model=Sequential()

    model.add(Embedding(10000,100))

    model.add(LSTM(256))

    model.add(Dense(100,activation='sigmoid'))

    model.add(Dense(2,activation='sigmoid'))

    return model
model=createLSTM()

model.summary()
X_train,X_test,y_train,y_test=train_test_split(padded_seqs,y,train_size=0.85,test_size=0.15,random_state=43)
np.shape(X_train),np.shape(y_train)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])

model.fit(X_train,y_train,batch_size=32,epochs=5,verbose=1)
acc=model.evaluate(X_test,y_test)[1]*100

print("The model accuracy is {}".format(acc))
idx=np.random.randint(len(rsDF['REVIEW']))

print(rsDF['REVIEW'].iloc[idx],'Sentiment:',rsDF['SENTIMENT'].iloc[idx])

test=[rsDF['REVIEW'].iloc[idx]]

test_seq=pad_sequences(tok.texts_to_sequences(test),maxlen=100)

pred=model.predict(test_seq)

proba=model.predict_proba(test_seq)

if np.argmax(pred)==0:

    print('NEG',proba[0][0]*100)

else:

    print('POS',proba[0][1]*100)
!pip install newspaper3k
from newspaper import Article
import requests

from bs4 import BeautifulSoup as BS
urls=["https://www.livemint.com/industry","https://www.livemint.com/politics","https://www.livemint.com/insurance"]
listForLink=[]

for url in urls:

    print(url)

    response = requests.get(url)

    soup = BS(response.content,"html.parser")

    listForLink.append(soup.find_all("h2"))

    
listForLinkFinal = [ j for sub in listForLink for j in sub]
titleLink={}

for item in listForLinkFinal:

    for a in item.find_all("a"):

        titleLink[a.text] = a["href"]
titleLink
import pandas as pd

from newspaper import Article

import nltk

columns=["Title","Date","Sentences","Probability""Score"]

index=[i for i in range(2000)]

dfSentimentAnalysis = pd.DataFrame(index=index,columns=columns)
dfSentimentAnalysis.head()
nltk.download("punkt")

listOfArticle=[]

for key, value in titleLink.items():

    article=Article(value)   

    article.download()

    article.parse()

    article.nlp()

    listOfArticle.append([article.title,article.publish_date.date(),nltk.tokenize.sent_tokenize(article.text)])
(listOfArticle[-1][0])
dfSentimentAnalysis.head(1160)
m=0

for i in range(len(listOfArticle)):

    for j in range(len(listOfArticle[i][2])):

        dfSentimentAnalysis.at[m,"Title"]=listOfArticle[i][0]

        dfSentimentAnalysis.at[m,"Date"]=listOfArticle[i][1]

        dfSentimentAnalysis.at[m,"Sentences"]=listOfArticle[i][2][j]

        m+=1

       
lenValue=0

for k in range(1160):

    if (type(dfSentimentAnalysis.at[k,"Title"]))== str:

        lenValue+=1

    else:    

        break

print(lenValue)
listOfArticle[0][0]
dfSentimentAnalysis.head(1120)
len(listOfArticle)
for i in range(lenValue):

    test = (dfSentimentAnalysis.at[i,"Sentences"])

    test_seq=pad_sequences(tok.texts_to_sequences(test),maxlen=100)

    pred=model.predict(test_seq)

    proba=model.predict_proba(test_seq)

    if np.argmax(pred)==0:

        dfSentimentAnalysis.at[i,"Probability"] = proba[0][0]*100

        dfSentimentAnalysis.at[i,"Score"] = 0

    else:

        dfSentimentAnalysis.at[i,"Probability"] = proba[0][0]*100

        dfSentimentAnalysis.at[i,"Score"] = 1      
m=0

pos=0

neg=0

finalSentimentResult={}

for i in range(len(listOfArticle)):

    for j in range(len(listOfArticle[i][2])):

        if dfSentimentAnalysis.at[m,"Score"] == 1:

            pos+=1

            m+=1    

        else:

            neg+=1

            m+=1                    

    if pos>neg:

        finalSentimentResult[listOfArticle[i][0]]="POS"

    else:

        finalSentimentResult[listOfArticle[i][0]]="NEG"

    pos,neg=0,0        
finalSentimentResult
import os

os.chdir(r'/kaggle/working')

dfSentimentAnalysis.to_csv(r'dfSentimentAnalysis.csv')

from IPython.display import FileLink

FileLink('dfSentimentAnalysis.csv')