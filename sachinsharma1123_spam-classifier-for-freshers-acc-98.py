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
df=pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')
df
#lets first drop the unknown columns

df=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df=df.rename({'v1':'target',

             'v2':'text'},axis=1)
df
#lets make another column i.e the length of the text

len_text=[]

for i in df['text']:

    len_text.append(len(i))
df['text_length']=len_text
df
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

df[df['target']=='spam']['text_length'].plot(bins=35,kind='hist',color='blue',label='spam',alpha=0.5)

plt.legend()

plt.xlabel('message length')

plt.show()
plt.figure(figsize=(12,5))

df[df['target']=='ham']['text_length'].plot(bins=35,kind='hist',color='red',label='spam',alpha=0.5)

plt.legend()

plt.xlabel('message length')

plt.show()
#from the above two histograms we can conclude that spam messages are mostly of length bw 150-200

#and ham messages are of shorter length
plt.figure(figsize=(12,5))

df['target'].value_counts().plot(kind='bar',color='green',label='spam-vs-nonspam')

plt.legend()

plt.show()
#from this figure we can conclude that ham messages are more than spam messages
import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
df['target']=np.where(df['target']=='spam',1,0)
spam=[]

ham=[]

spam_class=df[df['target']==1]['text']

ham_class=df[df['target']==0]['text']
def extract_ham(ham_class):

    global ham

    words = [word.lower() for word in word_tokenize(ham_class) if word.lower() not in stopwords.words("english") and word.lower().isalpha()]

    ham=ham+words
def extract_spam(spam_class):

    global spam

    words = [word.lower() for word in word_tokenize(spam_class) if word.lower() not in stopwords.words("english") and word.lower().isalpha()]

    spam=spam+words
spam_class.apply(extract_spam)

ham_class.apply(extract_ham )
from wordcloud import WordCloud

spam_wordcloud = WordCloud(width=600, height=400).generate(" ".join(spam))

plt.figure( figsize=(10,8), facecolor='k')

plt.imshow(spam_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show()
ham_cloud=WordCloud(width=600,height=400,background_color='white').generate(" ".join(ham))

plt.figure(figsize=(10,8),facecolor='k')

plt.imshow(ham_cloud)

plt.tight_layout(pad=0)

plt.show()
#top 10 spam words=

spam_words=np.array(spam)

pd.Series(spam_words).value_counts().head(n=10)
#top 10 ham words

ham_words=np.array(ham)

pd.Series(ham_words).value_counts().head(n=10)
import seaborn as sns

sns.set_style('whitegrid')



f, ax = plt.subplots(1, 2, figsize = (20, 6))



sns.distplot(df[df["target"] == 1]["text_length"], bins = 20, ax = ax[0])

ax[0].set_xlabel("Spam Message Word Length")



sns.distplot(df[df["target"] == 0]["text_length"], bins = 20, ax = ax[1])

ax[0].set_xlabel("Ham Message Word Length")



plt.show()
#now  we are done with visualizations task,next move into text ceaning
from nltk.stem import SnowballStemmer

import string

stemmer = SnowballStemmer("english")



def cleanText(message):

    

    message = message.translate(str.maketrans('', '', string.punctuation))

    words = [stemmer.stem(word) for word in message.split() if word.lower() not in stopwords.words("english")]

    

    return " ".join(words)



df["text"] = df["text"].apply(cleanText)

df.head(n = 10)    
y=df['target']

x=df['text']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score

cv=CountVectorizer()

lr=LogisticRegression(max_iter=10000)

x_train=cv.fit_transform(x_train)
lr.fit(x_train,y_train)

pred_1=lr.predict(cv.transform(x_test))

score_1=accuracy_score(y_test,pred_1)

score_1
from sklearn.naive_bayes import MultinomialNB

nb=MultinomialNB()

nb.fit(x_train,y_train)

pred_2=nb.predict(cv.transform(x_test))

score_2=accuracy_score(y_test,pred_2)

score_2
from sklearn.svm import SVC

svm=SVC()

svm.fit(x_train,y_train)

pred_3=svm.predict(cv.transform(x_test))

score_3=accuracy_score(y_test,pred_3)

score_3