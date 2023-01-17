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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import nltk
nltk.download('stopwords')

%matplotlib inline
sms=pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')
print(sms.shape)
sms.head(5)


sms.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
sms.columns=['Label','Messages']
sms.head(5)

print(sms.isnull().sum())
print(sms.shape)
sms['Messages'].head(15)

print(f'ham= {len(sms[sms["Label"] == "ham"])}')
print(f'spam= {len(sms[sms["Label"] == "spam"])}')
fig,ax1=plt.subplots(figsize=(7,5))
sns.countplot(x="Label",data=sms)

sms['Text_len']= sms['Messages'].apply(lambda x:len(x))
sms.head(6)
f, ax = plt.subplots(1, 2, figsize = (28, 7))

sns.distplot(sms[sms["Label"] == "spam"]["Text_len"], bins = 50, ax = ax[0],rug=True,kde_kws={"color": "r"},rug_kws={"color": "g"})
ax[0].set_xlabel("Spam Message Word Length")

sns.distplot(sms[sms["Label"] == "ham"]["Text_len"], bins = 100, ax = ax[1],rug=True,kde_kws={"color": "r"},rug_kws={"color": "g"})
ax[1].set_xlabel("Ham Message Word Length")

plt.show()
import string
def punc_count(text):
    count_punc=sum([1 for c in text if c in string.punctuation])
    return 100*count_punc/len(text)

sms['punc_%']= sms['Messages'].apply(lambda x:punc_count(x))
sms.head(6)
#I am creating a function that will iterate over all the characters in the texts and search for punctuation as mentioned in "string.punctuation"

def remove_punctuation(text):
    text_nopunc="".join([c for c in text if c not in string.punctuation])
    return text_nopunc

#Reason why I used join is, while I ran it I gave me"," between every letters
sms['clean_text']=sms['Messages'].apply(lambda x:remove_punctuation(x))
sms.head(7)
def tokenize(text):
    tokens=re.split('\W+',text)#W here stands for non-word and "w" stands for word, it will spilt on non-word
    return tokens

sms['text_tokens']=sms['clean_text'].apply(lambda x:tokenize(x.lower())) #x.lower to tell python that uppercase and lowercase with spellings are same words

sms.head()
from nltk.corpus import stopwords
stopword= nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text_no_sw= [word for word in text if word not in stopword]
    return text_no_sw

sms['text_clean']=sms['text_tokens'].apply(lambda x:remove_stopwords(x))
sms.head()

from nltk.stem import PorterStemmer
ps=nltk.PorterStemmer()
print(ps.stem('cats'))
def stemming(text_clean):
    stemmed=[ps.stem(word)for word in text_clean]
    return stemmed


sms['text_stemmed']=sms['text_clean'].apply(lambda x:stemming(x))
sms.head(5)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf= TfidfVectorizer()
X_tfidf = tfidf.fit_transform(sms['Messages'])

X=pd.concat([sms['Text_len'], sms['punc_%'], pd.DataFrame(X_tfidf.toarray())], axis=1)
X.head(5)

Y=sms['Label']
from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3)
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_jobs=-1) #n_jobs=-1 means the all processors CPU jobs will be running concurrently. 
rf.fit(X_train,Y_train)
Y_pred = rf.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(Y_test,Y_pred))
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_pred)