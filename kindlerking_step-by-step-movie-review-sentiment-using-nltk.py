#importing basic libraries
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

%%time
df=pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
df.info()
#no missing values
#equal division of positive and negative sentiment
df['sentiment'].value_counts().plot(kind='pie',autopct='%.1f')
%%time
#1. Removing all html tags

from bs4 import BeautifulSoup
def html_remover(text):
    soup=BeautifulSoup(text,'html.parser')
    a=soup.get_text()
    return a
df['review']=df['review'].apply(html_remover)
df['review'][0]
%%time
#2. Removal of punctuations and special characters
import re
def sp_char_remover(review):
    review = re.sub('\[[^]]*\]', ' ', review)
    review = re.sub('[^a-zA-Z]', ' ', review)
    return review
df['review']=df['review'].apply(sp_char_remover)
df['review'][1]
%%time
#Converting To lower
def lower(text):
    return text.lower()
df['review']=df['review'].apply(lower)
df['review'][2]
%%time
#3. Removal of stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def stopword_remover(text):
    x=[]
    text=text.split()    #splitting into individual words
    for i in text:
        if i not in stopwords.words('english'):
            x.append(i)
    return x

df['review']=df['review'].apply(stopword_remover)
df['review'][0]
%%time
#4. Lemmatizing the stopwords and then joining it back
from nltk.stem import WordNetLemmatizer
lem=WordNetLemmatizer()

def temp(text):
    text=" ".join(text)
    return text

def lemma_join(text):
    text=[lem.lemmatize(word) for word  in text]
    text=temp(text)
    return text

df['review']=df['review'].apply(lemma_join)        
df['review'][0]
#Separation into training and testing
from sklearn.model_selection import train_test_split
df_train, df_test, train_data_label, test_data_label = train_test_split(df['review'], df['sentiment'], test_size=0.20, random_state=42)
#Changing Labels to 1 and 0 for the ease of understanding where 1 is positive review and 0 is negative review.
train_data_label=(train_data_label.replace({'positive':1,'negative':0}))
test_data_label=(test_data_label.replace({'positive':1,'negative':0}))
#Creating cleaned corpus from the cleaned df['review'] dataset for the purpose of training
corpus_train = []
corpus_test  = []

for i in df_train.index:
    temp=df_train[i]
    corpus_train.append(temp)

for j in df_test.index:
    temp1=df_test[j]
    corpus_test.append(temp1)
    
    
#Dummy corpus to perform Vectorization
corpus_train2=corpus_train
corpus_test2=corpus_test
%%time
#5. Count Vectorization (Bag of words model)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
cv_train=cv.fit_transform(corpus_train2)
cv_test=cv.transform(corpus_test2)
%%time
#6. Using a Support vector classifier for training our model
from sklearn.svm import LinearSVC
lin_svc=LinearSVC(C=0.5,random_state=42,max_iter=10000)
lin_svc.fit(cv_train,train_data_label)

y_pred=lin_svc.predict(cv_test)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


print(classification_report(test_data_label,y_pred))
print("ACCURACY SCORE IS: ",accuracy_score(test_data_label,y_pred))