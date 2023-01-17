import spacy
import numpy as np
import pandas as pd
nlp = spacy.load('en_core_web_lg')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import metrics
import re
import seaborn as sns
import matplotlib.pyplot as plt
import os
true=pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")
false=pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")
true.sample(20)
false.sample(20)
def clean_true_news(text):
    try:
        match = re.search(r'\WReuters\W\s-\s',text)
        new_text=text[match.span()[1]:]
    except: 
        new_text=text
    return new_text
true['text']=true['text'].apply(clean_true_news)
true["target"]=1
false["target"]=0
df=pd.concat([true,false])
df.reset_index(drop=True,inplace=True)
df.sample(5)
#Number of token in the text:
def token_len(row_text):
    row=nlp(row_text)
    return len(row)

#Number of sentences in the text
def n_of_sentences(row_text):
    row=nlp(row_text)
    tot_sent=[sentence for sentence in row.sents]
    return len(tot_sent)
    
df['title_len']=df['title'].apply(token_len)
df['text_len']=df['text'].apply(token_len)

df['text_sent']=df['text'].apply(n_of_sentences)
df['capital_letters']=df['title'].apply(lambda x: len(re.findall(r'[A-Z]',x)))
df['non_alpha']=df['title'].apply(lambda x: len(re.findall(r'\W',x)))
df.sample(5)
sns.countplot(df.target)
chart=sns.countplot(x = "subject", hue = "target" , data = df)
labels=chart.get_xticklabels()
chart.set_xticklabels(labels,rotation=45)
sns.boxplot(x="target", y="title_len", data=df)
sns.boxplot(x="target", y="text_len", data=df)
sns.boxplot(x="target", y="text_sent", data=df)
sns.boxplot(x="target", y="capital_letters", data=df)
sns.boxplot(x="target", y="non_alpha", data=df)
df.isnull().sum()
blanks = []  
for i,text in df['text'].items():  
    if text.isspace():         
        blanks.append(i)     
print(len(blanks))
X=df['title']+' '+df['text']
y=df['target']
X_2=df[['title_len','capital_letters','non_alpha']]
X_train, X_test, X_2_train, X_2_test, y_train, y_test = train_test_split(X,X_2,y,test_size=0.2,random_state=42)
X_train_1m, X_test_1m, X_test_2m, X_train_2m, y_1m, y_2m = train_test_split(X_train,X_2_train,y_train,test_size=0.5,random_state=42)
text_clf=Pipeline([('tfidf',TfidfVectorizer()),
                  ('clf',LinearSVC())])
text_clf.fit(X_train_1m,y_1m)
yhat1=text_clf.predict(X_test)
print(metrics.classification_report(y_test,yhat1))
print(metrics.accuracy_score(y_test,yhat1))
#Create the prediction field for the second step model:
X_train_2m['y_predict']=text_clf.predict(X_test_1m)
#save the prediction on the test set for the final model:
X_2_test['y_predict']=text_clf.predict(X_test)
clf=LinearSVC(max_iter=10000,dual=False)
clf.fit(X_train_2m,y_2m)
yhat2=clf.predict(X_2_test)
print(metrics.classification_report(y_test,yhat2))
print(metrics.accuracy_score(y_test,yhat2))