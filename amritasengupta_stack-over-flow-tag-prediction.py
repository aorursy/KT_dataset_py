import numpy as np
import pandas as pd
import string
question= pd.read_csv('../input/stacksample/Questions.csv', encoding='latin')
answer= pd.read_csv('../input/stacksample/Answers.csv', encoding='latin')
tags= pd.read_csv('../input/stacksample/Tags.csv', encoding='latin')
question.head()
answer.head()
tags.head()
print(question.shape, answer.shape, tags.shape)
print(question.Id.nunique(), answer.ParentId.nunique(), tags.Id.nunique())
answer.drop(columns=['Id','OwnerUserId', 'CreationDate'],inplace=True)
answer.columns=['Id', 'A_Score', 'A_Body']
grouped_answer = answer.groupby("Id")['A_Body'].apply(lambda answer: ' '.join(answer))
grouped_answer= grouped_answer.to_frame()
grouped_answer= grouped_answer.sort_values(by='Id')
grouped_answer.head()
tags['Tag']= tags['Tag'].astype(str)
grouped_tags = tags.groupby("Id")['Tag'].apply(lambda tags: ' '.join(tags))
#grouped_tags = tags.groupby("Id")['Tag'].apply(lambda tags: ' '.join(tags))

grouped_tags= grouped_tags.to_frame()
grouped_tags= grouped_tags.sort_values(by='Id')
grouped_tags.head()
print(grouped_answer.shape, grouped_tags.shape)
grouped_answer['Ids']= grouped_answer.index
grouped_tags['Ids']= grouped_tags.index
question.columns= ['Ids', 'OwnerUserId', 'CreationDate', 'ClosedDate', 'Score', 'Title',
       'Body']
question= question.sort_values(by='Ids')
df= pd.merge(question,grouped_answer,how='left')
df1= pd.merge(df,grouped_tags,how='left',on='Ids')
df1.head()
df1.drop(columns=['Ids', 'OwnerUserId', 'CreationDate', 'ClosedDate'],inplace=True)
df1.head()
df1=df1.drop_duplicates()
df1.shape
print(df1.Score.min(), df1.Score.max())
z= df1['Tag'].value_counts().sort_values(ascending=False)
z.index
df2= df1.groupby(by='Tag')['Tag'].count().sort_values(ascending=False).to_frame()
df2.columns= ['Tag_count']
df2['Tags']=df2.index
df1.columns= ['Score', 'Title', 'Body', 'A_Body', 'Tags']
df1= pd.merge(df1,df2,how='left',on='Tags')
df1.head()
df1= df1[df1['Tag_count']>=1000]
df1= df1[df1['Score']>3]
df1.shape
df1.Tags.value_counts().sort_values(ascending=False)
print(df1.isnull().sum())

print('Shape of df1:',df1.shape)
df1.drop(columns=['A_Body'],inplace=True)
def remove_punctuation(text):
    for punctuation in string.punctuation:
        text= text.replace(punctuation,'')
    return text
    
df1['Title']= df1['Title'].astype(str)

df1['Title1']= df1['Title'].apply(remove_punctuation)
df1['Title1']=df1['Title1'].str.lower()
df1['Title1']= df1['Title1'].str.split()
df1['Title1'].head()
df1['Body']= df1['Body'].astype(str)
import re

df1['Body1']= df1['Body'].apply(lambda x: re.sub('<[^<]+?>','',x))
df1['Body1'].head()
df1['Body1']= df1['Body1'].apply(remove_punctuation)
df1['Body1']=df1['Body1'].str.lower()
df1['Body1']= df1['Body1'].str.split()
df1['Body1'].head()
from nltk.stem import WordNetLemmatizer
lematizer= WordNetLemmatizer()

def word_lemmatizer(text):
    lem_text=[lematizer.lemmatize(i) for i in text]
    return lem_text
df1['Title1']= df1['Title1'].apply(lambda x: word_lemmatizer(x))
df1['Body1']= df1['Body1'].apply(lambda x: word_lemmatizer(x))
import spacy
sp= spacy.load('en_core_web_sm')
all_stopwords= sp.Defaults.stop_words
df1['Title1']= df1['Title1'].apply(lambda x:[word for word in x if not word in all_stopwords])
df1['Title1'].head()                                   

import spacy
sp= spacy.load('en_core_web_sm')
all_stopwords= sp.Defaults.stop_words
df1['Body1']= df1['Body1'].apply(lambda x:[word for word in x if not word in all_stopwords])
df1['Body1'].head()                                   
df1.drop(columns=['Title', 'Body', 'Tag_count','Score'], inplace=True)
df1.head()
from sklearn.feature_extraction.text import TfidfVectorizer

df1['Title1']= df1['Title1'].astype(str)
vectorizer = TfidfVectorizer()
X1 = vectorizer.fit_transform(df1['Title1'].str.lower())
df1['Body1']= df1['Body1'].astype(str)
vectorizer = TfidfVectorizer()
X2 = vectorizer.fit_transform(df1['Body1'].str.lower())
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder() 
df1['Tags']= le.fit_transform(df1['Tags'])
y = df1['Tags'].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X2, y, test_size=0.30, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
clf = LogisticRegression(C=10)

# Creating the model on Training Data
LOG=clf.fit(x_train,y_train)
prediction=LOG.predict(x_test)

# Printing the Overall Accuracy of the model
from sklearn import metrics
print(metrics.confusion_matrix(y_test, prediction))
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)
clf=XGBClassifier(max_depth=2, learning_rate=0.2, n_estimators=400, objective='binary:logistic', booster='gbtree')

# Creating the model on Training Data
XGB=clf.fit(x_train,y_train)
prediction=XGB.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)
model = MultinomialNB().fit(x_train,y_train)
prediction= model.predict(x_test)

from sklearn import metrics
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)
clf = KNeighborsClassifier(n_neighbors=4)

# Creating the model on Training Data
KNN=clf.fit(x_train,y_train)
prediction=KNN.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)

clf = RandomForestClassifier(max_depth=4, n_estimators=600,criterion='entropy')

# Creating the model on Training Data
RF=clf.fit(x_train,y_train)
prediction=RF.predict(x_test)

# Measuring accuracy on Testing Data
from sklearn import metrics
print(metrics.confusion_matrix(y_test, prediction))

# Printing the Overall Accuracy of the model
F1_Score=metrics.classification_report(y_test, prediction).split()[-2]
print('Accuracy of the model:', F1_Score)
