import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import sklearn

from ggplot import *

from wordcloud import WordCloud

import nltk

df = pd.read_csv('../input/rdany_conversations_2016-03-01.csv')

## Check whether dataset is loaded

df.head()
def data_cleansing(corpus):

    letters_only = re.sub("[^a-zA-Z]", " ", corpus) 

    words = letters_only.lower().split()                            

    return( " ".join( words ))     
df['text'] = df['text'].apply(lambda x:data_cleansing(x))
df = df[['source','text']]

##Check the data cleansing

for i in range(5):

    print(df['text'][i])
df['len'] = df['text'].apply(lambda x: len(x.split()))
p = ggplot(aes(x='len'), data=df) + geom_histogram(binwidth=2)+ theme_bw() + ggtitle('Histogram of length of conversation')+ facet_wrap('source')

print(p)
import matplotlib.pyplot as plt

%matplotlib inline
def wordcloud(source):

    tmp = df[df['source']==source]

    clean_text=[]

    for each in tmp['text']:

        clean_text.append(each)

    clean_text = ' '.join(clean_text)

    if source == 'robot' :

        color='black'

    else:

        color='white'

    wordcloud = WordCloud(background_color=color,

                      width=3000,

                      height=2500

                     ).generate(clean_text)

    print('==='*30)

    print('word cloud of '+source+' is plotted below')

    plt.figure(1,figsize=(8,8))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
wordcloud('robot')

wordcloud('human')
## Split train/test sets

from sklearn.cross_validation import train_test_split

train, test = train_test_split(df,test_size=0.3)
## Create tfidf variables

train_corpus = []

test_corpus = []

for each in train['text']:

    train_corpus.append(each)

for each in test['text']:

    test_corpus.append(each)

## Start creating them

from sklearn.feature_extraction.text import TfidfVectorizer

v = TfidfVectorizer()

train_features = v.fit_transform(train_corpus)

test_features=v.transform(test_corpus)
print(train_features.shape)

print(test_features.shape)
## Call ML models from sklearn

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
Classifiers = {'lg':LogisticRegression(random_state=42,C=5,max_iter=200),'dt':DecisionTreeClassifier(random_state=42,min_samples_leaf=1),'rf':RandomForestClassifier(random_state=42,n_estimators=100,n_jobs=-1),'gb':GradientBoostingClassifier(random_state=42,n_estimators=100,learning_rate=0.3)}

def ML_Pipeline(clf_name):

    clf = Classifiers[clf_name]

    fit = clf.fit(train_features,train['source'])

    pred = clf.predict(test_features)

    Accuracy = accuracy_score(test['source'],pred)

    Confusion_matrix = confusion_matrix(test['source'],pred)

    print('==='*35)

    print('Accuracy of '+ clf_name +' is '+str(Accuracy))

    print('==='*35)

    print(Confusion_matrix)
ML_Pipeline('lg')
ML_Pipeline('dt')
ML_Pipeline('rf')
ML_Pipeline('gb')
clf = RandomForestClassifier(random_state=42,n_estimators=100,n_jobs=-1)

fit = clf.fit(train_features,train['source'])
words = v.get_feature_names()

importance = clf.feature_importances_

impordf = pd.DataFrame({'Word' : words,'Importance' : importance})

impordf = impordf.sort_values(['Importance', 'Word'], ascending=[0, 1])

impordf.head(20)