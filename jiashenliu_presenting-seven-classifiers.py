import numpy as np 

import pandas as pd 

import sklearn

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.cross_validation import train_test_split

from wordcloud import WordCloud,STOPWORDS

import re

import nltk

from nltk.corpus import stopwords

df = pd.read_csv('../input/Combined_News_DJIA.csv')

print(df.shape)

import matplotlib

matplotlib.rcParams["figure.figsize"] = "8, 8"
df.head(5)
df['Combined']=df.iloc[:,2:27].apply(lambda row: ''.join(str(row.values)), axis=1)
train,test = train_test_split(df,test_size=0.2,random_state=42)
non_decrease = train[train['Label']==1]

decrease = train[train['Label']==0]

print(len(non_decrease)/len(df))
def to_words(content):

    letters_only = re.sub("[^a-zA-Z]", " ", content) 

    words = letters_only.lower().split()                             

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words if not w in stops] 

    return( " ".join( meaningful_words )) 
non_decrease_word=[]

decrease_word=[]

for each in non_decrease['Combined']:

    non_decrease_word.append(to_words(each))



for each in decrease['Combined']:

    decrease_word.append(to_words(each))
wordcloud1 = WordCloud(background_color='black',

                      width=3000,

                      height=2500

                     ).generate(decrease_word[0])
plt.figure(1,figsize=(8,8))

plt.imshow(wordcloud1)

plt.axis('off')

plt.show()
wordcloud2 = WordCloud(background_color='white',

                      width=3000,

                      height=2500

                     ).generate(non_decrease_word[0])
plt.figure(1,figsize=(8,8))

plt.imshow(wordcloud2)

plt.axis('off')

plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer()

train_text = []

test_text = []

for each in train['Combined']:

    train_text.append(to_words(each))



for each in test['Combined']:

    test_text.append(to_words(each))

train_features = tfidf.fit_transform(train_text)

test_features = tfidf.transform(test_text)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve

from ggplot import *
Classifiers = [

    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators=200),

    AdaBoostClassifier(),

    GaussianNB()]
dense_features=train_features.toarray()

dense_test= test_features.toarray()

Accuracy=[]

Model=[]

for classifier in Classifiers:

    try:

        fit = classifier.fit(train_features,train['Label'])

        pred = fit.predict(test_features)

        prob = fit.predict_proba(test_features)[:,1]

    except Exception:

        fit = classifier.fit(dense_features,train['Label'])

        pred = fit.predict(dense_test)

        prob = fit.predict_proba(dense_test)[:,1]

    accuracy = accuracy_score(pred,test['Label'])

    Accuracy.append(accuracy)

    Model.append(classifier.__class__.__name__)

    print('Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))

    fpr, tpr, _ = roc_curve(test['Label'],prob)

    tmp = pd.DataFrame(dict(fpr=fpr, tpr=tpr))

    g = ggplot(tmp, aes(x='fpr', y='tpr')) +geom_line() +geom_abline(linetype='dashed')+ ggtitle('Roc Curve of '+classifier.__class__.__name__)

    print(g)