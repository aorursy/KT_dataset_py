import sklearn

import matplotlib.pyplot as plt

import pandas

from sklearn.cross_validation import train_test_split

import numpy
sms = pandas.read_csv("../input/spam.csv", encoding='latin-1')

sms.head()
sms=sms[['v1','v2']]

sms.columns=['isSpam','text']
def transformSpamColumn(x):

    if x=='ham':

        return 0

    return 1
sms['isSpam']=sms['isSpam'].apply(transformSpamColumn)
sms.head()
spam_distribution=sms['isSpam'].value_counts()

spam_distribution
Index = [1,2]

plt.bar(Index, spam_distribution)

plt.xticks(Index, ['Not Spam','Spam'],rotation=90)

plt.ylabel('Spam Distribution')

plt.xlabel('Spam')

plt.title('Spam Distribution')

plt.show()
from wordcloud import WordCloud,STOPWORDS

import re

from nltk.corpus import stopwords
def cleanedWords(raw_sentence):

    letters_only = re.sub("[^a-zA-Z]", " ", raw_sentence)

    words = letters_only.lower().split()                            

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words

                       if w not in stops]

    return meaningful_words
def getWordCloud(df, isSpam):

    df=df[df['isSpam'] == isSpam]

    words = ' '.join(df['text'])

    cleaned_word = " ".join(cleanedWords(words))

    wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='black',

                      width=3000,

                      height=2500

                     )

    wordcloud.generate(cleaned_word)

    plt.figure(1,figsize=(12, 12))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
getWordCloud(sms,1)
getWordCloud(sms,0)
import nltk
def sms_to_words(raw_sms):

    return( " ".join( cleanedWords(raw_sms) ))
sms['clean_text']=sms['text'].apply(lambda x: sms_to_words(x))
sms.head()
train,test = train_test_split(sms,test_size=0.2,random_state=42)
from sklearn.feature_extraction.text import CountVectorizer

v = CountVectorizer(analyzer = "word")

train_features= v.fit_transform(train['clean_text'].values)

test_features=v.transform(test['clean_text'].values)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score
Classifiers = [

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,

          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,

          verbose=0, warm_start=False),

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

        fit = classifier.fit(train_features,train['isSpam'])

        pred = fit.predict(test_features)

    except Exception:

        fit = classifier.fit(dense_features,train['isSpam'])

        pred = fit.predict(dense_test)

    accuracy = accuracy_score(pred,test['isSpam'])

    Accuracy.append(accuracy)

    Model.append(classifier.__class__.__name__)

    print('Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))
Index = [1,2,3,4,5,6,7]

plt.bar(Index, Accuracy)

plt.xticks(Index, Model, rotation=90)

plt.ylabel('Accuracy')

plt.xlabel('Model')

plt.title('Accuracies of Models')

plt.show()