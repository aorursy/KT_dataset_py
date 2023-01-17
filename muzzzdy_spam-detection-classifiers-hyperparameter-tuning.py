import numpy as np

import pandas as pd

import string

from nltk.stem import SnowballStemmer

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

%matplotlib inline
sms = pd.read_csv('../input/spam.csv', encoding='latin-1')

sms = sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

sms = sms.rename(columns = {'v1':'label','v2':'message'})
def text_process(text):

    

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]

    words = ""

    for i in text:

            stemmer = SnowballStemmer("english")

            words += (stemmer.stem(i))+" "

    return words
text_feat = sms['message'].copy()

text_feat = text_feat.apply(text_process)

vectorizer = TfidfVectorizer("english")

features = vectorizer.fit_transform(text_feat)
features_train, features_test, labels_train, labels_test = train_test_split(features, sms['label'], test_size=0.3, random_state=111)
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
pred_scores = []

krnl = {'rbf' : 'rbf','polynominal' : 'poly', 'sigmoid': 'sigmoid'}

for k,v in krnl.items():

    for i in np.linspace(0.05, 1, num=20):

        svc = SVC(kernel=v, gamma=i)

        svc.fit(features_train, labels_train)

        pred = svc.predict(features_test)

        pred_scores.append((k, [i, accuracy_score(labels_test,pred)]))
df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Gamma','Score'])

df['Score'].plot(kind='line', figsize=(11,6), ylim=(0.8,1.0))
df[df['Score'] == df['Score'].max()]
pred_scores = []

for i in range(3,61):

    knc = KNeighborsClassifier(n_neighbors=i)

    knc.fit(features_train, labels_train)

    pred = knc.predict(features_test)

    pred_scores.append((i, [accuracy_score(labels_test,pred)]))

df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])

df.plot(figsize=(11,6))
df[df['Score'] == df['Score'].max()]
pred_scores = []

for i in np.linspace(0.05, 1, num=20):

    mnb = MultinomialNB(alpha=i)

    mnb.fit(features_train, labels_train)

    pred = mnb.predict(features_test)

    pred_scores.append((i, [accuracy_score(labels_test,pred)]))

df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])

df.plot(figsize=(11,6))
df[df['Score'] == df['Score'].max()]
pred_scores = []

for i in range(2,21):

    dtc = DecisionTreeClassifier(min_samples_split=i, random_state=111)

    dtc.fit(features_train, labels_train)

    pred = dtc.predict(features_test)

    pred_scores.append((i, [accuracy_score(labels_test,pred)]))

df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])

df.plot(figsize=(11,6))
df[df['Score'] == df['Score'].max()]
slvr = {'newton-cg' : 'newton-cg', 'lbfgs': 'lbfgs', 'liblinear': 'liblinear', 'sag': 'sag'}

pred_scores = []

for k,v in slvr.items():

    lrc = LogisticRegression(solver=v, penalty='l2')

    lrc.fit(features_train, labels_train)

    pred = lrc.predict(features_test)

    pred_scores.append((k, [accuracy_score(labels_test,pred)]))

df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])

df.plot(figsize=(11,6))
df[df['Score'] == df['Score'].max()]
pred_scores = []

lrc = LogisticRegression(solver='liblinear', penalty='l1')

lrc.fit(features_train, labels_train)

pred = lrc.predict(features_test)

print(accuracy_score(labels_test,pred))
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier
pred_scores = []

for i in range(2,36):

    rfc = RandomForestClassifier(n_estimators=i, random_state=111)

    rfc.fit(features_train, labels_train)

    pred = rfc.predict(features_test)

    pred_scores.append((i, [accuracy_score(labels_test,pred)]))

df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])

df.plot(figsize=(11,6))
df[df['Score'] == df['Score'].max()]
pred_scores = []

for i in range(25,76):

    abc = AdaBoostClassifier(n_estimators=i, random_state=111)

    abc.fit(features_train, labels_train)

    pred = abc.predict(features_test)

    pred_scores.append((i, [accuracy_score(labels_test,pred)]))

df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])

df.plot(figsize=(11,6))
df[df['Score'] == df['Score'].max()]
from sklearn.ensemble import BaggingClassifier

pred_scores = []

for i in range(2,21):

    bc = BaggingClassifier(n_estimators=i, random_state=111)

    bc.fit(features_train, labels_train)

    pred = bc.predict(features_test)

    pred_scores.append((i, [accuracy_score(labels_test,pred)]))

df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])

df.plot(figsize=(11,6))
df[df['Score'] == df['Score'].max()]
from sklearn.ensemble import ExtraTreesClassifier

pred_scores = []

for i in range(2,21):

    etc = ExtraTreesClassifier(n_estimators=i, random_state=111)

    etc.fit(features_train, labels_train)

    pred = etc.predict(features_test)

    pred_scores.append((i, [accuracy_score(labels_test,pred)]))

df = pd.DataFrame.from_items(pred_scores,orient='index', columns=['Score'])

df.plot(figsize=(11,6))
df[df['Score'] == df['Score'].max()]