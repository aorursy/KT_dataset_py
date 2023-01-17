!pip install spacy
import spacy

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk

%matplotlib inline
buisness = pd.read_csv('../input/yelp-csv/yelp_academic_dataset_business.csv')

review = pd.read_csv('../input/yelp-csv/yelp_academic_dataset_review.csv')
buisness_n = buisness[buisness['categories'].str.contains('Restaurant') == True]
buisness_n = buisness_n.fillna(0)
review_n = review[review.business_id.isin(buisness_n['business_id']) == True]
review_new = review_n.sample(n = 35000, random_state = 42)
text = review_new['text']
texts_n = []

for i in text:

    i = i.replace("\n", " ")

    texts_n.append(i)
texts_n[0]
from nltk.tokenize import word_tokenize

token_texts = []

for i in range (len(texts_n)):

    r = word_tokenize(texts_n[i])

    token_texts.append(r)
from string import punctuation

texts_new = [" ".join([word for word in text if word not in punctuation and not word.isnumeric() \

                      and len(word) > 1]) for text in token_texts]

texts_new[1]
from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()

lem_texts = []

token_texts2 = []

for i in range (len(texts_new)):

    r = word_tokenize(texts_new[i])

    token_texts2.append(r)

for text in token_texts2:

    l = ' '.join([lem.lemmatize(w) for w in text])

    lem_texts.append(l)

lem_texts[0]
review_new = review_new.drop(['text'], axis=1)
review_new['text'] = lem_texts
review = review_new[['text','stars']]
review.head()
revieww = review.sample(n=8000, random_state=42)
X = revieww['text']

y = revieww.drop(['text'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
X_train.shape
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cnt = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.8, norm="l1", sublinear_tf=1)

cntv = cnt.fit_transform(X_train)
cntvv = cntv.toarray()
y_train = y_train.stars.tolist()
y_test = y_test.stars.tolist()
X_test_feat = cnt.transform(X_test)
X_test_feat = X_test_feat.toarray()
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=7, C=80, penalty='l2', class_weight="balanced", solver="lbfgs", multi_class="multinomial")

clf.fit(cntv, y_train)
pred1 = clf.predict(X_test_feat)
accuracy_score(pred1, y_test)
cm = confusion_matrix(y_test, pred1)
sns.set()

sns.heatmap(cm, annot=True, fmt="d", cmap="spring");
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
#params = {'n_estimators':(100,200,300),

          #'min_samples_leaf':(2,3,4,5),

          #'max_features':('auto','log2'),

          #'random_state':(5,6,7,42)}

#clf = RandomForestClassifier

#model = GridSearchCV(clf,params, cv=3)

#model.fit(cntvv,y_train)
clf1 = RandomForestClassifier(random_state=9, max_depth=20, n_estimators=70)
clf1.fit(cntvv,y_train)
pred = clf1.predict(X_test_feat)
accuracy_score(pred,y_test)
cm = confusion_matrix(y_test, pred)

sns.set()

sns.heatmap(cm, annot=True, fmt="d", cmap="spring");
def best_coefs(cnt,clf,coef,n):

    feat2coef = {word: coef for word, coef in zip(cnt.get_feature_names(), clf.coef_[coef])}

    for i in sorted(feat2coef.items(), key=lambda x: x[1], reverse=True)[:n]:

        print(i[0])
best_coefs(cnt,clf,0,12)
best_coefs(cnt,clf,1,12)