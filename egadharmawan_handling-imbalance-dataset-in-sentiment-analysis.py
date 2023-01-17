import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('../input/google-play-store-apps/googleplaystore_user_reviews.csv')
df.head()
df.info()
# I decide to drop the missing value since it's way to much

df.dropna(inplace=True)
df.info()
# Type of Sentiment

df.Sentiment.unique()
## Best App

plt.figure(figsize=(10,7))
best_app = df.App[df['Sentiment'] == 'Positive'].value_counts().head(10)
sns.barplot(x=best_app, y=best_app.index, data=df)
# Worst App

plt.figure(figsize=(10,7))
best_app = df.App[df['Sentiment'] == 'Negative'].value_counts().head(10)
sns.barplot(x=best_app, y=best_app.index, data=df)
## Handling Categorical

df['Sentiment'].replace(to_replace=['Positive','Negative','Neutral'], value=['1','0','2'],inplace=True)
## Spliting Data Feature & Target

X = df.Translated_Review
y = df.Sentiment
## PREPROCESSING

import re

def clean_text(text):
    # lowerxase
    text = text.lower()
    # Clear punctuation
    text = re.sub('\[.*?\]','',text)
    return text

clean = lambda x: clean_text(x)

X = X.apply(clean)
X
## REMOVE STOPWORDS

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
en_stops = set(stopwords.words('english'))

def stopwords(text):

    tokens = word_tokenize(text)
    filtered = []    
    for w in tokens:
        if w not in en_stops:
            filtered.append(w)
    result = ' '.join(filtered)
    return result

st = lambda x: stopwords(x)

X = X.apply(st)
X
## STEMMING

from nltk.stem import PorterStemmer

def stemming(text):
    ps = PorterStemmer()
    text = ps.stem(text)
    return text

stem = lambda x: stemming(x)

X = X.apply(stem)
X
# I split the data with train test split
from sklearn.model_selection import train_test_split

# And transform the data with TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Transform the feature data to vector before handling imbalance dataset with undersample
# X Train Vector
XTV = vectorizer.fit_transform(X_train)
# X Test Vector
XTSV = vectorizer.transform(X_test)
# Data Distribution before applying undersampling

y_train.value_counts()
## Near Miss Undersampling

from imblearn.under_sampling import NearMiss
### NearMiss-1:
###   Majority class examples with minimum average distance to three closest minority class examples.

undersample = NearMiss(version=1, n_neighbors=3)
XTM1, YTM1 = undersample.fit_resample(XTV, y_train)

# Check value distribution
YTM1.value_counts()
### NearMiss-2:
###    Majority class examples with minimum average distance to three furthest minority class examples.

undersample = NearMiss(version=2, n_neighbors=3)
XTM2, YTM2 = undersample.fit_resample(XTV, y_train)

# Check value distribution
YTM2.value_counts()
### NearMiss-3:
###    Majority class examples with minimum distance to each minority class example.

undersample = NearMiss(version=3, n_neighbors_ver3=3)
XTM3, YTM3 = undersample.fit_resample(XTV, y_train)

# Check value distribution
YTM3.value_counts()
### Undersample with Tomek Links
from imblearn.under_sampling import TomekLinks

undersample = TomekLinks()
XTTL, YTTL = undersample.fit_resample(XTV, y_train)

# Check value distribution
YTTL.value_counts()
### Undersample with ENN

from imblearn.under_sampling import EditedNearestNeighbours

undersample = EditedNearestNeighbours(n_neighbors=3)
XTENN, YTENN = undersample.fit_resample(XTV, y_train)

# Check value distribution
YTENN.value_counts()
### Undersample with OSS

from imblearn.under_sampling import OneSidedSelection

undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=200)
XTOSS, YTOSS = undersample.fit_resample(XTV, y_train)

# Check value distribution
YTOSS.value_counts()
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import xgboost as xgb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
## MultinomialNB
mnb = MultinomialNB()

### Near Miss 1
mnb.fit(XTM1, YTM1)
pred = mnb.predict(XTSV)
actual = np.array(y_test)
MNB_NM1 = accuracy_score(actual, pred)

print('Near Miss 1')
print('Accuracy Score :', MNB_NM1)
print('Report : ')
print(classification_report(actual, pred))

### Near Miss 2
mnb.fit(XTM2, YTM2)
pred = mnb.predict(XTSV)
actual = np.array(y_test)
MNB_NM2 = accuracy_score(actual, pred)

print('Near Miss 2')
print('Accuracy Score :', MNB_NM2)
print('Report : ')
print(classification_report(actual, pred))

### Near Miss 3
mnb.fit(XTM3, YTM3)
pred = mnb.predict(XTSV)
actual = np.array(y_test)
MNB_NM3 = accuracy_score(actual, pred)

print('Near Miss 3')
print('Accuracy Score :', MNB_NM3)
print('Report : ')
print(classification_report(actual, pred))

### Tomek Links
mnb.fit(XTTL, YTTL)
pred = mnb.predict(XTSV)
actual = np.array(y_test)
MNB_TL = accuracy_score(actual, pred)

print('Tomek Links')
print('Accuracy Score :', MNB_TL)
print('Report : ')
print(classification_report(actual, pred))

### Edited Nearest Neighbour
mnb.fit(XTENN, YTENN)
pred = mnb.predict(XTSV)
actual = np.array(y_test)
MNB_ENN = accuracy_score(actual, pred)

print('Edited Nearest Neighbour')
print('Accuracy Score :', MNB_ENN)
print('Report : ')
print(classification_report(actual, pred))

### One Side Selection
mnb.fit(XTOSS, YTOSS)
pred = mnb.predict(XTSV)
actual = np.array(y_test)
MNB_OSS = accuracy_score(actual, pred)

print('One Side Selection')
print('Accuracy Score :', MNB_OSS)
print('Report : ')
print(classification_report(actual, pred))
svc = SVC(kernel='rbf', C=1000, gamma=0.001)

### Near Miss 1
svc.fit(XTM1, YTM1)
pred = svc.predict(XTSV)
actual = np.array(y_test)
SVC_NM1 = accuracy_score(actual, pred)

print('Near Miss 1')
print('Accuracy Score :', SVC_NM1)
print('Report : ')
print(classification_report(actual, pred))

### Near Miss 2
svc.fit(XTM2, YTM2)
pred = svc.predict(XTSV)
actual = np.array(y_test)
SVC_NM2 = accuracy_score(actual, pred)

print('Near Miss 2')
print('Accuracy Score :', SVC_NM2)
print('Report : ')
print(classification_report(actual, pred))

### Near Miss 3
svc.fit(XTM3, YTM3)
pred = svc.predict(XTSV)
actual = np.array(y_test)
SVC_NM3 = accuracy_score(actual, pred)

print('Near Miss 3')
print('Accuracy Score :', SVC_NM3)
print('Report : ')
print(classification_report(actual, pred))

### Tomek Links
svc.fit(XTTL, YTTL)
pred = svc.predict(XTSV)
actual = np.array(y_test)
SVC_TL = accuracy_score(actual, pred)

print('Tomek Links')
print('Accuracy Score :', SVC_TL)
print('Report : ')
print(classification_report(actual, pred))

### Edited Nearest Neighbour
svc.fit(XTENN, YTENN)
pred = svc.predict(XTSV)
actual = np.array(y_test)
SVC_ENN = accuracy_score(actual, pred)

print('Edited Nearest Neighbour')
print('Accuracy Score :', SVC_ENN)
print('Report : ')
print(classification_report(actual, pred))

### One Side Selection
svc.fit(XTOSS, YTOSS)
pred = svc.predict(XTSV)
actual = np.array(y_test)
SVC_OSS = accuracy_score(actual, pred)

print('One Side Selection')
print('Accuracy Score :', SVC_OSS)
print('Report : ')
print(classification_report(actual, pred))
### XGBoost

XGB = xgb.XGBClassifier(objective='multi:softmax', num_class=3, n_estimators=150, seed=123)

### Near Miss 1
XGB.fit(XTM1, YTM1)
pred = XGB.predict(XTSV)
actual = np.array(y_test)
XGB_NM1 = accuracy_score(actual, pred)

print('Near Miss 1')
print('Accuracy Score :', XGB_NM1)
print('Report : ')
print(classification_report(actual, pred))

### Near Miss 2
XGB.fit(XTM2, YTM2)
pred = XGB.predict(XTSV)
actual = np.array(y_test)
XGB_NM2 = accuracy_score(actual, pred)

print('Near Miss 2')
print('Accuracy Score :', XGB_NM2)
print('Report : ')
print(classification_report(actual, pred))

### Near Miss 3
XGB.fit(XTM3, YTM3)
pred = XGB.predict(XTSV)
actual = np.array(y_test)
XGB_NM3 = accuracy_score(actual, pred)

print('Near Miss 3')
print('Accuracy Score :', XGB_NM3)
print('Report : ')
print(classification_report(actual, pred))

### Tomek Links
XGB.fit(XTTL, YTTL)
pred = XGB.predict(XTSV)
actual = np.array(y_test)
XGB_TL = accuracy_score(actual, pred)

print('Tomek Links')
print('Accuracy Score :', XGB_TL)
print('Report : ')
print(classification_report(actual, pred))

### Edited Nearest Neighbour
XGB.fit(XTENN, YTENN)
pred = XGB.predict(XTSV)
actual = np.array(y_test)
XGB_ENN = accuracy_score(actual, pred)

print('Edited Nearest Neighbour')
print('Accuracy Score :', XGB_ENN)
print('Report : ')
print(classification_report(actual, pred))

### One Side Selection
XGB.fit(XTOSS, YTOSS)
pred = XGB.predict(XTSV)
actual = np.array(y_test)
XGB_OSS = accuracy_score(actual, pred)

print('One Side Selection')
print('Accuracy Score :', XGB_OSS)
print('Report : ')
print(classification_report(actual, pred))
x = {'Undersample Method'  : ['Near Miss 1','Near Miss 2','Near Miss 3','Tomek Links',
                             'Edited Nearest Neighbour','One Side Selection'],     
     'MultinomialNB'      : [MNB_NM1, MNB_NM2, MNB_NM3, MNB_TL, MNB_ENN, MNB_OSS],
     'SVM'                : [SVC_NM1, SVC_NM2, SVC_NM3, SVC_TL, SVC_ENN, SVC_OSS],
     'XGBoost'            : [XGB_NM1, XGB_NM2, XGB_NM3, XGB_TL, XGB_ENN, XGB_OSS]}
result = pd.DataFrame(x)
result