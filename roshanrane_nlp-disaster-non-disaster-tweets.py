import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, normalize, TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression, RidgeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.svm import LinearSVC, SVC

from sklearn.model_selection import StratifiedKFold



pd.options.display.max_rows = None

plt.style.use('dark_background')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Loading train & test files

trainDf = pd.read_csv('../input/nlp-getting-started/train.csv')

testDf = pd.read_csv('../input/nlp-getting-started/test.csv')
trainDf = trainDf.set_index('id')

print(trainDf.duplicated().sum())

trainDf.drop_duplicates(inplace=True)
sns.countplot(y=trainDf.target)
trainDf.isnull().sum()
vctrain =  trainDf.keyword.value_counts()

plt.figure(figsize=(16,9))

sns.countplot(y=trainDf.keyword, order=vctrain.iloc[:10].index)
plt.figure(figsize=(16,9))

sns.countplot(y=trainDf.keyword, order=vctrain.iloc[:10].index)

plt.subplot(121)

sns.countplot(y=trainDf.keyword[trainDf.target == 0], order= trainDf.keyword[trainDf.target == 0].value_counts().iloc[:10].index, color='r')

plt.title('Top Keyword for non-Disaster')

plt.subplot(122)

sns.countplot(y=trainDf.keyword[trainDf.target == 1], order= trainDf.keyword[trainDf.target == 1].value_counts().iloc[:10].index, color='g')

plt.title('Top Keyword for Disaster')
vcltrain =  trainDf.location.value_counts()

plt.figure(figsize=(16,9))

sns.countplot(y=trainDf.location, order=vcltrain.iloc[:10].index)
def tweet_cleaning(text):

    stopword = stopwords.words('english')

    lemmentizer = WordNetLemmatizer()

    text = text.lower()

    text =  re.sub(r"https?://\S+",'',text)

    text = re.sub(r"@\w+", '',text)

    text = re.sub(r"#", '',text)

    text = re.sub(r'\n',' ', text)

    text = re.sub('\s+', ' ', text).strip()

    text = word_tokenize(text)

    text = [word for word in text if word.isalnum()]

    text = [word for word in text if word not in stopword]

    text = [lemmentizer.lemmatize(word) for word in text]

    text = " ".join(word for word in text)

    

    return text
def mentions(text):

    text = re.findall(r"@\w+",text)

    return [txt.strip('@') for txt in text]
def hashtags(text):

    text = re.findall(r"#\w+",text)

    return [txt.strip('#') for txt in text]
trainDf['clean_tweets'] = trainDf.text.apply(tweet_cleaning)

trainDf['mentions'] = trainDf.text.apply(mentions)

trainDf['hashtags'] = trainDf.text.apply(hashtags)
testDf['clean_tweets'] = testDf.text.apply(tweet_cleaning)

testDf['mentions'] = testDf.text.apply(mentions)

testDf['hashtags'] = testDf.text.apply(hashtags)
vdata = pd.concat([trainDf.drop('target', axis=1), testDf], axis=0)

vdata.head()
vectorizer = TfidfVectorizer()

vectorizer.fit(vdata.clean_tweets)

vectorize_tweets_train = vectorizer.transform(trainDf.clean_tweets)

vectorize_tweets_test = vectorizer.transform(testDf.clean_tweets)

train = pd.DataFrame(vectorize_tweets_train.toarray(),index=trainDf.index)

test = pd.DataFrame(vectorize_tweets_test.toarray(), index=testDf.index)
train = pd.concat([train, trainDf.target],axis=1)
X = train.iloc[:,:-1].values

y = train.target.values
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3)
def train_model(clf, Xtrain=Xtrain, ytrain=ytrain, Xtest=Xtest, ytest=ytest,nsplit=3):

    skf = StratifiedKFold(n_splits=nsplit)

    

    for trainIndex, testIndex in skf.split(Xtrain,ytrain):

        clf.fit(Xtrain[trainIndex],ytrain[trainIndex])

        pred = clf.predict(Xtrain[testIndex])



        print("Confusion Matrix :\n", confusion_matrix(ytrain[testIndex], pred))

        print("Classification Report :\n", classification_report(ytrain[testIndex], pred))

        print("Accuracy :\n", accuracy_score(ytrain[testIndex], pred))

        

    pred = clf.predict(Xtest)

    print("\033[1mConfusion Matrix :\033[0m\n", confusion_matrix(ytest, pred))

    print("\033[1mClassification Report :\033[0m\n", classification_report(ytest, pred))

    print("\033[1mAccuracy :\033[0m\n", accuracy_score(ytest, pred))

    

    return clf
mnb = MultinomialNB(alpha=1.5)
mnb = train_model(mnb)
gnb = GaussianNB(var_smoothing=1e-01)
gnb = train_model(gnb)
rgc = RidgeClassifier(alpha=2)
rgc = train_model(rgc)
dtc = DecisionTreeClassifier()
dtc = train_model(dtc)
rfc = RandomForestClassifier(n_estimators=200)
rfc = train_model(rfc)
xgc = XGBClassifier(n_jobs= -1, warm_start = True)
xgc = train_model(xgc)
svc = SVC(C=0.5)
svc = train_model(svc)
prediction = rgc.predict(test)

result = pd.DataFrame(prediction, index=testDf.id, columns=['target'])
result = result.reset_index()
result.columns = [['id', 'target']]
result.index = range(1, 3264, 1)
result.to_csv('nlp_sub1.csv',index=False)
result