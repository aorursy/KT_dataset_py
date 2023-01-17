# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/classifying-20-newsgroups/train.csv', delimiter=',')
train_data.head()
X = train_data[['id', 'message']]
y = train_data['topic']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#I use these samples for testing the steps without waiting a long time
sample_size = 1000
X_sample = X_train.iloc[0:sample_size,0:2] #rows, columns
y_sample = y_train.iloc[0:sample_size] #rows, columns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
import nltk

def Tokenizer(str_input):
    str_input = str_input.lower()
    words = word_tokenize(str_input)
    #remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    #stem the words
    porter_stemmer=nltk.PorterStemmer()
    words = [porter_stemmer.stem(word) for word in words]
    return words
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier

text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=Tokenizer, max_df=0.3, min_df=0.001, max_features=100000)),
    #('svd',   TruncatedSVD(algorithm='randomized', n_components=500)),
    ('clf',   XGBClassifier(objective='multi:softmax', n_estimators=500, num_class=20, learning_rate=0.075, colsample_bytree=0.7, subsample=0.8, eval_metric='merror')),
])
from sklearn.model_selection import GridSearchCV

parameters = {
    #'tfidf__max_df': (0.25, 0.5, 0.75),
    #'tfidf__min_df': (0.001, 0.0025, 0.005),
    #'tfidf__max_features': (50000, 100000, 150000),
    #'tfidf__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    #'svd__n_components': (250, 500, 750),
    #'clf__n_estimators': (250, 500, 750),
    'clf__max_depth': (4, 6, 8),
    'clf__min_child_weight': (1, 5, 10),
    #'clf__alpha': (0.00001, 0.000001),
    #'clf__penalty': ('l2', 'elasticnet'),
    #'clf__max_iter': (10, 50, 80),
}

#gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)
#gs_clf.fit(X_sample.message, y_sample)

#print("Best score: %0.3f" % gs_clf.best_score_)
#print("Best parameters set:")
#best_parameters = gs_clf.best_estimator_.get_params()
#for param_name in sorted(parameters.keys()):
#    print("\t%s: %r" % (param_name, best_parameters[param_name]))

text_clf.fit(X_train.message, y_train)
predictions = text_clf.predict(X_test.message)
print("The training predictions are ready")
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

vectorizer = TfidfVectorizer(tokenizer=Tokenizer, min_df=0.001, max_df=0.3)
X_tfidf = vectorizer.fit_transform(X.message)
#vectorizer.get_feature_names()
X_tfidf.shape
from sklearn.decomposition import TruncatedSVD

svd_transformer = TruncatedSVD(algorithm='randomized', n_components=300)
X_svd = svd_transformer.fit_transform(X_tfidf)
X_svd.shape
from xgboost import XGBClassifier

xgb_classifier = XGBClassifier(max_depth=3, n_estimators=1000, learning_rate=0.075, colsample_bytree=0.7, subsample=0.8)
xgb_classifier.fit(X_svd, y)
print("The model is ready.")

X_test = pd.read_csv('../input/classifying-20-newsgroups-test/test.csv', delimiter=',')
X_test.head()
X_test_tfidf = vectorizer.transform(X_test.message)
X_test_svd = svd_transformer.transform(X_test_tfidf)

xgb_predictions = xgb_classifier.predict(X_test_svd)
predictions = xgb_predictions
predictions[0:10]
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix

print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, average='weighted'))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
output = X_test.copy()
output.insert(2, 'topic', predictions)
output.to_csv('submission.csv', sep=',', columns=['id', 'topic'], index=False)
print(os.listdir("../working"))
#output.iloc[1000:5010, :]
results = pd.read_csv('submission.csv', delimiter=',')
results.iloc[5000:5010, :]
