# Load packages

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score

from sklearn.metrics import cohen_kappa_score, make_scorer, classification_report

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC



# Read data

data = pd.read_csv('../input/spam.csv',

                   encoding = 'ISO-8859-1')

data = data.ix[:, [1, 0]]

data.rename(columns={'v2':'text', 'v1':'ham_spam'}, inplace=True)



# Code spam flag

le = LabelEncoder()

y = le.fit_transform(data['ham_spam'])



# Create text variable

X = data['text']



# Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2008)
print('Proportion of spam messages in the training data:', round(np.mean(y_train), 2))

kappa = make_scorer(cohen_kappa_score)
count_vec = CountVectorizer(analyzer='word',

                            stop_words='english',

                            max_features=500)
nb = MultinomialNB()



nb_clf = Pipeline([('Count vectorizer', count_vec),

                   ('Naive Bayes', nb)])



print('Kappa (10-fold CV): ', 

      round(np.mean(cross_val_score(nb_clf, X_train, y_train, scoring=kappa, cv=10)), 3))
rf = RandomForestClassifier()



rf_clf = Pipeline([('Count vectorizer', count_vec),

                   ('Random Forest', rf)])



print('Kappa (10-fold CV): ', 

      round(np.mean(cross_val_score(rf_clf, X_train, y_train, scoring=kappa, cv=10)), 3))
lr = LogisticRegression(penalty='l1')



lr_l1_clf = Pipeline([('Count vectorizer', count_vec),

                      ('Logistic regression', lr)])



print('Kappa (10-fold CV): ', 

      round(np.mean(cross_val_score(lr_l1_clf, X_train, y_train, scoring=kappa, cv=10)), 3))
lr = LogisticRegression(penalty='l2')



lr_l2_clf = Pipeline([('Count vectorizer', count_vec),

                      ('Logistic regression', lr)])



print('Kappa (10-fold CV): ', 

      round(np.mean(cross_val_score(lr_l2_clf, X_train, y_train, scoring=kappa, cv=10)), 3))
svm = SVC(kernel='linear')



svm_clf = Pipeline([('Count vectorizer', count_vec),

                    ('SVM', svm)])



print('Kappa (10-fold CV): ', 

      round(np.mean(cross_val_score(svm_clf, X_train, y_train, scoring=kappa, cv=10)), 3))
svm_clf.fit(X_train, y_train)

y_test_pred = svm_clf.predict(X_test)

print('Classification report: ')

print("----------------------")

print("")

print(classification_report(y_test, y_test_pred))

print("")

print("Cohen's kappa:")

print("--------------")

kappa = cohen_kappa_score(y_test, y_test_pred)

print(round(kappa, 2))