# Importing the libraries

import numpy as np

import pandas as pd

# Input data files

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

submiss=pd.read_csv("../input/sample_submission.csv")
X_Train=train['text'].str.replace('[^a-zA-Z0-9]', ' ')

y_train=train['author']

X_Test=test['text'].str.replace('[^a-zA-Z0-9]', ' ')
## Multinomial Naive Bayes Classifier ##

# Build pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

classifier = Pipeline([('vect', CountVectorizer(lowercase=False)),

                      ('tfidf', TfidfTransformer()),

                      ('clf', MultinomialNB()),

])



# parameter tuning with grid search

from sklearn.model_selection import GridSearchCV

parameters = {'vect__ngram_range': [(1, 1), (1, 2),(1,3)],

              'vect__max_df': ( 0.7,0.8,0.9,1.0),

              'vect__min_df': (1,2),    

              'clf__alpha': ( 0.022,0.025, 0.028),

}

gs_clf = GridSearchCV(classifier, parameters,n_jobs=-1, verbose=1,cv=5)

gs_clf.fit(X_Train, y_train)

best_parameters = gs_clf.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))



# Predicting the Test set results

y_pred_proba = gs_clf.predict_proba(X_Test)

y_pred_proba
submiss['EAP']=y_pred_proba[:,0]

submiss['HPL']=y_pred_proba[:,1]

submiss['MWS']=y_pred_proba[:,2]

submiss.to_csv("submission_nb_word.csv",index=False)

submiss.head(10)