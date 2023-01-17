import gc



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer



import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_json('../input/train.json')

test = pd.read_json('../input/test.json')



gc.collect();
import re; 



procent = re.compile('[0-9]+ ?%')

non_alphas = re.compile('[^a-z]+')



def tok_origin(tokens):

    return tokens;



def tok_clean(tokens):

    cleaned = []

    for token in tokens:

        if len(token) <= 2:

            continue



        token = token.lower()

        

        # remove procent

        token = procent.sub('', token)

        

        # remove non-alphanums

        token = non_alphas.sub(' ', token)

        

        cleaned.append(token.strip())

    return cleaned;
encoder = LabelEncoder()

y_train = encoder.fit_transform(train['cuisine'])
%%script false



from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer



parameters = {

    'tfidf__tokenizer': [tok_origin, tok_clean],

    'tfidf__binary': [False],

    'tfidf__use_idf': [True],

    'tfidf__sublinear_tf': [True],

    'classifier': [

        OneVsRestClassifier(MultinomialNB()), 

        OneVsRestClassifier(LogisticRegression(solver='saga', multi_class='multinomial'))

    ],

#    'classifier__estimator__warm_start': [True, False]

}



pipe = Pipeline(

    memory=None,

    steps=[

        ('tfidf', TfidfVectorizer(preprocessor=None, lowercase=False)),

        ('classifier', OneVsRestClassifier(MultinomialNB())),

    ])



grid = GridSearchCV(pipe, parameters, verbose=2, cv=2, iid=False, scoring='f1_micro')

grid.fit(train['ingredients'], y_train)



print(grid.best_params_)

print(grid.best_score_)

'''

{

    'classifier': OneVsRestClassifier(LogisticRegression()),

    'tfidf__binary': False, 

    'tfidf__sublinear_tf': True, 

    'tfidf__tokenizer': <function tok_split at 0x7f3614b04400>, 

    'tfidf__use_idf': True

}

0.7747515641456806

'''
%%script false



from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.multiclass import OneVsRestClassifier



parameters = {

    'estimator__multi_class'   : ['ovr', 'multinomial'],

    'estimator__C'             : [1, 5, 10, 50],

    'estimator__fit_intercept' : [False, True],

    'estimator__class_weight'  : [None, 'balanced'],

}



model = OneVsRestClassifier(LogisticRegression(solver='saga', max_iter=500, warm_start=True, random_state=42))



grid = GridSearchCV(model, parameters, verbose=2, cv=2, iid=False, scoring='f1_micro', n_jobs=3)

grid.fit(X_train_vec, y_train)



print(grid.best_params_)

print(grid.best_score_)



"""

{ 

    'estimator__C': 5, 

    'estimator__class_weight': None, 

    'estimator__fit_intercept': True, 

    'estimator__multi_class': 'ovr'

}

0.7807353580375396

"""
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer



parameters = {

    'tfidf__tokenizer': [tok_clean],

    'tfidf__binary': [False],

    'tfidf__use_idf': [True],

    'tfidf__sublinear_tf': [True],

    'classifier__estimator__multi_class'   : ['multinomial'],

    'classifier__estimator__C'             : [2.6],

    'classifier__estimator__fit_intercept' : [True],

    'classifier__estimator__class_weight'  : [None],

    'classifier__estimator__warm_start': [True]

}



pipe = Pipeline(

    memory=None,

    steps=[

        ('tfidf', TfidfVectorizer(preprocessor=None, lowercase=False)),

        ('classifier', OneVsRestClassifier(LogisticRegression(solver='saga', max_iter=500, random_state=42))),

    ])



grid = GridSearchCV(pipe, parameters, verbose=2, cv=2, iid=False, scoring='f1_micro', n_jobs=3)

grid.fit(train['ingredients'], y_train)



print(grid.best_params_)

print(grid.best_score_)

predicted_classes = encoder.inverse_transform(grid.predict(test['ingredients']))



submissions = pd.DataFrame({"id": test['id'], "cuisine": predicted_classes})

submissions.to_csv("submission.csv", index=False, header=True)