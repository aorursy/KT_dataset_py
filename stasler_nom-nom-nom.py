# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, cross_validate

from sklearn.svm import SVC

from xgboost import XGBClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_json('../input/whats-cooking-kernels-only/train.json').set_index('id')

test_df = pd.read_json('../input/whats-cooking-kernels-only/test.json').set_index('id')



X_train = train_df.ingredients.apply(','.join)

y_train = train_df.cuisine

X_test = test_df.ingredients.apply(','.join)

#encoder = LabelEncoder()

#y_train = encoder.fit_transform(y_train)
vectorizer = TfidfVectorizer(binary=True, tokenizer=lambda x: [i.strip() for i in x.split(',')])

classifier = LogisticRegression(multi_class='ovr', max_iter=500, C=2.5, penalty='l1')



model = Pipeline(steps=[('vectorizer', vectorizer),

                        ('classifier', classifier)])

#parameters = {"classifier__C": [2.0, 5.0],

#              "classifier__solver": ['liblinear', 'lbfgs'],

#              "classifier__penalty": ['l1', 'l2']}

#grid_search = GridSearchCV(model, param_grid=parameters, error_score=np.nan, verbose=5)

#grid_search.fit(X_train, y_train)

#print(grid_search.best_score_)

#print(grid_search.best_params_)

#model.fit(X_train, y_train)
#preds_logreg = model.predict(X_test)

#submission_logreg = pd.Series(preds_logreg, index=X_test.index).rename('cuisine')

#submission_logreg.to_csv('submission_logreg.csv', index=True, header=True)
classifier_svm = SVC()



model_svm = Pipeline(steps=[('vectorizer', vectorizer),

                            ('classifier', classifier_svm)])
lsa = TruncatedSVD(n_components=50)

classifier_xgboost = XGBClassifier(verbose=True)



model_xgboost = Pipeline(steps=[('vectorizer', vectorizer),

                                ('lsa', lsa),

                                ('classifier', classifier_xgboost)],

                         verbose=True)

#model_xgboost.fit(X_train, y_train)
#foo = model_xgboost.predict(X_test)
#scores = cross_validate(model_xgboost, X_train, y_train)

#scores['test_score'].mean()