import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import warnings, re, string



from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression, SGDClassifier

from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import precision_score, r2_score, recall_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import make_pipeline 

from sklearn.svm import SVC



warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
df = (pd.read_csv('../input/fake-news/train.csv')

      .fillna(' ')

      .sample(frac = 1, random_state = 0)

      .set_index('id'))



df.head(3)
Params = {'Regresión Logística': LogisticRegression(C = 1e5

                                                    , random_state = 0)

           , 'Pasive Aggressive Classifier': PassiveAggressiveClassifier(max_iter = 50

                                                                         , random_state = 0)

           , 'SGD': SGDClassifier(max_iter = 5

                                  , tol = None)

           , 'SVC': SVC(kernel = 'linear'

                        , random_state = 0

                        , gamma = 'scale' )}



X = df.title + ' ' + df.author + ' ' + df.text

y = df.label
Result = {}

kf = KFold(n_splits = 10)

           

for i in Params:

    Score = []

    clf = make_pipeline(TfidfVectorizer(stop_words = 'english', max_df = 0.7)

                         , Params[i])

    

    clf.fit(X, y)

    Score.append(round(np.mean(y == clf.predict(X)) * 100, 2))

    Score.append(cross_val_score(clf, X, y, cv = kf, scoring = 'accuracy').mean())

    Score.append(round(precision_score(y, clf.predict(X), labels = [0, 1], pos_label = 1) * 100, 2))

    Score.append(r2_score(y, clf.predict(X)))

    Score.append(recall_score(y, clf.predict(X), average = None).round(2))

    Result[i] = Score



Result = (pd.DataFrame(Result, index=['Precisión (accuracy)'

                                      , 'Cross Val' 

                                      , 'Score (True)'

                                      , 'R Cuadrado'

                                      , 'Recall'])

          .transpose()

          .sort_values(by = 'Cross Val'

                       , ascending = False)

          .reset_index()

          .rename(columns = {'index':'Modelo'})

         )



Best = make_pipeline(TfidfVectorizer(stop_words = 'english', max_df = 0.7)

                         , Params[Result['Modelo'][0]])



Best.fit(X, y)



Result
df = (pd.read_csv('../input/fake-news/test.csv')

      .fillna(' ')

      .set_index('id'))



df['label'] = Best.predict(df.title + ' ' + df.author + ' ' + df.text)



df.head(3)
df.reset_index()[['id','label']].to_csv('submit.csv', index=False)