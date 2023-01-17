import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

text_clf_mnb = Pipeline([

        ('vect', CountVectorizer(ngram_range=(1, 3))),

        ('tfidf', TfidfTransformer()),

        ('clf', MultinomialNB())])
X = train.text

y = train.target

text_clf_mnb.fit(X, y)
X_submit = test.text

y_submit = text_clf_mnb.predict(X_submit)
submission_df = pd.DataFrame({'id':test.id, 'target':y_submit})

submission_df.to_csv('submission.csv', index = False)