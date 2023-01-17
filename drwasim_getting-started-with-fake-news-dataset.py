import numpy as np

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline



from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB
dataset = pd.read_csv('../input/classifying-the-fake-news/training.csv',index_col='Id')

test = pd.read_csv('../input/classifying-the-fake-news/test.csv',index_col='Id')

dataset.head()
X_train = dataset.loc[:, "text"]

y_train = dataset.loc[:, "label"]
pipe = Pipeline([('vect', CountVectorizer()),

                 ('tfidf', TfidfTransformer()),

                 ('model', MultinomialNB())])
model = pipe.fit(X_train, y_train)
X_test = test['text']

predictions = model.predict(X_test)
submission = pd.DataFrame({

        "Id": test.index,

        "Predicted": predictions

    })
submission.to_csv('submission.csv',index=False)