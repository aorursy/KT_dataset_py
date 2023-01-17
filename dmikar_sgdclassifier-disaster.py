import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.linear_model import SGDClassifier

text_clf = Pipeline([

        ('vect', CountVectorizer()),

        ('tfidf', TfidfTransformer()),

        ('clf', SGDClassifier(loss='hinge', penalty='l2',

                          alpha=1e-3, random_state=42,

                          max_iter=5, tol=None))])

X = train.text

y = train.target

#from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

text_clf.fit(X, y)

#y_pred = text_clf.predict(X_test)

#np.mean(y_pred == y_test)



#from sklearn import metrics

#print(metrics.classification_report(y_test, y_pred))
X_submit = test.text

y_submit = text_clf.predict(X_submit)

submission_df = pd.DataFrame({'id':test.id, 'target':y_submit})

submission_df.to_csv('submission.csv', index = False)