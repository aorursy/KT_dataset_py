import pandas as pd

import numpy as np



data = pd.read_csv('../input/Reviews.csv')

data.dropna(inplace=True)

print(data.columns)

print(data.shape)

data[['Summary', 'Text', 'Score']].head()
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(min_df=1)



rnds = data.sample(frac = 0.1, random_state=87824, axis=0)

bow = vect.fit_transform(rnds['Text'])
print(vect.transform(['asdfadj']).count_nonzero())

print(vect.transform(['great']).count_nonzero())
from sklearn.ensemble import AdaBoostClassifier

ab = AdaBoostClassifier(n_estimators=200)



ab.fit(bow, rnds['Score'])
tests = data.sample(frac = 0.1, random_state=824, axis=0)

preds = ab.predict(vect.transform(tests['Text']))



from sklearn.metrics import f1_score

print('Accuracy: ' + str( 100*sum(preds == tests['Score'].values)/len(preds) ) + '%')

print('F1      : ' + str(f1_score(tests['Score'],preds, average='micro')))