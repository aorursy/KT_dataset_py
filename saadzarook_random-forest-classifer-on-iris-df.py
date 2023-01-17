import pandas as pd

import numpy as np

np.random.seed(0)
df = pd.read_csv('../input/IRIS.csv')

df.head()
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

df.head(10)
train, test = df[df['is_train'] == True], df[df['is_train'] == False]

print("Size of training data: ", len(train))

print("Size of test data: ", len(test))
y = pd.factorize(train['species'])[0]

y
from sklearn.ensemble import RandomForestClassifier as RFC
features = df.columns[:4]

features
clf = RFC(n_jobs = 2, random_state = 0)

clf.fit(train[features], y)
clf.predict(test[features])
clf.predict_proba(test[features])[10:20]
preds = df.species[clf.predict(test[features])]

preds[:5]
from sklearn.metrics import accuracy_score

accuracy_score(test['species'], preds)