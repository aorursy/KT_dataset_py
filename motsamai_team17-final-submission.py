import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pprint import pprint

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

from matplotlib.pyplot import *

import seaborn as sns



%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
train_df = pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')
train_df.head()
dist = train_df['type'].value_counts()

plt.hlines(y=list(range(16)), xmin=0, xmax=dist, color='red')

plt.plot(dist, list(range(16)), "D")

plt.title('Distribution of Personlity Types')

plt.yticks(list(range(16)), dist.index)

plt.ylabel('Personality Types', fontsize=13)

plt.xlabel('Number of posts', fontsize=13)

plt.show()
combined=pd.concat([train_df[['posts']],test_df[['posts']]])
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english',analyzer = "word",tokenizer = None,preprocessor = None,max_features = 10000)

combined=vectorizer.fit_transform(combined['posts'])
train = combined[:train_df.shape[0]]

test = combined[train_df.shape[0]:]

y=train_df['type']
y_mind = y.apply(lambda x: 0 if x[0] == 'I' else 1)

y_energy = y.apply(lambda x: 0 if x[1] == 'S' else 1)

y_nature = y.apply(lambda x: 0 if x[2] == 'F' else 1)

y_tactics = y.apply(lambda x: 0 if x[3] == 'P' else 1)
from sklearn.metrics import f1_score # better metric due to small frequence of date for few types

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_validate

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=200,learning_rate=0.15, random_state=0)

kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

scoring = {'acc': 'accuracy', 'f1': 'f1_micro'}
result = cross_validate(clf,train, y_mind, scoring=scoring,

                        cv=kfolds, n_jobs=-1, verbose=1)
print('Y_mind model performance:')

pprint(result)



for key in result:

    print(key + ' : ', result[key].mean())
result = cross_validate(clf,train, y_energy, scoring=scoring,

                        cv=kfolds, n_jobs=-1, verbose=1)
print('Y_energy model performance:')

pprint(result)



for key in result:

    print(key + ' : ', result[key].mean())
result = cross_validate(clf,train, y_nature, scoring=scoring,

                        cv=kfolds, n_jobs=-1, verbose=1)
print('Y_nature model performance:')

pprint(result)



for key in result:

    print(key + ' : ', result[key].mean())
result = cross_validate(clf,train, y_tactics, scoring=scoring,

                        cv=kfolds, n_jobs=-1, verbose=1)
print('Y_tactics model performance:')

pprint(result)



for key in result:

    print(key + ' : ', result[key].mean())
# fit the model

clf.fit(train,y_mind)

# predict the outcome for testing data

X_t = test

predictions_Mind = pd.DataFrame(clf.predict(X_t))
# fit the model

clf.fit(train,y_energy)

# predict the outcome for testing data

predictions_Energy = pd.DataFrame(clf.predict(X_t))
# fit the model

clf.fit(train,y_tactics)

# predict the outcome for testing data

predictions_Tactic = pd.DataFrame(clf.predict(X_t))
# fit the model

clf.fit(train,y_nature)

# predict the outcome for testing data

predictions_Nature = pd.DataFrame(clf.predict(X_t))
submission = pd.concat([predictions_Mind,predictions_Energy,predictions_Nature,predictions_Tactic], axis=1)
submission.reset_index(inplace=True)

submission.head()
submission['index'] = submission['index'] +1

submission.columns = ['id', 'mind', 'energy', 'nature', 'tactics']

submission.head()
submission.to_csv('submission_Gradient.csv', index=False)