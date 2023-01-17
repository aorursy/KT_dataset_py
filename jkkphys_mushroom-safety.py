# Jason King

# Mushroom Safety: Initial Exploration



%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import time



pd.options.display.max_columns = 999

seed = 101
df = pd.read_csv('../input/mushrooms.csv')

df.head()
df['class'].value_counts()
df['class'] = np.where(df['class'] == 'e', 0, 1)

df.head()
feats = list(df.columns)

feats.remove('class')

df[feats].head()
def encodeOrdered(df, feat):

    '''

    This function pulls categorical features and encodes them as integers ordered by bin population.

    '''

    keys = list(df[feat].value_counts().index)

    values = range(df[feat].unique().shape[0])

    featMap = dict(zip(keys, values))

    df[feat] = df[feat].map(lambda x: featMap[x])

    return df



for feat in feats:

    df = encodeOrdered(df, feat)

    print('Finished %s.' % feat)
from sklearn.model_selection import train_test_split



X = df[feats].values

y = df['class'].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = seed)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators = 100, n_jobs = -1, random_state = seed)



start = time.time()

clf = clf.fit(X_train, y_train)

end = time.time()

print('Time to train classifier: %0.2fs' % (end - start))
y_prob = clf.predict_proba(X_test)[:,1]

y_pred = np.where(y_prob > 0.5, 1, 0)
from sklearn.metrics import confusion_matrix, classification_report



print(confusion_matrix(y_test, y_pred))