import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from pathlib import Path



from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
comp_path = Path('/kaggle/input/cat-in-the-dat/')



train = pd.read_csv(comp_path / 'train.csv', index_col='id')

test = pd.read_csv(comp_path / 'test.csv', index_col='id')

sample_submission = pd.read_csv(comp_path / 'sample_submission.csv', index_col='id')
y_train = train.pop('target')



# Simple label encoding

for c in train.columns:

    le = LabelEncoder()

    # this is cheating in real life; you won't have the test data ahead of time ;-)

    le.fit(pd.concat([train[c], test[c]])) 

    train[c] = le.transform(train[c])

    test[c] = le.transform(test[c])



clf = RandomForestClassifier(n_estimators=2000, n_jobs=-1, min_samples_leaf=2)

clf.fit(train, y_train)



sample_submission['target'] = clf.predict_proba(test)[:, 1]

sample_submission.to_csv('rf_benchmark.csv')