# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import GaussianNB

from sklearn.grid_search import GridSearchCV

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import StandardScaler



train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



train_X = train_data.drop(['id','species'], axis = 1).values

le = LabelEncoder().fit(train_data['species'])

train_y = le.transform(train_data['species'])

scaler = StandardScaler().fit(train_X)

train_X = scaler.transform(train_X)



test_ids = test_data.pop('id')

test_X = test_data.values

scaler = StandardScaler().fit(test_X)

test_X = scaler.transform(test_X)



params = {'solver':('svd','lsqr'), 'n_components': [10,20,30,40,50,60,70]}

lda = LinearDiscriminantAnalysis()

clf = GridSearchCV(lda, params,cv=5)

clf.fit(train_X, train_y)



test_y=clf.predict_proba(test_X)



submission = pd.DataFrame(test_y, index=test_ids, columns=le.classes_)

submission.to_csv('submission_lda.csv')