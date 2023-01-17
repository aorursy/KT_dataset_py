# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pyarrow.parquet as pq



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pq.read_pandas('../input/train.parquet').to_pandas()

train_metadata = pd.read_csv('../input/metadata_train.csv')
train = train_data[:100]

target = train_metadata.target[:100]
#import matplotlib.pyplot as plt

#train.corr().style.format("{:.2}").background_gradient(cmap='PuBu', axis=1)

train.info()
from sklearn.decomposition import PCA

pca = PCA(n_components=0.5, whiten=True)



X_pca = pca.fit_transform(train)



print('Número original de atributos:', train.shape[1])

print('Número reduzido de atributos:', X_pca.shape[1])
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.33, random_state=42)



model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print('Acurácia nos dados originais:', accuracy_score(y_test, y_pred))



#######



X_train, X_test, y_train, y_test = train_test_split(X_pca, target, test_size=0.33, random_state=42)



model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print('Acurácia nos dados reduzidos:', accuracy_score(y_test, y_pred))



X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.33, random_state=42)



pca = PCA(n_components=0.5, whiten=True)



X_train = pca.fit_transform(X_train)

X_test = pca.transform(X_test)



model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)



model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print('Acurácia nos dados originais:', accuracy_score(y_test, y_pred))
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif



X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.33, random_state=42)



#sc = StandardScaler()

#X_std = sc.fit_transform(X_train)



fvalue_selector = SelectKBest(f_classif, k=20)

X_kbest = fvalue_selector.fit_transform(X_train, y_train)



print('Número original de atributos:', train.shape[1])

print('Número reduzido de atributos:', X_kbest.shape[1])



model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Acurácia nos dados originais:', accuracy_score(y_test, y_pred))



model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)

model.fit(X_kbest, y_train)

X_test_kbest = fvalue_selector.transform(X_test)

y_pred = model.predict(X_test_kbest)

print('Acurácia nos dados Kbest:', accuracy_score(y_test, y_pred))
submission = pd.DataFrame()

submission['target'] = y_pred

submission.to_csv('submission.csv', index=False)

submission.head()