# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('/kaggle/input/1056lab-brain-cancer-classification/train.csv')

df_test = pd.read_csv('/kaggle/input/1056lab-brain-cancer-classification/test.csv')
df_train['type'] = df_train['type'].map({'normal':0, 'ependymoma':1, 'glioblastoma':2, 'medulloblastoma':3, 'pilocytic_astrocytoma':4})
X = df_train.drop('type', axis=1).values

y = df_train['type'].values

X_test = df_test.values
print(X.shape)
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=0.1)

sel.fit(X)

X_ = sel.transform(X)

X_test = sel.transform(X_test)
print(X_.shape)
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier

est = RandomForestClassifier()

fs  = SelectFromModel(est)

fs.fit(X_, y)

X_ = fs.transform(X_)

X_test = fs.transform(X_test)
print(X_.shape)
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

pca = PCA(n_components=91)

pca.fit(X_)

plt.bar([n for n in range(1, len(pca.explained_variance_ratio_)+1)], pca.explained_variance_ratio_)
# 寄与率の確認

np.set_printoptions(precision=3, suppress=True) # numpyの小数点以下表示桁数と、指数表記設定

print('explained variance ratio: {}'.format(pca.explained_variance_ratio_))
from sklearn.decomposition import PCA

fs = PCA(n_components=12)

fs.fit(X_)

X_ = fs.transform(X_)

X_test = fs.transform(X_test)
print(X_.shape)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_, y, test_size=0.2, random_state=0)
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

'''params = {'criterion':('gini', 'entropy'), 'n_estimators':[120], 'max_depth':[i for i in range(3, 10)]}

gscv = GridSearchCV(rfc, params, cv=5)

gscv.fit(X_train, y_train)'''
'''scores = gscv.cv_results_['mean_test_score']

params = gscv.cv_results_['params']

for score, param in zip(scores, params):

  print('%.3f  %r' % (score, param))'''
'''print('%.3f  %r' % (gscv.best_score_, gscv.best_params_))'''
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(criterion= 'gini', max_depth = 6, n_estimators = 120)

rfc.fit(X_train, y_train)
from sklearn.metrics import f1_score



pred = rfc.predict(X_valid)

f1_score(y_valid, pred,average='weighted')
p = rfc.predict(X_test)
p
df_submit = pd.read_csv('/kaggle/input/1056lab-brain-cancer-classification/sampleSubmission.csv', index_col=0)

df_submit['type'] = p

df_submit.to_csv('submission.csv')