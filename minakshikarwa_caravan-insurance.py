# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, fbeta_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import make_scorer

import seaborn as sns
cv_train = pd.read_csv(r'/kaggle/input/carvan_train.csv')

cv_test = pd.read_csv(r'/kaggle/input/carvan_test.csv')
cv_train.isna().sum()
cv_test.head()
cv_test['V86'] = np.nan

cv_train['data'] = 'train'

cv_test['data'] = 'test'
cv_all = pd.concat([cv_train, cv_test], axis = 0, ignore_index=True)

cv_all.head()
print(cv_all['V1'].value_counts()) #L0

l1 = cv_all['V1'].value_counts()

l1 = l1.index[l1>400][:-1]

for l in l1:

    name = 'V1' +'_'+str(l)

    cv_all[name] = (cv_all['V1'] == l).astype(int)

del cv_all['V1']
cv_all.head()
cat_col = ['V4','V5','V6','V44']

for col in cat_col:

    l1 = cv_all[col].value_counts() #L1

    cats = l1.index[l1>800][:-1]

    for cat in cats:

        name = col +'_'+str(cat)

        cv_all[name] = (cv_all[col] == cat).astype(int)

    del cv_all[col]   
cv_all.head()
#sns.heatmap(cv_all.corr(), annotation = True)
cv_test = cv_all[cv_all['data'] == 'test']

cv_test.head()

cv_test=cv_test.reset_index(drop=True)

del cv_test['data']

del cv_test['V86']

cv_test.head()

cv_test.shape
cv_train = cv_all[cv_all['data'] == 'train']

cv_train.head()

del cv_train['data']

sns.countplot(cv_train['V86'])
train_x = cv_train.drop('V86',1)

train_y = cv_train["V86"]
ftwo_scorer = make_scorer(fbeta_score, beta=2)

param_dist ={'n_estimators' :[400,450,475,500],

             'max_features' :[60, 65,70,80,90],

             'bootstrap' :[True, False],

             'class_weight':[None,'balanced'],

             'criterion':['entropy','gini'],

             'max_depth' :[13,15,17,19],

             'min_samples_leaf' :[15,20,23,25,27],

             'min_samples_split':[15,17,19,20]}
n_iter_search = 60

random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions = param_dist, 

                                   n_iter = n_iter_search, 

                                   scoring = ftwo_scorer,

                                   cv = 5, n_jobs = -1, verbose = 10)
random_search.fit(train_x, train_y)
random_search.best_estimator_
random_search.best_score_
pred_all = random_search.predict(cv_test)

(pred_all == 0).sum(), (pred_all == 1).sum() 

pred_all 