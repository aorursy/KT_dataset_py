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
import numpy as np

import pandas as pd



from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack

from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score

from sklearn import linear_model

from sklearn.model_selection import StratifiedKFold



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
cols1 = ['Respondent', 'ConvertedSalary', 'Hobby', 'OpenSource', 'Employment', 'Dependents', 'Gender']

cols2 = ['Respondent', 'Hobby', 'OpenSource', 'Employment', 'Dependents', 'Gender']



#df_train = pd.read_csv('../input/train.csv', dtype = None, index_col=0)

#df_test = pd.read_csv('../input/test.csv', dtype = None, index_col=0)

df_train = pd.read_csv('../input/train.csv', usecols=cols1, index_col=0)

df_test = pd.read_csv('../input/test.csv', usecols=cols2, index_col=0)

df_train.head()
#df_train = df_train['ConvertedSalary', 'Hobby', 'OpenSource', 'Employment', 'Dependents', 'Gender']

#df_train = df_train['ConvertedSalary']
y_train = df_train['ConvertedSalary']

X_train = df_train.drop(['ConvertedSalary'], axis=1)

X_test = df_test
# 一時的に学習データとテストデータを結合

X_train['train'] = 1

X_test['train'] = 0

X_all = pd.concat([X_train, X_test])
def yesno_to_num(yn):

    if str(yn) == 'Yes':

        return 1

    elif str(yn) == 'No':

        return 0

    

# Hobby, OpenSourceを数値に変換

X_all['Hobby'] = X_all['Hobby'].apply(yesno_to_num)

X_all['OpenSource'] = X_all['OpenSource'].apply(yesno_to_num)

X_all['Dependents'] = X_all['Dependents'].apply(yesno_to_num)



X_all['Hobby'].head()
def employment_to_num(en):

    if str(en) == 'Employed full-time':

        return 1

    elif en == 'Employed part-time':

        return 2

    elif en == 'Independent contractor, freelancer, or self-employed':

        return 3

    elif en == 'Not employed, and not looking for work':

        return 4

    elif en == 'Not employed, but looking for work':

        return 4

    elif en == 'Retired':

        return 5

    

X_all['Employment'] = X_all['Employment'].apply(employment_to_num)
X_all['Employment'].head()
def gender_to_num(gn):

    if 'Male' in str(gn):

        return 1

    elif 'Female' in str(gn):

        return 2

    elif 'Transgender' in str(gn):

        return 3

    elif gn == 'Non-binary, genderqueer, or gender non-conforming':

        return 4

    

X_all['Gender'] = X_all['Gender'].apply(gender_to_num)

X_all['Gender'].head()
X_all.fillna(-9999, inplace=True)
# 再度学習データとテストデータに分割

X_train = X_all[X_all['train'] == 1]

X_test = X_all[X_all['train'] == 0]



X_train.drop(['train'], axis=1, inplace=True)

X_test.drop(['train'], axis=1, inplace=True)



X_train.shape, X_test.shape
SEED = 71

NFOLDS = 5

skf = StratifiedKFold(n_splits=NFOLDS, random_state=SEED, shuffle=True)
y_pred_test = np.zeros(len(X_test))



for i, (train_ix, test_ix) in enumerate(skf.split(X_train, y_train)):

    # トレーニングデータ・検証データに分割

    X_tr, y_tr = X_train.values[train_ix], y_train.values[train_ix]

    X_te, y_te = X_train.values[test_ix], y_train.values[test_ix]

    

    # トレーニングデータからモデルを作成

#    clf = LGBMClassifier(

#        learning_rate = 0.05,

#        num_leaves=31,

#        colsample_bytree=0.9,

#        subsample=0.9,

#        n_estimators=9999,

#        random_state=71,

#        importance_type='gain'

#    )

#    clf.fit(X_tr, y_tr, early_stopping_rounds=50, eval_metric='auc', eval_set=[(X_te, y_te)], verbose=100)

    

    clf = linear_model.SGDRegressor(max_iter=1000)

    clf.fit(X_tr, y_tr)

 

    clf_rm = linear_model.SGDRegressor(max_iter=1000)

    clf_rm.fit(X_te, y_te)

    

    # テストデータに対して予測

    #y_pred_test += clf.predict_proba(X_test)[:,1]

    y_pred=clf_rm.predict(X_test)





y_pred_test /= NFOLDS
submission = pd.read_csv('../input/sample_submission.csv', index_col=0)

submission['ConvertedSalary'] = y_pred_test

submission.to_csv('kadai.csv')