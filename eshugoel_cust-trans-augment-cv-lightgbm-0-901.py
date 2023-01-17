# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB,GaussianNB



from keras.models import Sequential

from keras.layers import Dense, Dropout,Conv2D, Flatten

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils



from sklearn.metrics import log_loss

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.model_selection import StratifiedKFold



import lightgbm

# Any results you write to the current directory are saved as output.
import os

os.listdir('../')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample_sub = pd.read_csv('../input/sample_submission.csv')
sample_sub.head()
print(train.head())

print("==")

print(train.shape)

print("===")

print(test.head())

print("==")

print(test.shape)
train['target'].value_counts(normalize=True)
independent_feat = ['var_'+str(i) for i in range(200)]

dependent_feat = 'target'
train[independent_feat].head()
# sc = StandardScaler()

# train[independent_feat] = sc.fit_transform(train[independent_feat])

# test[independent_feat] = sc.transform(test[independent_feat])
## Data Augmentation

# https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment



def augment(x,y,t=3):

    xs,xn = [],[]

    for i in range(t):

        mask = y>0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xs.append(x1)



    for i in range(t//2):

        mask = y==0

        x1 = x[mask].copy()

        ids = np.arange(x1.shape[0])

        for c in range(x1.shape[1]):

            np.random.shuffle(ids)

            x1[:,c] = x1[ids][:,c]

        xn.append(x1)



    xs = np.vstack(xs)

    xn = np.vstack(xn)

    ys = np.ones(xs.shape[0])

    yn = np.zeros(xn.shape[0])

    x = np.vstack([x,xs,xn])

    y = np.concatenate([y,ys,yn])

    return x,y
# https://www.kaggle.com/deepak525/best-parameters-lb-0-900



# V-01

# param = {'num_leaves': 13,

#          'min_data_in_leaf': 80,

#           'min_sum_hessian_in_leaf': 10.0,

#          'objective': 'binary',

#           'boost_from_average': False,

#          'max_depth': -1,

#          'learning_rate': 0.0083,

#          'boost': 'gbdt',

#          'bagging_freq': 5,

#          'tree_learner': "serial",

#          'bagging_fraction': 0.333,

#          'feature_fraction': 0.041,

#          #'reg_alpha': 1.738,

#          #'reg_lambda': 4.99,

#          'metric': 'auc',

#          #'min_gain_to_split': 0.01077313523861969,

#          #'min_child_weight': 19.428902804238373,

#          'num_threads': 8}



# V-02



random_state = 42

param = {

    "objective" : "binary",

    "metric" : "auc",

    "boosting": 'gbdt',

    "max_depth" : -1,

    "num_leaves" : 13,

    "learning_rate" : 0.01,

    "bagging_freq": 5,

    "bagging_fraction" : 0.4,

    "feature_fraction" : 0.05,

    "min_data_in_leaf": 80,

    "min_sum_heassian_in_leaf": 10,

    "tree_learner": "serial",

    "boost_from_average": "false",

    #"lambda_l1" : 5,

    #"lambda_l2" : 5,

    "bagging_seed" : random_state,

    "verbosity" : 1,

    "seed": random_state,

    "num_threads":20

}
n_fold = 5

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

prediction = np.zeros(len(test))



for fold_n, (train_index, valid_index) in enumerate(folds.split(train[independent_feat],train[dependent_feat])):

    print(fold_n)

    X_train, X_valid = train[independent_feat].iloc[train_index], train[independent_feat].iloc[valid_index]

    y_train, y_valid = train[dependent_feat].iloc[train_index], train[dependent_feat].iloc[valid_index]

    print("Creation of Train validation Done")

    X_tr, y_tr = augment(X_train.values, y_train.values)

    X_tr = pd.DataFrame(X_tr)

    print("Creation of Augment data Done")

    train_data = lightgbm.Dataset(X_tr, label=y_tr)

    val_data = lightgbm.Dataset(X_valid, label=y_valid)

    print("Creation of LighGbm data Done")

    model = lightgbm.train(param,

                       train_data,

                       valid_sets=val_data,

                       num_boost_round=100000,

                       verbose_eval=5000, 

                       early_stopping_rounds = 10000

                       )

    prediction += model.predict(test[independent_feat])/n_fold
prediction
result = pd.DataFrame({'ID_code':test['ID_code'],'target':list(prediction)})

result.head()
result.to_csv('lgbm_aug_cv.csv',index=False)