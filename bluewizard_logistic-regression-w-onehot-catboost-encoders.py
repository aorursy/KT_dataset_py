import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')

train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')
%%time



# Random permutation is needed for CatBoostEncoder to reduce leakage

def random_permutation(x):

    perm = np.random.permutation(len(x)) 

    x = x.iloc[perm].reset_index(drop=True) 

    return x



train = random_permutation(train)

test = random_permutation(test)



train_ids = train.id

test_ids = test.id



train.drop('id', 1, inplace=True)

test.drop('id', 1, inplace=True)



train_targets = train.target

train.drop('target', 1, inplace=True)
from category_encoders.cat_boost import CatBoostEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
%%time



for col in train.columns:

    train[col] = train[col].astype(str)

    train[col].fillna('NA', inplace=True)

    test[col] = test[col].astype(str)

    test[col].fillna('NA', inplace=True)
%%time



# noms 5-9

for i in [5,6,7,8,9]:

    cbe = CatBoostEncoder()

    train[f'nom_{i}'] = cbe.fit_transform(train[f'nom_{i}'], train_targets)

    test[f'nom_{i}'] = cbe.transform(test[f'nom_{i}'])



# ord 5

cbe = CatBoostEncoder()

train['ord_5'] = cbe.fit_transform(train['ord_5'], train_targets)

test['ord_5'] = cbe.transform(test['ord_5'])
%%time



ohe_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4',

            'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4',

            'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4',

            'day', 'month']



# ColumnTransformer enables applying OneHotEncoder to the entire dataframe

transformer = ColumnTransformer(

    [

        ("ohe",

         OneHotEncoder(sparse=True, drop='first'),

         ohe_cols

         )

    ], remainder='passthrough'

)

train = transformer.fit_transform(train)

test = transformer.fit_transform(test)
%%time



from sklearn.linear_model import LogisticRegressionCV

clf = LogisticRegressionCV(cv=5, 

                           scoring='roc_auc', 

                           random_state=42, 

                           verbose=True, 

                           n_jobs=-1,

                           max_iter = 1000)

clf.fit(train, train_targets)
np.mean(clf.scores_[1])
preds = clf.predict_proba(test)[:, 1]

preds = pd.DataFrame(list(zip(test_ids, preds)), columns = ['id', 'target'])

preds.sort_values(by=['id'], inplace = True)



preds.to_csv("./my_submission.csv", index=False)