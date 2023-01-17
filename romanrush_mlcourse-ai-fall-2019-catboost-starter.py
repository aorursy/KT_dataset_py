# 1



import warnings

import numpy as np

import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
PATH_TO_DATA = Path('../input/flight-delays-fall-2018/')
train_df = pd.read_csv(PATH_TO_DATA / 'flight_delays_train.csv')
train_df.head()
test_df = pd.read_csv(PATH_TO_DATA / 'flight_delays_test.csv')
test_df.head()
train_df['flight'] = train_df['Origin'] + '-->' + train_df['Dest']

test_df['flight'] = test_df['Origin'] + '-->' + test_df['Dest']
categ_feat_idx = np.where(train_df.drop('dep_delayed_15min', axis=1).dtypes == 'object')[0]

categ_feat_idx

# import OneHotEncoder

from sklearn.preprocessing import OneHotEncoder

# instantiate OneHotEncoder

ohe = OneHotEncoder(categorical_features = categorical_feature_mask, sparse=False ) 

# categorical_features = boolean mask for categorical columns

# sparse = False output an array not sparse matrix

# apply OneHotEncoder on categorical feature columns

X_ohe = ohe.fit_transform(train_df) # It returns an numpy array
X_train = train_df.drop('dep_delayed_15min', axis=1).values

#y_train = train_df['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values

y_train = train_df['dep_delayed_15min'].values

X_test = test_df.values
categorical_feature_mask = X_train.dtypes==object
# Categorical boolean mask

categorical_feature_mask = train_df.dtypes==object

# filter categorical columns using mask and turn it into a list

categorical_cols = train_df.columns[categorical_feature_mask].tolist()

# import labelencoder

from sklearn.preprocessing import LabelEncoder

# instantiate labelencoder object

le = LabelEncoder()

# apply le on categorical feature columns

train_df[categorical_cols] = train_df[categorical_cols].apply(lambda col: le.fit_transform(col))

train_df[categorical_cols].head(10)
X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, 

                                                                test_size=0.3, 

                                                                random_state=17)
ctb = CatBoostClassifier(random_seed=17, silent=False)
%%time

ctb.fit(X_train_part, y_train_part,

        cat_features=categ_feat_idx);
ctb_valid_pred = ctb.predict_proba(X_valid)[:, 1]
roc_auc_score(y_valid, ctb_valid_pred)
%%time

ctb.fit(X_train, y_train,

        cat_features=categ_feat_idx);
ctb_test_pred = ctb.predict_proba(X_test)[:, 1]
with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    

    sample_sub = pd.read_csv(PATH_TO_DATA / 'sample_submission.csv', 

                             index_col='id')

    sample_sub['dep_delayed_15min'] = ctb_test_pred

    sample_sub.to_csv('ctb_pred.csv')
!head ctb_pred.csv