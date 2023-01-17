import pandas as pd 

import numpy as np
DATA_DIR = "/kaggle/input/summeranalytics2020/"



train = pd.read_csv(DATA_DIR+"train.csv", index_col="Id")

test = pd.read_csv(DATA_DIR+"test.csv", index_col="Id")

train.head()
train.info()
from sklearn.preprocessing import LabelEncoder



def train_cats(train_df, test_df):

    for col in train_df.columns:

        if train_df[col].dtype == 'O' :

            le = LabelEncoder()

            train_df[col] = le.fit_transform(train_df[col])

            test_df[col] = le.transform(test_df[col])

    return train_df, test_df

            

train, test = train_cats(train, test)

train.info()
X = train.drop(columns="Attrition")

y = train.Attrition

X.shape, y.shape, test.shape
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_val_score
rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=7, min_samples_leaf=5)

cross_val_score(rfc, X, y, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc').mean()
DATA_DIR = "/kaggle/input/summeranalytics2020/"



train = pd.read_csv(DATA_DIR+"train.csv", index_col="Id")

test = pd.read_csv(DATA_DIR+"test.csv", index_col="Id")

train.head()
print(f"Number of duplicates in the training data are {train.duplicated().sum()} of {len(train)}, ie {(100* train.duplicated().sum()/len(train)).round(2)} % of data duplicated")

print(f"Number of duplicates in the testing data are {test.duplicated().sum()} of {len(test)}, ie {(100* test.duplicated().sum()/len(test)).round(2)} % of data duplicated")
train.drop_duplicates(inplace=True)

print(f"Number of duplicates in the training data are {train.duplicated().sum()} of {len(train)}, ie {(100* train.duplicated().sum()/len(train)).round(2)} % of data duplicated")
from sklearn.preprocessing import LabelEncoder



def train_cats(train_df, test_df):

    for col in train_df.columns:

        if train_df[col].dtype == 'O' :

            le = LabelEncoder()

            train_df[col] = le.fit_transform(train_df[col])

            test_df[col] = le.transform(test_df[col])

    return train_df, test_df

            

train, test = train_cats(train, test)

train.info()
X = train.drop(columns="Attrition")

y = train.Attrition

X.shape, y.shape, test.shape
rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=7, min_samples_leaf=5)

cross_val_score(rfc, X, y, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc').mean()
lgbmc = LGBMClassifier(random_state=7, n_estimators=100, colsample_bytree=0.5, 

                       max_depth=2, learning_rate=0.1, boosting_type='gbdt')

cross_val_score(lgbmc, X, y, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc').mean()
xgbc = XGBClassifier(seed=7, n_jobs=-1, n_estimators=100, random_state=7, max_depth=2, learning_rate=0.1)

cross_val_score(xgbc, X, y, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc').mean()
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(estimators=[('rfc', rfc), ('xgbc', xgbc), ('lgbmc', lgbmc)],

                                         voting='soft', n_jobs=-1)

cross_val_score(ensemble, X, y, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc').mean()
ensemble.fit(X, y)

y_pred = ensemble.predict_proba(test)[:, 1]

sub_df = pd.DataFrame({"Id":test.index, "Attrition": y_pred})

sub_df.to_csv("SA_submission_2.csv", index=False)