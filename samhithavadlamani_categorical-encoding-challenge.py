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
# LIBRARIES 

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
# DATA

TRAIN=pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

TEST=pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')

labels=TRAIN["target"]

X_train=TRAIN.drop("target", axis=1)

df=pd.concat([X_train, TEST])

df.info()
# cat columns 

cat_col=[c for c in df.columns if df[c].dtypes=='object']

n_levels=df[cat_col].nunique()

print("cardinality of categorical columns:\n",n_levels)



high_cardinal=[c for c in cat_col if df[c].nunique()>100]

low_cardinal=list(set(cat_col)-set(high_cardinal))



# # LABEL ENCODER (high cardinal)

# from sklearn.preprocessing import LabelEncoder

# enc=LabelEncoder()

# for c in high_cardinal:

# #     #drop rows from test that have unencoded labels different from train data

# #     TEST.drop(TEST.loc[~TEST[c].isin(X_train[c])].index,inplace=True)

# #     X_train=enc.fit_transform(X_train)

# #     TEST=enc.transform(TEST)

#     # full data encode since different levels in test

#     df[c]=enc.fit_transform(df[c])



# # split back to train and test

# X_train=df.iloc[0:300000,:]

# TEST=df.iloc[300000:,:]





# OHE (low cardinal)

from sklearn.preprocessing import OneHotEncoder



# Produces 1,0 data columns corresponding to all the unique categorical entries in low_cardinal columns list

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_X_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinal]))

OH_TEST = pd.DataFrame(OH_encoder.transform(TEST[low_cardinal]))



#OH encoding removes index in the data set. Putting the index back again

OH_X_train.index = X_train.index

OH_TEST.index = TEST.index



# Remove low cardinal columns (will replace with one-hot encoding)

num_X_train = X_train.drop(low_cardinal, axis=1)

num_TEST= TEST.drop(low_cardinal, axis=1)

print(num_X_train.shape)



# Add one-hot encoded columns to the original data

X_train = pd.concat([num_X_train, OH_X_train], axis=1)

TEST = pd.concat([num_TEST, OH_TEST], axis=1)







# # BINARY ENCODING 

# from category_encoders import BinaryEncoder

# encoder = BinaryEncoder(cols=low_cardinal)

# X_train = encoder.fit_transform(X_train)

# TEST = encoder.transform(TEST)



# # TARGET ENCODING (high cardinal)

# from category_encoders import TargetEncoder

# t_enc = TargetEncoder()

# X_target=X_train.copy()

# X_target['target']=labels

# X_target[high_cardinal] = t_enc.fit_transform(X_target[high_cardinal], X_target['target'])

# TEST[high_cardinal] = t_enc.transform(X_target[high_cardinal])

# X_train=X_target.drop(['target'], axis=1)



# K-FOLD TARGET ENCODING (high cardinal)

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.base import clone

from sklearn.model_selection import check_cv, KFold



from category_encoders import CatBoostEncoder



class TargetEncoderCV(BaseEstimator, TransformerMixin):



    def __init__(self, cv, **cbe_params):

        self.cv = cv

        self.cbe_params = cbe_params



    @property

    def _n_splits(self):

        return check_cv(self.cv).n_splits



    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:

        self.cbe_ = []

        cv = check_cv(self.cv)



        cbe = CatBoostEncoder(

            cols=X.columns.tolist(),

            return_df=False,

            **self.cbe_params

        )



        X_transformed = np.zeros_like(X, dtype=np.float64)

        for train_idx, valid_idx in cv.split(X, y):

            self.cbe_.append(

                clone(cbe).fit(X.loc[train_idx], y[train_idx])

            )

            X_transformed[valid_idx] = self.cbe_[-1].transform(

                X.loc[valid_idx]

            )



        return pd.DataFrame(X_transformed, columns=X.columns)



    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        X_transformed = np.zeros_like(X, dtype=np.float64)

        for cbe in self.cbe_:

            X_transformed += cbe.transform(X) / self._n_splits

        return pd.DataFrame(X_transformed, columns=X.columns)



te_cv = TargetEncoderCV(KFold(n_splits=5))

X_train = te_cv.fit_transform(X_train, labels)

TEST = te_cv.transform(TEST)

print(X_train.shape)

# VIS 



# target variable distribution - bar plot

labels.value_counts().sort_index().plot.bar()



#correlation between target and all other variables

X_train["target"]=labels

corr=X_train.corrwith(X_train["target"])

X_train=X_train.drop("target", axis=1)



# feature selection

feat=[c for c in corr.index if np.abs(corr[c])>0.05]

feat.remove('target')

# MODEL

#Logreg

# from sklearn.linear_model import LogisticRegression

# clf=LogisticRegression(C=1, solver="lbfgs", max_iter=5000) 



# clf.fit(X_train, labels)

# pred=clf.predict_proba(TEST)[:,1]

# print(pred)



#random forest (giving better accuracy than log reg)

from sklearn.ensemble import RandomForestRegressor 

rfr = RandomForestRegressor(n_estimators = 100, random_state = 0) 

rfr.fit(X_train, labels)

pred=rfr.predict(TEST)

print(pred.shape)



subm_df = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')

subm_df['target'] = pred

subm_df.to_csv('bakaito_submission.csv', index=False)
