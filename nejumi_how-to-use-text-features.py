import gc

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder



from tqdm import tqdm_notebook as tqdm
df_train = pd.read_csv('../input/homework-for-students4plus/train.csv', index_col=0, parse_dates=['issue_d'], skiprows=lambda x: x%20!=0)

#df_train = pd.read_csv('../input/homework-for-students4plus/train.csv', index_col=0, parse_dates=['issue_d'])

y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = pd.read_csv('../input/homework-for-students4plus/test.csv', index_col=0, parse_dates=['issue_d'])



del df_train

gc.collect()
cat = []

num = []



for col in X_train.columns:

    if X_train[col].dtype == 'object':

        if col != 'emp_title':

            cat.append(col)

    else:

        if col != 'issue_d':

            num.append(col)

            

cat_train = X_train[cat]

txt_train = X_train.emp_title

X_train = X_train[num]



cat_test = X_test[cat]

txt_test = X_test.emp_title

X_test = X_test[num]
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train.fillna(X_train.median()))

X_test = scaler.transform(X_test.fillna(X_test.median()))
for col in tqdm(cat):

    ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)

    oe = OrdinalEncoder(return_df=False)

    

    cat_train[col] = oe.fit_transform(cat_train[[col]])

    cat_test[col] = oe.transform(cat_test[[col]])    

    

    train = ohe.fit_transform(cat_train[[col]])

    test = ohe.transform(cat_test[[col]])

    

    X_train = sp.sparse.hstack([X_train, train]) # numericにconcatしていく

    X_test = sp.sparse.hstack([X_test, test])
tfidf = TfidfVectorizer(max_features=100000, analyzer='word', ngram_range=(1, 2))



train = tfidf.fit_transform(txt_train.fillna('#'))

test = tfidf.transform(txt_test.fillna('#'))



X_train = sp.sparse.hstack([X_train, train]) # numeric, categoricalにconcatする

X_test = sp.sparse.hstack([X_test, test])



X_train = X_train.tocsr()# 行方向のスライスができるように変換する

X_test = X_test.tocsr()



del cat_train, cat_test, txt_train, txt_test

gc.collect()
X_train.shape, X_test.shape
num_train = int(X_train.shape[0]*0.7)



X_train_ = X_train[:num_train, :]

y_train_ = y_train[:num_train]



X_val = X_train[num_train:, :]

y_val = y_train[num_train:]
clf = LogisticRegression(C=0.01)

clf.fit(X_train_, y_train_)

y_pred = clf.predict_proba(X_val)[:,1]



print(roc_auc_score(y_val, y_pred)) # 検定スコア
clf = LogisticRegression(C=0.01)

clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)[:,1]
df_submit = pd.read_csv('../input/homework-for-students4plus/sample_submission.csv', index_col=0)

df_submit['loan_condition'] = y_pred

df_submit.to_csv('submission.csv')