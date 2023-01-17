# ライブラリのインポート

import gc



import numpy as np

np.random.seed(71)

import scipy as sp

import pandas as pd

from pandas import DataFrame



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from category_encoders import OrdinalEncoder

from tqdm import tqdm_notebook as tqdm



from rgf.sklearn import FastRGFClassifier

from rgf.utils import cleanup



import lightgbm as lgb

from lightgbm import LGBMClassifier



from keras.layers import Input, Dense ,Dropout, BatchNormalization

from keras.layers import Dropout

from keras.optimizers import Adam, SGD

from keras.models import Model

from keras.callbacks import EarlyStopping
def CountEncoder(X_train, X_test, col):

    X_concat = pd.concat([X_train, X_test])

    

    summary = X_concat[col].value_counts()

    

    del X_concat

    gc.collect()

    

    return X_train[col].map(summary), X_test[col].map(summary)
def TargetEncoder(X_train, y_train, X_test, target, col, alpha = 0.5, min_samples_leaf = 10, smooth_coeff = 1.0):

    X_temp = X_train.copy()

    X_temp[target] = y_train

    

    global_mean = X_temp[target].astype(float).mean()

    summary = X_temp[[col, target]].groupby([col])[target].agg(['mean', 'count'])

    

    smoove = 1 / (1 + np.exp(-(summary['count'] - min_samples_leaf) / smooth_coeff))

    smoothing = global_mean * (1 - smoove) + summary['mean'] * smoove

    smoothing[summary['count'] == 1] = global_mean 

    

    del X_temp

    

    return X_test[col].map(smoothing)
def negative_downsample(X_train, y_train, text_train, ratio=1, seed=71):

    X_train0 = X_train[y_train==0]

    X_train1 = X_train[y_train==1]

    

    n_samples = int(ratio*len(X_train1))

    

    X_train0_sample = X_train0.sample(n=n_samples, replace=False, random_state=seed)

    

    X_train = pd.concat([X_train0_sample, X_train1])

    

    y_train = y_train.loc[X_train.index]

    text_train = text_train.loc[X_train.index]

    

    return X_train, y_train, text_train
# データの読み込み

df_train = pd.read_csv('../input/train.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'])

df_test = pd.read_csv('../input/test.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'])
df_train = df_train[df_train.issue_d.dt.year >= 2011]
# X, yに分離する。

y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = df_test.copy()



del df_train, df_test

gc.collect()
# クレジットライン開始日から申請日までの日数をカウントする

X_train['sinse_earliest_cr_line'] = (X_train.issue_d - X_train.earliest_cr_line) / pd.Timedelta(days=1)

X_test['sinse_earliest_cr_line'] = (X_test.issue_d - X_test.earliest_cr_line) / pd.Timedelta(days=1)



X_train.drop(['issue_d', 'earliest_cr_line'], axis=1, inplace=True)

X_test.drop(['issue_d', 'earliest_cr_line'], axis=1, inplace=True)
# テキスト特徴量を分離する

text_train = X_train[['emp_title']].fillna('#').copy()

text_test = X_test[['emp_title']].fillna('#').copy()
# emp_titleのtrainとtestに共通して出現するもの以外はall_othersとしてまとめる

common_titles = np.intersect1d(X_train.emp_title.dropna().unique(), X_test.emp_title.dropna().unique())

print(len(common_titles))



X_train.loc[~X_train.emp_title.isin(common_titles), 'emp_title'] = 'all others'

X_test.loc[~X_test.emp_title.isin(common_titles), 'emp_title'] = 'all others'
# loan_amntはいくつかの典型的な貸付額にピークが立っているのでカテゴリにして拾ってみる



X_train['annual_inc_cat'] = X_train.annual_inc.astype(str)

X_test['annual_inc_cat'] = X_test.annual_inc.astype(str)



X_train['loan_amnt_cat'] = X_train.loan_amnt.astype(str)

X_test['loan_amnt_cat'] = X_test.loan_amnt.astype(str)



X_train['annual_inc_cat_concat_loan_amnt_cat'] = X_train['annual_inc_cat'] + '_' + X_train['loan_amnt_cat']

X_test['annual_inc_cat_concat_loan_amnt_cat'] = X_test['annual_inc_cat'] + '_' + X_test['loan_amnt_cat']





cols = ['annual_inc_cat', 'loan_amnt_cat', 'annual_inc_cat_concat_loan_amnt_cat']

X_concat = pd.concat([X_train, X_test])



for col in cols:

    summary = X_concat[col].value_counts()

    X_train[col] = X_train[col].map(summary)

    X_test[col] = X_test[col].map(summary)

    

del X_concat

gc.collect()
# ローン額を月々の返済額で割って返済期間を見積もる

X_train['loan_period'] = X_train.loan_amnt / X_train.installment

X_test['loan_period'] = X_test.loan_amnt / X_test.installment
# まだdtypeがobjectのカラムを仮にカテゴリとみなす

cat = []



for col in tqdm(X_train.columns):

    if X_train[col].dtype == 'object':

        

        cat.append(col)
# X_testを Count Encoding

for col in cat:

    X_test[col + '_te'] = TargetEncoder(X_train, y_train, X_test, 'loan_condition', col)
# X_trainをoofでTargetEncoding

for col in cat:

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

    temp = np.zeros(len(X_train))



    for i, (train_ix, test_ix) in enumerate(skf.split(X_train, y_train)):

        X_train_, y_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix]

        X_val, y_val = X_train.iloc[test_ix], y_train.iloc[test_ix]

        

        temp[test_ix] = TargetEncoder(X_train_, y_train_, X_val, 'loan_condition', col)

    

    X_train[col + '_te'] = temp
# X_testを Count Encoding

for col in cat:

    X_train[col + 'ce'], X_test[col + 'ce'] = CountEncoder(X_train, X_test, col)
# 元のcategoricalをdrop

X_train.drop(cat, axis=1, inplace=True)

X_test.drop(cat, axis=1, inplace=True)
X_train.to_csv('../X_train_tree.csv')

X_test.to_csv('../X_test_tree.csv')



text_train.to_csv('../text_train_tree.csv')

text_test.to_csv('../text_test_tree.csv')
# データの読み込み

df_train = pd.read_csv('../input/train.csv', index_col=0, parse_dates=['issue_d'])

df_test = pd.read_csv('../input/test.csv', index_col=0, parse_dates=['issue_d'])
# 古すぎて傾向の異なるデータをdrop

df_train = df_train[df_train.issue_d.dt.year >= 2011]
# X, yに分離する。

y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = df_test.copy()



del df_train, df_test

gc.collect()
# 特徴量の型をざっくりチェック

cat = []

num = []

txt = ['emp_title', 'title']



for col in X_train.columns:

    if X_train[col].dtype == 'object':

        if col not in txt:

            cat.append(col)

    else:

        if col != 'issue_d':

            num.append(col)
# 特徴量の型ごとに分ける

cat_train = X_train[cat]

txt_train = X_train[txt]

X_train = X_train[num]



cat_test = X_test[cat]

txt_test = X_test[txt]

X_test = X_test[num]
# 数値をrankGauss

scaler = QuantileTransformer(copy=True, ignore_implicit_zeros=False, n_quantiles=1000,

          output_distribution='normal', random_state=71,

          subsample=100000)

X_train = scaler.fit_transform(X_train.fillna(X_train.median()))

X_test = scaler.transform(X_test.fillna(X_test.median()))
# カテゴリをOne-hot

for col in tqdm(cat):

    ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)

    oe = OrdinalEncoder(return_df=False)

    

    cat_train[col] = oe.fit_transform(cat_train[[col]])

    cat_test[col] = oe.transform(cat_test[[col]])    

    

    train = ohe.fit_transform(cat_train[[col]])

    test = ohe.transform(cat_test[[col]])

    

    X_train = sp.sparse.hstack([X_train, train])

    X_test = sp.sparse.hstack([X_test, test])
# テキストをTFIDF

for (analyzer, n) in [('word',(1, 2)) , ('char',(2, 5))]:

    for f in txt:

        tfidf = TfidfVectorizer(max_features=100000, analyzer=analyzer, ngram_range=n)

        

        train = tfidf.fit_transform(txt_train[f].fillna('#'))

        test = tfidf.transform(txt_test[f].fillna('#'))

        

        X_train = sp.sparse.hstack([X_train, train])

        X_test = sp.sparse.hstack([X_test, test])

        

X_train = X_train.tocsr()

X_test = X_test.tocsr()



del cat_train, cat_test, txt_train, txt_test

gc.collect()
# シンプルなMLP

def create_model(input_len):

    inp = Input(shape=(input_len,), sparse=True) # 疎行列を入れる

    x = Dense(194+np.random.randint(5), activation='relu')(inp)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(64+np.random.randint(5), activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(64+np.random.randint(5), activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    outp = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy', optimizer='adam')

    

    return model
%%time

# 30回学習・予測を行って、Averaging

num_ave = 30

y_pred_nn = np.zeros(X_test.shape[0])



for i in range(num_ave):

    model = create_model(X_train.shape[1])

    model.fit(X_train, y_train, batch_size=512, epochs=2)

    y_pred_nn += model.predict(X_test).ravel()

    

y_pred_nn /= num_ave
X_train = pd.read_csv('../X_train_tree.csv', index_col=0)

X_test = pd.read_csv('../X_test_tree.csv', index_col=0)



text_train = pd.read_csv('../text_train_tree.csv', index_col=0)

text_test = pd.read_csv('../text_test_tree.csv', index_col=0)
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=100000, use_idf=False)



text_train = tfidf.fit_transform(text_train.emp_title)

text_test = tfidf.transform(text_test.emp_title)



X_train = sp.sparse.hstack([X_train, text_train])

X_test = sp.sparse.hstack([X_test, text_test])  
%%time

# 20回学習・予測を行って、Seed Averaging

num_ave = 20

y_pred_lgb = np.zeros(X_test.shape[0])



for i in range(num_ave):

    seed = np.random.randint(99999)

    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

                                        importance_type='split', learning_rate=0.05, max_depth=-1,

                                        min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                                        n_estimators=1000, n_jobs=-1, num_leaves=31, objective=None,

                                        random_state=seed, reg_alpha=1.0, reg_lambda=1.0, silent=True,

                                        subsample=0.9, subsample_for_bin=200000, subsample_freq=0)



    clf.fit(X_train, y_train)

    y_pred_lgb += clf.predict_proba(X_test)[:,1]

    

y_pred_lgb /= num_ave
X_train = pd.read_csv('../X_train_tree.csv', index_col=0).fillna(-99999)

X_test = pd.read_csv('../X_test_tree.csv', index_col=0).fillna(-99999)



text_train = pd.read_csv('../text_train_tree.csv', index_col=0)

text_test = pd.read_csv('../text_test_tree.csv', index_col=0)
%%time

# 20回学習・予測を行って、Balanced Bootstrap

num_ave = 20

y_pred_rgf = np.zeros(X_test.shape[0])



for i in range(num_ave):

    seed = np.random.randint(99999)

    

    # negative downsampling

    X_train_, y_train_, text_train_ = negative_downsample(X_train, y_train, text_train, seed=seed)

    

    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=5000, use_idf=False)

    

    text_train_ = tfidf.fit_transform(text_train_.emp_title)

    text_test_ = tfidf.transform(text_test.emp_title)

    

    X_train_ = sp.sparse.hstack([X_train_, text_train_])

    X_test_ = sp.sparse.hstack([X_test, text_test_])   

    

    

    clf = FastRGFClassifier(calc_prob='sigmoid', data_l2=2.0, l1=1.0, l2=1000.0,

                                 learning_rate=0.1, loss='LS', max_bin=None, max_depth=6,

                                 max_leaf=50, min_child_weight=5.0, min_samples_leaf=5,

                                 n_estimators=1000, n_jobs=-1, opt_algorithm='rgf',

                                 sparse_max_features=80000, sparse_min_occurences=5,

                                 tree_gain_ratio=1.0, verbose=1)



    clf.fit(X_train_, y_train_)

    y_pred_rgf += clf.predict_proba(X_test_)[:,1]

    cleanup()

    

y_pred_rgf /= num_ave
y_pred = 4*sp.stats.rankdata(y_pred_nn) + 4*sp.stats.rankdata(y_pred_lgb) + 3*sp.stats.rankdata(y_pred_rgf)

y_pred /= y_pred.max()
# sample submissionを読み込んで、予測値を代入の後、保存する

submission = pd.read_csv('../input/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred

submission.to_csv('submission.csv')