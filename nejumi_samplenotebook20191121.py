# TensorFlow/kerasの最新versionで疎行列周りでエラーが出るのでバージョン指定しています。

# 現状では未解決のようで、PyTorch使ったほうがいいかも。

!pip uninstall tensorflow -y

!pip install tensorflow==1.11.0



!pip uninstall keras -y

!pip install keras==2.2.4
import gc

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from sklearn.preprocessing import LabelEncoder



from tqdm import tqdm_notebook as tqdm

from category_encoders import OrdinalEncoder



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



import lightgbm as lgb

from lightgbm import LGBMClassifier
df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0, skiprows=lambda x: x%20!=0)



y_train = df_train.tot_cur_bal

X_train = df_train.drop(['tot_cur_bal'], axis=1)



X_test = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0, skiprows=lambda x: x%20!=0)
X_train = X_train[y_train.isnull()==False]

y_train = y_train[y_train.isnull()==False]
# テキストは除いておく。

X_train.drop(['issue_d', 'emp_title'], axis=1, inplace=True)

X_test.drop(['issue_d', 'emp_title'], axis=1, inplace=True)
from category_encoders import OrdinalEncoder



cats = []



for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

oe = OrdinalEncoder(cols=cats)



X_train = oe.fit_transform(X_train)

X_test = oe.transform(X_test)
X_train.fillna(-99999, inplace=True)

X_test.fillna(-99999, inplace=True)
X_train.head()
X_test.head()
%%time

# CVしてスコアを見てみる

# なお、そもそもStratifiedKFoldが適切なのかは別途考える必要があります

# 次回Build Modelの内容ですが、是非各自検討してみてください

scores = []



skf = KFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    clf = GradientBoostingRegressor() 

    

    clf.fit(X_train_, y_train_)

    y_pred = clf.predict(X_val)

    score = mean_squared_error(y_val, y_pred)**0.5

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))
# Fold_4のactual vs predをプロットしてみよう

plt.figure(figsize=[7,7])

plt.scatter(y_val, y_pred, s=5)

plt.xlabel('actual')

plt.ylabel('pred')

plt.show()
%%time

# 金額系なので、RMSLEで最適化してみる。

# 何も指定しないと大抵はRMSEで最適化される。ここでは対数を取っておくとちょうどRMSLEの最適化に相当する。

scores = []



skf = KFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    clf = GradientBoostingRegressor() 

    

    clf.fit(X_train_, np.log1p(y_train_))

    y_pred = np.expm1(clf.predict(X_val))

    score = mean_squared_log_error(y_val, y_pred)**0.5

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))
# Fold_4のactual vs predをプロットしてみよう

plt.figure(figsize=[7,7])

plt.scatter(y_val, y_pred, s=5)

plt.xlabel('actual')

plt.ylabel('pred')

plt.show()
df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0, skiprows=lambda x: x%20!=0)



y_train = df_train.purpose

X_train = df_train.drop(['purpose'], axis=1)



X_test = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0, skiprows=lambda x: x%20!=0)
y_train 
le = LabelEncoder()

y_train = le.fit_transform(y_train)
# テキストは除いておく。

X_train.drop(['issue_d', 'emp_title'], axis=1, inplace=True)

X_test.drop(['issue_d', 'emp_title'], axis=1, inplace=True)
y_train
cats = []



for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

oe = OrdinalEncoder(cols=cats)



X_train = oe.fit_transform(X_train)

X_test = oe.transform(X_test)
X_train.fillna(-99999, inplace=True)

X_test.fillna(-99999, inplace=True)
%%time

# CVしてスコアを見てみる

# なお、そもそもStratifiedKFoldが適切なのかは別途考える必要があります

scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train[test_ix]

    

    clf = GradientBoostingClassifier() 

    

    clf.fit(X_train_, y_train_)

    y_pred = clf.predict_proba(X_val)

    score = log_loss(y_val, y_pred)

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))
y_pred
labels = np.argmax(y_pred, axis=1)

labels
# 予測値を元のラベルに戻してみる。

le.inverse_transform(labels)
le.inverse_transform(y_val)
df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0, skiprows=lambda x: x%20!=0)



y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0, skiprows=lambda x: x%20!=0)
# テキストは除いておく。

X_train.drop(['issue_d', 'emp_title'], axis=1, inplace=True)

X_test.drop(['issue_d', 'emp_title'], axis=1, inplace=True)
cats = []



for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

oe = OrdinalEncoder(cols=cats)



X_train = oe.fit_transform(X_train)

X_test = oe.transform(X_test)
X_train.fillna(-99999, inplace=True)

X_test.fillna(-99999, inplace=True)
# あとでまた使うので保存しておく

X_train.to_csv('../X_train_tree.csv')

X_test.to_csv('../X_test_tree.csv')
%%time

# CVしてスコアを見てみる

# なお、そもそもStratifiedKFoldが適切なのかは別途考える必要があります

scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    clf = GradientBoostingClassifier() 

    

    clf.fit(X_train_, y_train_)

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))
scores = np.array(scores)
scores.mean(), scores.std()
df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0, parse_dates=['issue_d'], skiprows=lambda x: x%20!=0)

y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0, parse_dates=['issue_d'], skiprows=lambda x: x%20!=0)



del df_train

gc.collect()
X_train.issue_d.head()
cats = []



for col in X_train.columns:

    if X_train[col].dtype == 'object':

        if col != 'issue_d':

            cats.append(col)

        

oe = OrdinalEncoder(cols=cats)



X_train = oe.fit_transform(X_train)

X_test = oe.transform(X_test)
year = X_train.issue_d.dt.year



X_train.drop(['issue_d', 'emp_title'], axis=1, inplace=True)

X_test.drop(['issue_d', 'emp_title'], axis=1, inplace=True)
X_train.fillna(-99999, inplace=True)

X_test.fillna(-99999, inplace=True)
X_train_, y_train_ = X_train[year < 2015], y_train[year < 2015]

X_val, y_val = X_train[year >= 2015], y_train[year >= 2015]
clf = GradientBoostingClassifier() 



clf.fit(X_train_, y_train_)

y_pred = clf.predict_proba(X_val)[:,1]

score = roc_auc_score(y_val, y_pred)



print('Time Split Score is %f' % (score))
df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0, skiprows=lambda x: x%20!=0)



y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0, skiprows=lambda x: x%20!=0)



del df_train

gc.collect()
# 住所（州）をグループ識別子として分離しておく。日付とテキストも除いておく。

groups = X_train.addr_state.values



X_train.drop(['issue_d', 'emp_title', 'addr_state'], axis=1, inplace=True)

X_test.drop(['issue_d', 'emp_title', 'addr_state'], axis=1, inplace=True)
groups
cats = []



for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

oe = OrdinalEncoder(cols=cats)



X_train = oe.fit_transform(X_train)

X_test = oe.transform(X_test)
X_train.fillna(-99999, inplace=True)

X_test.fillna(-99999, inplace=True)
score
gkf = GroupKFold(n_splits=5)

scores = []



for i, (train_ix, test_ix) in enumerate(tqdm(gkf.split(X_train, y_train, groups))):

    

    X_train_, y_train_, groups_train_ = X_train.iloc[train_ix], y_train.iloc[train_ix], groups[train_ix]

    X_val, y_val, groups_val = X_train.iloc[test_ix], y_train.iloc[test_ix], groups[test_ix]

    

    print('Train Groups', np.unique(groups_train_))

    print('Val Groups', np.unique(groups_val))

    

    clf = GradientBoostingClassifier(n_estimators=1) 

    

    clf.fit(X_train_, y_train_)

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))

    print('\n')
clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                                importance_type='split', learning_rate=0.05, max_depth=-1,

                                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                                n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                                random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
%%time

clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])
imp = DataFrame(clf.booster_.feature_importance(importance_type='gain'), index = X_train.columns, columns=['importance']).sort_values(['importance'], ascending=False)

imp
imp.shape
use_col = imp.index[:10]
X_train_[use_col]
fig, ax = plt.subplots(figsize=(5, 8))

lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')
y_pred = clf.predict_proba(X_val)[:,1]

y_pred
df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0, parse_dates=['issue_d'], skiprows=lambda x: x%20!=0)

#df_train = pd.read_csv('../input/train.csv', index_col=0)

y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0, parse_dates=['issue_d'], skiprows=lambda x: x%20!=0)



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
# train/test

# 特徴量タイプごとに分割する

cat_train = X_train[cat]

txt_train = X_train.emp_title

X_train = X_train[num]



cat_test = X_test[cat]

txt_test = X_test.emp_title

X_test = X_test[num]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train.fillna(X_train.median()))

X_test = scaler.transform(X_test.fillna(X_test.median()))
from sklearn.preprocessing import OneHotEncoder

from category_encoders import OrdinalEncoder

from tqdm import tqdm_notebook as tqdm
for col in tqdm(cat):

    ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)

    oe = OrdinalEncoder(return_df=False)

    

    cat_train[col] = oe.fit_transform(cat_train[[col]])

    cat_test[col] = oe.transform(cat_test[[col]])    

    

    train = ohe.fit_transform(cat_train[[col]])

    test = ohe.transform(cat_test[[col]])

    

    X_train = sp.sparse.hstack([X_train, train])

    X_test = sp.sparse.hstack([X_test, test])
X_train.shape, X_test.shape
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=100000, analyzer='word', ngram_range=(1, 2))
train = tfidf.fit_transform(txt_train.fillna('#'))

test = tfidf.transform(txt_test.fillna('#'))



X_train = sp.sparse.hstack([X_train, train])

X_test = sp.sparse.hstack([X_test, test])



X_train = X_train.tocsr()# 行方向のスライスができるように変換する

X_test = X_test.tocsr()
del cat_train, cat_test, txt_train, txt_test

gc.collect()
num_train = int(X_train.shape[0]*0.7)



X_train_ = X_train[:num_train, :]

y_train_ = y_train[:num_train]



X_val = X_train[num_train:, :]

y_val = y_train[num_train:]
from keras.layers import Input, Dense ,Dropout, BatchNormalization

from keras.optimizers import Adam, SGD

from keras.models import Model

from keras.callbacks import EarlyStopping
# シンプルなMLP



def create_model(input_dim):

    inp = Input(shape=(input_dim,), sparse=True) # 疎行列を入れる

    x = Dense(194, activation='relu')(inp)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(64, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    x = Dense(64, activation='relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(0.5)(x)

    outp = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy', optimizer='adam')

    

    return model
X_train_.shape
model = create_model(X_train_.shape[1])



es = EarlyStopping(monitor='val_loss', patience=0)



model.fit(X_train_, y_train_, batch_size=32, epochs=999, validation_data=(X_val, y_val), callbacks=[es])
roc_auc_score(y_val, model.predict(X_val))
X_train = pd.read_csv('../X_train_tree.csv', index_col=0)

X_test = pd.read_csv('../X_test_tree.csv', index_col=0)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.ensemble import GradientBoostingClassifier



skf = StratifiedKFold(n_splits=3, random_state=71, shuffle=True)

gkf = GroupKFold(n_splits=3, )



param_grid = {'learning_rate': [0.05],

                    'max_depth':  np.linspace(5,12,4,dtype = int),

                    'colsample_bytree': np.linspace(0.5, 1.0, 3),

                    'random_state': [71]}



fit_params = {"early_stopping_rounds": 20,

                    "eval_metric": 'auc',

                    "eval_set": [(X_val, y_val)]}



clf  = LGBMClassifier(n_estimators=9999, n_jobs=1)



gs = GridSearchCV(clf, param_grid, scoring='roc_auc',  

                              n_jobs=-1, cv=skf, verbose=True)



gs.fit(X_train_, y_train_, **fit_params,)
gs.best_score_
gs.best_estimator_
from hyperopt import fmin, tpe, hp, rand, Trials

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



from lightgbm import LGBMClassifier
def objective(space):

    scores = []



    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



    for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



        clf = LGBMClassifier(n_estimators=9999, **space) 



        clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

        y_pred = clf.predict_proba(X_val)[:,1]

        score = roc_auc_score(y_val, y_pred)

        scores.append(score)

        

    scores = np.array(scores)

    print(scores.mean())

    

    return -scores.mean()
space ={

        'max_depth': hp.choice('max_depth', np.arange(10, 30, dtype=int)),

        'subsample': hp.uniform ('subsample', 0.8, 1),

        'learning_rate' : hp.quniform('learning_rate', 0.025, 0.5, 0.025),

        'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05)

    }
trials = Trials()



best = fmin(fn=objective,

              space=space, 

              algo=tpe.suggest,

              max_evals=20, 

              trials=trials, 

              rstate=np.random.RandomState(71) 

             )
LGBMClassifier(**best)
trials.best_trial['result']