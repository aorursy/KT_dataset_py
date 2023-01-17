import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#Some basic features:

train['question_len']   = train['question_text'].apply(lambda x: len(str(x)))
test['question_len']   = test['question_text'].apply(lambda x: len(str(x)))

df_all = pd.concat([train, test], axis=0)
print(train.shape)
print(train.head())
print(train.columns)


from sklearn.feature_extraction.text import TfidfVectorizer

corpus_3 = ["one two three four ",
           "one two three  ",
           "one two ",
           "one"]
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus_3)
idf = vectorizer.idf_
print(dict(zip(vectorizer.get_feature_names(), idf)))
print('Preprocessing text...')
cols = ["question_text"]
n_features = [
    400, # Number of different features for project_title
    4040, 
    400
]

for c_i, c in tqdm(enumerate(cols)):
    tfidf = TfidfVectorizer(max_features=n_features[c_i], min_df=3)
    tfidf.fit(df_all[c])
    tfidf_train = np.array(tfidf.transform(train[c]).todense(), dtype=np.float16) 
    tfidf_test = np.array(tfidf.transform(test[c]).todense(), dtype=np.float16)

    for i in range(n_features[c_i]):
        train[c + '_tfidf_' + str(i)] = tfidf_train[:, i]
        test[c + '_tfidf_' + str(i)] = tfidf_test[:, i]
        
    del tfidf, tfidf_train, tfidf_test    
print('Done.')
del df_all

train.reset_index(drop= True)
test.reset_index(drop= True)

pca_columns=train.columns[train.columns.str.contains("_tfidf_")]
train_pca = train[pca_columns]
test_pca = test[pca_columns]

train = train.drop(pca_columns, axis=1, errors='ignore')
test = test.drop(pca_columns, axis=1, errors='ignore')

train = train.merge(train_pca, right_index=True, left_index = True)
test = test.merge(test_pca, right_index=True, left_index = True)
cols_to_drop = ["qid", "question_text", "target"]

X = train.drop(cols_to_drop, axis=1, errors='ignore')
y = train['target']
X_test = test.drop(cols_to_drop, axis=1, errors='ignore')
id_test = test['qid'].values
feature_names = list(X.columns)
print(X.shape, X_test.shape)


# Build the model
cnt = 0
p_buf = []
n_splits = 5
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=0)
auc_buf = []   


for train_index, valid_index in kf.split(X):
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    params = {
          'objective': 'binary',
          'metric': 'auc',
          'boosting_type': 'dart',
          'learning_rate': 0.08,
          'max_bin': 15,
          'max_depth': 10, #17
          'num_leaves': 30, #63
          'subsample': 0.8,
          'subsample_freq': 5,
          'colsample_bytree': 0.8,
          'reg_lambda': 7,
          'num_threads': 4}
    
    model = lgb.train(
        params,
        lgb.Dataset(X.loc[train_index], y.loc[train_index], feature_name=feature_names),
        num_boost_round=10000,
        valid_sets=[lgb.Dataset(X.loc[valid_index], y.loc[valid_index])],
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    if cnt == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        print(tuples[:50])

    p = model.predict(X.loc[valid_index], num_iteration=model.best_iteration)
    auc = roc_auc_score(y.loc[valid_index], p)

    print('{} AUC: {}'.format(cnt, auc))

    p = model.predict(X_test, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p)
    else:
        p_buf += np.array(p)
    auc_buf.append(auc)

    cnt += 1
    if cnt > 0: # Comment this to run several folds
        break
    
    del model
    gc.collect

auc_mean = np.mean(auc_buf)
auc_std = np.std(auc_buf)
print('AUC = {:.6f} +/- {:.6f}'.format(auc_mean, auc_std))

preds = p_buf/cnt
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import plot_importance
train = train.rename(columns = {'<lambda>':'lamda'})
test = test.rename(columns = {'<lambda>':'lamda'})

X = train.drop(cols_to_drop, axis=1, errors='ignore')
y = train['target']
X_test = test.drop(cols_to_drop, axis=1, errors='ignore')
id_test = test['qid'].values
feature_names = list(X.columns)
print(X.shape, X_test.shape)

# Build the model
cnt = 0
p_buf = []
n_splits = 5
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=0)
auc_buf = []   


for train_index, valid_index in kf.split(X):
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    
    xlf = XGBRegressor(
          objective= 'binary:logistic',
          eval_metric= 'auc',
          eta=0.01,
          max_depth= 7,
          subsample =0.8, 
          colsample_bytree= 0.4,
          min_child_weight= 10,
          gamma= 2)
    
    
    #xlf.fit(X.loc[train_index], y.loc[train_index], eval_metric='rmse')
    xlf.fit(X.loc[train_index], y.loc[train_index], eval_metric='rmse', verbose = True, eval_set = [(X[train_index], y[train_index]), (X[valid_index], y[valid_index])], early_stopping_rounds=80)
    p = xlf.predict(X.loc[valid_index])

subm = pd.DataFrame()
subm['qid'] = id_test
subm['prediction'] = preds.round().astype(int)
subm.to_csv('submission.csv', index=False)

