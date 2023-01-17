import gc

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



# エンコーダー

from sklearn.preprocessing import quantile_transform

from category_encoders import OrdinalEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm_notebook as tqdm



# モデリング

from sklearn.ensemble import GradientBoostingClassifier

#import lightgbm as lgb

from lightgbm import LGBMClassifier



#交差検定や確認時に利用

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score, log_loss
df_train = pd.read_csv('../input/homework-for-students2/train.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'])

df_test = pd.read_csv('../input/homework-for-students2/test.csv', index_col=0, parse_dates=['issue_d', 'earliest_cr_line'])
# 古いデータはtot_coll_amtやtot_cur_balに値が入っておらず、信ぴょう性が低いため、2013年以降のデータを予測に利用

df_train = df_train[df_train['issue_d'].dt.year >= 2013]
y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)

X_test = df_test

del df_train, df_test

gc.collect()
X_all = pd.concat([X_train, X_test], axis=0)
# 融資実施日とcredit lin開通月の月数(特徴量の追加)

if X_all.earliest_cr_line is not None:

    X_all['cr_month'] = (X_all.issue_d.dt.year * 12 + X_all.issue_d.dt.month) - (X_all.earliest_cr_line.dt.year * 12 + X_all.earliest_cr_line.dt.month)
# 日付をYYYYMMの数字形式に変換。issue_dは将来予測に意味がないため削除

X_all = X_all.drop(['issue_d'], axis=1)

X_all['earliest_cr_line'] = X_all.earliest_cr_line.dt.year * 100 + X_all.earliest_cr_line.dt.month
# 特徴量追加で使う項目は、medianで補完

X_all['dti'].fillna(X_all['dti'].median(), inplace=True)
#　金額特徴量の比率を計算

X_all['loan1'] = X_all['loan_amnt'] / X_all['installment']

X_all['loan2'] = X_all['loan_amnt'] / X_all['annual_inc']

X_all['loan3'] = X_all['loan_amnt'] / X_all['revol_bal']

X_all['loan4'] = X_all['loan_amnt'] / X_all['tot_cur_bal']



X_all['installment1'] = X_all['installment'] / X_all['annual_inc']

X_all['installment2'] = X_all['installment'] / X_all['revol_bal']

X_all['installment3'] = X_all['installment'] / X_all['tot_cur_bal']



X_all['revol_bal1'] = X_all['revol_bal'] / X_all['annual_inc']

X_all['revol_bal2'] = X_all['revol_bal'] / X_all['tot_cur_bal']



X_all['tot_cur_bal1'] = X_all['tot_cur_bal'] / X_all['annual_inc']



X_all['dti1'] = X_all['dti'] / 100 * X_all['annual_inc'] / 12

X_all['dti2'] = X_all['dti1'] - X_all['installment']



# 0除算によるエラー対応

X_all.replace([np.inf, -np.inf], np.nan, inplace=True)



X_all.describe()
nums = []

for col in X_all.columns:

    if X_all[col].dtype != 'object':

        nums.append(col)

        print(col, X_all[col].nunique())
X_all[nums] = quantile_transform(X_all[nums], n_quantiles=100, random_state=0, output_distribution='normal')
# gradeとsub_gradeは順序があるため、順序に応じて変換する。

grade_map = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}

sub_grade_map = {'A1':1,'A2':2,'A3':3,'A4':4,'A5':5,

                 'B1':6,'B2':7,'B3':8,'B4':9,'B5':10, 

                 'C1':11,'C2':12,'C3':13,'C4':14,'C5':15, 

                 'D1':16,'D2':17,'D3':18,'D4':19,'D5':10, 

                'E1':21,'E2':22,'E3':23,'E4':24,'E5':25, 

                 'F1':26,'F2':27,'F3':28,'F4':29,'F5':20, 

                 'G1':31,'G2':32,'G3':33,'G4':34,'G5':35}

X_all['grade'] = X_all['grade'].map(grade_map)

X_all['sub_grade'] = X_all['sub_grade'].map(sub_grade_map)
# emp_lengthは勤務年数なので、数値化する。

emp_len_map = {'< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,

               '6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10}

X_all['emp_length'] = X_all['emp_length'].map(emp_len_map)

X_all['emp_length'].fillna(-1,inplace=True)
# zip_codeは先頭3文字切り出し

X_all['zip_code'] = X_all['zip_code'].str[:3]
# 学習データとテストデータでtitleに差があるため、共通しないtitleを置換

X_all['title'] = X_all['title'].str.lower()

stitle = set(X_test['title'].str.lower()) ^ set(X_train['title'].str.lower())

X_all.loc[X_all['title'].isin(stitle),'title'] = '#train only#'

X_all['title'].fillna('#null#',inplace=True)
# emp_titleはテキスト化で行うため、一旦削除

txt_all = X_all['emp_title'].str.lower()

X_all.drop(['emp_title'], axis=1, inplace=True)
# カテゴリのユニーク数（学習データとテストデータも出力するとtitleに大きな差異があった）

cats = []

for col in X_all.columns:

    if X_all[col].dtype == 'object':

        cats.append(col)

        print(col, X_all[col].nunique())
X_all['grade_cnt'] = X_all['grade'].map(X_all['grade'].value_counts())

X_all['sub_grade_cnt'] = X_all['sub_grade'].map(X_all['sub_grade'].value_counts())

X_all['emp_length_cnt'] = X_all['emp_length'].map(X_all['emp_length'].value_counts())

X_all['zip_code_cnt'] = X_all['zip_code'].map(X_all['zip_code'].value_counts())

#X_all['emp_title_cnt'] = X_all['emp_title'].map(X_all['emp_title'].value_counts())

X_all['title_cnt'] = X_all['title'].map(X_all['title'].value_counts())

X_all['home_ownership_cnt'] = X_all['home_ownership'].map(X_all['home_ownership'].value_counts())

X_all['purpose_cnt'] = X_all['purpose'].map(X_all['purpose'].value_counts())

X_all['addr_state_cnt'] = X_all['addr_state'].map(X_all['addr_state'].value_counts())

X_all['initial_list_status_cnt'] = X_all['initial_list_status'].map(X_all['initial_list_status'].value_counts())

X_all['application_type_cnt'] = X_all['application_type'].map(X_all['application_type'].value_counts())
# カテゴリをエンコーディング

encoder = OrdinalEncoder(cols=cats)

X_all[cats] = encoder.fit_transform(X_all[cats])
tfidf = TfidfVectorizer(strip_accents='ascii', stop_words='english', max_features=300, ngram_range=(1,2))

txt_all = tfidf.fit_transform(txt_all.fillna('#'))

X_all = pd.concat([X_all, pd.DataFrame(txt_all.todense(), index=X_all.index)], axis=1)
X_all.fillna(-9999, inplace=True)
# トレーニングデータ・テストデータに分割

X_train = X_all.iloc[:X_train.shape[0], :]

X_test = X_all.iloc[X_train.shape[0]:, :]

del X_all

gc.collect()
skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

scores = []

total_score = 0

y_tests = np.zeros(len(X_test.index))



clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

                     importance_type='split', learning_rate=0.05, max_depth=-1,

                     min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                     n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

                     random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

                     subsample=0.9, subsample_for_bin=200000, subsample_freq=0)



for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    clf.fit(X_train_, y_train_, early_stopping_rounds=200, eval_metric='auc', eval_set=[(X_val, y_val)])

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    y_tests += clf.predict_proba(X_test)[:,1]
y_pred = y_tests/len(scores)
# sample submissionを読み込んで、予測値を代入の後、保存する

# こちらもスムーズな進行のために20分の１に間引いていますが、本番では"skiprows=lambda x: x%20!=0"を削除して用いてください。

submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv', index_col=0)

submission.loan_condition = y_pred

submission.to_csv('submission.csv')