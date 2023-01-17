import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



import lightgbm as lgb

from lightgbm import LGBMClassifier



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.ensemble import GradientBoostingClassifier
#ハンズオンのスムーズな進行のために全体の20分の1だけ読み込むことにします。

#実際に課題でモデリングする際には"skiprows=lambda x: x%20!=0"を削除してください。

df_train = pd.read_csv('../input/homework-for-students2/train.csv') #, index_col=0,skiprows=lambda x: x%20!=0

df_test = pd.read_csv('../input/homework-for-students2/test.csv')
# DataFrameのshapeで行数と列数を確認してみましょう。

df_train.shape, df_test.shape
# 先頭5行をみてみます。

df_train.head()
df_test.head()
df_train[df_train.loan_condition==1].loan_amnt.mean() # 貸し倒れたローンの平均額
df_train[df_train.loan_condition==0].loan_amnt.mean()# 上の貸し倒れたローンに対するものを参考に、貸し倒れていないローンの平均額を算出みてください。
df_train.describe()
df_test.describe()
f = 'loan_amnt'



plt.figure(figsize=[7,7])

df_train[f].hist(density=True, alpha=0.5, bins=50)

df_test[f].hist(density=True, alpha=0.5, bins=50)

# testデータに対する可視化を記入してみましょう

plt.xlabel(f)

plt.ylabel('density')

plt.show()
f = 'purpose'

df_train[f].value_counts() / len(df_train)

# value_countsを用いてtrainのpurposeに対して集計結果をみてみましょう。
# 同様にtestデータに対して

df_test[f].value_counts() / len(df_test)
y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = df_test
from sklearn.preprocessing import StandardScaler,MinMaxScaler

scaler = MinMaxScaler()



train_scaled = scaler.fit_transform(X_train[['loan_amnt']])

test_scaled = scaler.fit_transform(X_train[['loan_amnt']])



f='loan_amnt'



plt.figure(figsize=[7,7])

df_train[f].hist(density=True, alpha=0.5, bins=50)

plt.xlabel(f)

plt.ylabel('density')

plt.show()
plt.figure(figsize=[7,7])

X_train.annual_inc.hist(bins=20)

plt.show()
# dtypeがobjectのカラム名とユニーク数を確認してみましょう。

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())
X_train['emp_title'].head(10) # カテゴリよりテキストとして扱ったほうが良いかもしれない
col = 'purpose'



encoder = OneHotEncoder()

enc_train = encoder.fit_transform(X_train[col].values)

enc_test = encoder.transform(X_test[col].values)
enc_train.head()
enc_test.head()
#結合用のキーカラムを追加する

#df_train['year'] = df_train.issue_d.dt.year

#df_train['month'] = df_train.issue_d.dt.month

#df_test['year'] = df_test.issue_d.dt.year

#df_test['month'] = df_test.issue_d.dt.month
# Onehotの例を参考にやってみましょう

# https://contrib.scikit-learn.org/categorical-encoding/ordinal.html

col = 'purpose'



encoder = OrdinalEncoder()

enc_train = encoder.fit_transform(X_train[col].values)

enc_test = encoder.transform(X_test[col].values)
enc_train.head()
enc_test.head()
# value_couontsで集計した結果を、

summary = X_train[col].value_counts() # value_countsします

summary
# mapする。

enc_train =  X_train[col].map(summary)

enc_test = X_test[col].map(summary)
enc_train.head()
enc_test.head()
target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



# X_testはX_trainでエンコーディングする

summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary) 



    

# X_trainのカテゴリ変数をoofでエンコーディングする

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)
X_train['loan_condition_y'] = enc_train

X_test['loan_condition_y'] = enc_test
enc_train
enc_test
TXT_train = X_train.emp_title.copy()

TXT_test = X_test.emp_title.copy()



cats.remove('emp_title')
from sklearn.preprocessing import StandardScaler , MinMaxScaler

scaler = MinMaxScaler()

X_train['loan_amnt'] = scaler.fit_transform(X_train[['loan_amnt']])

X_test['loan_amnt'] = scaler.transform(X_test[['loan_amnt']])

# Category Encoding

grade_cols = ['grade', 'sub_grade']



for col in grade_cols:

    unique = pd.unique(X_train[col])

    unique.sort()

    

    items = []

    indicies = []

    for i, item in enumerate(unique):

        items.append(item)

        indicies.append(i)



    grade_vals = pd.Series(indicies, index=items)

    X_train[col] = X_train[col].map(grade_vals)

    X_test[col] = X_test[col].map(grade_vals)
TXT_train
TXT_test
cats
encoder = OrdinalEncoder(cols=cats)

X_train[cats] = encoder.fit_transform(X_train[cats])

X_test[cats] = encoder.fit_transform(X_test[cats])
X_train.head()
X_test.head()
# 以下を参考に自分で書いてみましょう 

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html

X_train.drop(['emp_title'],axis=1,inplace=True)

X_test.drop(['emp_title'],axis=1,inplace=True)



X_train.fillna(-9999,inplace=True)

X_test.fillna(-9999,inplace=True)
# train

#summary = X_train['purpose'].value_counts()

#temp = X_train['purpose'].map(summary)

#temp1 = temp.reset_index()



#X_train = pd.merge(X_train, temp1, on='purpose')
temp1
# test

#summary = X_test['purpose'].value_counts()

#temp = X_test['purpose'].map(summary)

#temp1 = temp.reset_index()

#X_test = pd.merge(X_test, temp1, on='purpose')
X_test.head()
# train

summary = X_train['grade'].value_counts()

X_train['grade_y'] = X_train['grade'].map(summary)

X_train.head()
#test

summary = X_test['grade'].value_counts()

X_test['grade_y'] = X_test['grade'].map(summary)

X_test.head()
# train

summary = X_train['emp_length'].value_counts()

X_train['emp_length_y'] = X_train['emp_length'].map(summary)

#X_train.head()
# test

summary = X_test['emp_length'].value_counts()

X_test['emp_length_y'] = X_test['emp_length'].map(summary)

#X_test.head()
#target = 'loan_condition'

#X_temp = pd.concat([X_train, y_train], axis=1)

#col = 'purpose_x'



# X_testはX_trainでエンコーディングする

#summary = X_temp.groupby([col])[target].mean()

#enc_test = X_test[col].map(summary) 



    

# X_trainのカテゴリ変数をoofでエンコーディングする

#skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)





#enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



#for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

#    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

#    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]



#    summary = X_train_.groupby([col])[target].mean()

#    enc_train.iloc[val_ix] = X_val[col].map(summary)
#enc_train
X_test.head()
# CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。

scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



    clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                                importance_type='split', learning_rate=0.05, max_depth=-1,

                                min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                                n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                                random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                                subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

    

    



    clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val, y_val)])

    

    #clf = GradientBoostingClassifier()

    

    #clf.fit(X_train_, y_train_)

    y_pred = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))
print(np.mean(scores))

print(scores)
X_train
# 全データで再学習し、testに対して予測する

clf.fit(X_train, y_train)



y_pred = clf.predict_proba(X_test)[:,1]
# sample submissionを読み込んで、予測値を代入の後、保存する

# こちらもスムーズな進行のために20分の１に間引いていますが、本番では"skiprows=lambda x: x%20!=0"を削除して用いてください。

submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv')



submission.loan_condition = y_pred

submission.to_csv('submission.csv',index = False) #
submission.head()