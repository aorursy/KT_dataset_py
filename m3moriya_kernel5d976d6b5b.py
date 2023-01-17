import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.ensemble import GradientBoostingClassifier
pd.options.display.max_columns = None
#ハンズオンのスムーズな進行のために全体の20分の1だけ読み込むことにします。

#実際に課題でモデリングする際には"skiprows=lambda x: x%20!=0"を削除してください。

#tarin_smallであればデータ少ないのでいらないです

df_train = pd.read_csv('../input/homework-for-students2/train.csv')

#df_test = #testデータの読み込みをtrainを参考に書いて見ましょう！
df_test = pd.read_csv('../input/homework-for-students2/test.csv')
y_train = df_train.loan_condition

df_train = df_train.drop(['loan_condition'], axis =1)



df_test = df_test
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
train_scaled = scaler.fit_transform(df_train[['loan_amnt']])

test_scaled = scaler.transform(df_test[['loan_amnt']])
df_loc = pd.read_csv('../input/homework-for-students2/statelatlong.csv')

df_zip = pd.read_csv('../input/homework-for-students2/free-zipcode-database.csv')

df_gdp = pd.read_csv('../input/homework-for-students2/US_GDP_by_State.csv')

df_spi = pd.read_csv('../input/homework-for-students2/spi.csv')
df2_train = pd.concat([df_train, df_train['zip_code'].str.split('xx', expand=True)], axis=1).drop('zip_code', axis=1)

df2_test = pd.concat([df_test, df_test['zip_code'].str.split('xx', expand=True)], axis=1).drop('zip_code', axis=1)
df2_train.rename(columns={0: 'zip_code'}, inplace=True)

df2_test.rename(columns={0: 'zip_code'}, inplace=True)
df2_train = df2_train.drop(1, axis=1)

df2_test = df2_test.drop(1, axis=1)
df2_train = pd.concat([df2_train,df2_train['issue_d'].str.split('-', expand=True)], axis=1).drop('issue_d', axis=1)

df2_test = pd.concat([df2_test,df2_test['issue_d'].str.split('-', expand=True)], axis=1).drop('issue_d', axis=1)
df2_train.rename(columns={0: 'i_month', 1:'i_year'}, inplace=True)

df2_test.rename(columns={0: 'i_month', 1:'i_year'}, inplace=True)
df2_train = pd.concat([df2_train,df2_train['earliest_cr_line'].str.split('-', expand=True)], axis=1).drop('earliest_cr_line', axis=1)

df2_test = pd.concat([df2_test,df2_test['earliest_cr_line'].str.split('-', expand=True)], axis=1).drop('earliest_cr_line', axis=1)
df2_train.rename(columns={0: 'ecl_month', 1:'ecl_year'}, inplace=True)

df2_test.rename(columns={0: 'ecl_month', 1:'ecl_year'}, inplace=True)
#df2_train = df2_train.dropna(subset=['ecl_month'])

#df2_test = df2_test.dropna(subset=['ecl_month'])
look_up = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05',

            'Jun': '06', 'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}



df2_train['i_month'] = df2_train['i_month'].apply(lambda x : look_up[x])

df2_test['i_month'] = df2_test['i_month'].apply(lambda x : look_up[x])



#df2_train['ecl_month'] = df2_train['ecl_month'].apply(lambda x : look_up[x])

#df2_test['ecl_month'] = df2_test['ecl_month'].apply(lambda x : look_up[x])
df_spi['date'].str.split('-', expand=True)
df_spi = pd.concat([df_spi,df_spi['date'].str.split('-', expand=True)], axis=1).drop('date', axis=1)
df_spi.rename(columns={0: 'spi_day', 1:'spi_month', 2:'spi_year'}, inplace=True)
look_up = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05',

            'Jun': '06', 'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}



df_spi['spi_month'] = df_spi['spi_month'].apply(lambda x : look_up[x])
df_gdp=df_gdp[df_gdp['year'] == 2015]



df_gdp.drop('year', axis=1)
df_spi = df_spi.groupby(['spi_year', 'spi_month']).mean()
drop_col = ['RecordNumber', 'City', 'State', 'LocationText', 'Location','Decommisioned','TaxReturnsFiled', 'Notes']
df_zip.drop(drop_col, axis=1)
len(y_train)
len(df_train_tlzgs)
df2_train["zip_code"] = df2_train["zip_code"].astype(str)

df2_test["zip_code"] = df2_test["zip_code"].astype(str)

df_zip["Zipcode"] = df_zip["Zipcode"].astype(str)
df_train_loc = pd.merge(df2_train, df_loc, left_on='addr_state', right_on='State', how='left').drop(columns='State')

df_test_loc = pd.merge(df2_test, df_loc, left_on='addr_state', right_on='State', how='left').drop(columns='State')
#df_train_tlz = pd.merge(df_train_loc, df_zip, left_on='zip_code', right_on='Zipcode', how='left').drop(columns='Zipcode')

#df_test_tlz = pd.merge(df_test_loc, df_zip, left_on='zip_code', right_on='Zipcode', how='left').drop(columns='Zipcode')
df_train_tlzg = pd.merge(df_train_loc, df_gdp, left_on='City', right_on='State', how='left').drop(columns='City')

df_test_tlzg = pd.merge(df_test_loc, df_gdp, left_on='City', right_on='State', how='left').drop(columns='City')
df_train_tlzg.rename(columns={'year': 'gdp_year'}, inplace=True)

df_test_tlzg.rename(columns={'year': 'gdp_year'}, inplace=True)
df_train_tlzg['i_year_tail'] = df_train_tlzg['i_year'].str[-2:]

df_test_tlzg['i_year_tail'] = df_test_tlzg['i_year'].str[-2:]
df_train_tlzgs = pd.merge(df_train_tlzg, df_spi, left_on=['i_month', 'i_year_tail'] , right_on=['spi_month','spi_year'], how='left').drop(columns='i_year_tail')

df_test_tlzgs = pd.merge(df_test_tlzg, df_spi, left_on=['i_month', 'i_year_tail'] , right_on=['spi_month','spi_year'], how='left').drop(columns='i_year_tail')
#df_train_tlzgs['zip_code'].value_counts(dropna=False)
df_train_tlzgs
df_train_tlzgs.drop('addr_state', axis=1)

df_test_tlzgs.drop('addr_state', axis=1)
X_train = df_train_tlzgs

X_test = df_test_tlzgs
X_train['loan_amnt'] = X_train['loan_amnt'].apply(np.log1p)

X_train['annual_inc'] = X_train['annual_inc'].apply(np.log1p)

X_train['dti'] = X_train['dti'].apply(np.log1p)

X_train['revol_bal'] = X_train['revol_bal'].apply(np.log1p)

X_train['tot_cur_bal'] = X_train['tot_cur_bal'].apply(np.log1p)



X_test['loan_amnt'] = X_test['loan_amnt'].apply(np.log1p)

X_test['annual_inc'] = X_test['annual_inc'].apply(np.log1p)

X_test['dti'] = X_test['dti'].apply(np.log1p)

X_test['revol_bal'] = X_test['revol_bal'].apply(np.log1p)

X_test['tot_cur_bal'] = X_test['tot_cur_bal'].apply(np.log1p)
# dtypeがobjectのカラム名とユニーク数を確認してみましょう。

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())

# application_typeのカラムが１行となっているので、textデータとして扱った方がいいかも
X_train['emp_title'].head(10) # カテゴリよりテキストとして扱ったほうが良いかもしれない
col = 'purpose'



encoder = OneHotEncoder()

enc_train = encoder.fit_transform(X_train[col].values)

enc_test = encoder.transform(X_test[col].values)
enc_train.head()
enc_test.head()
# Onehotの例を参考にやってみましょう

# https://contrib.scikit-learn.org/categorical-encoding/ordinal.html
encoder = OrdinalEncoder()

enc_train = encoder.fit_transform(X_train[col].values)

enc_test = encoder.transform(X_test[col].values)
enc_train.head()
enc_test.head()
# value_couontsで集計した結果を、

summary = X_train[col].value_counts()

summary
# mapする。

enc_train =  X_train[col].map(summary)

enc_test = X_test[col].map(summary)
enc_train.head()
enc_test.head()
'''target = 'loan_condition'

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

    enc_train.iloc[val_ix] = X_val[col].map(summary)'''
# emp_titleを除去

TXT_train = X_train.emp_title.copy()

TXT_test = X_test.emp_title.copy()



cats.remove('emp_title')
oe = OrdinalEncoder(cols=cats, return_df=False)



X_train[cats] = oe.fit_transform(X_train[cats])

X_test[cats] = oe.transform(X_test[cats])
encoder = OrdinalEncoder(cols=cats)
X_train[cats] = encoder.fit_transform(X_train[cats])

X_test[cats] = encoder.fit_transform(X_test[cats])
X_train = X_train.drop('emp_title', axis=1)
X_test = X_test.drop('emp_title', axis=1)
# 欠損を埋めておく 

#X_train.drop(['emp_title'], axis=1, inplace=True)

#X_test.drop(['emp_title'], axis=1, inplace=True)



X_train.fillna(-9999, inplace=True)

X_test.fillna(-9999, inplace=True)



# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html
X_train.head()
import lightgbm as lgb



from sklearn import datasets

from sklearn.model_selection import train_test_split



import numpy as np
len(X_train)
len(y_train)
# CVしてスコアを見てみる。層化抽出で良いかは別途よく考えてみてください。

scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    

    model = lgb.LGBMClassifier()

    model.fit(X_train, y_train)

    

    y_pred = model.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred)

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))
print(np.mean(scores))

print(scores)
# 全データで再学習し、testに対して予測する

model.fit(X_train, y_train)



y_pred = model.predict_proba(X_test)[:,1]
# sample submissionを読み込んで、予測値を代入の後、保存する

# こちらもスムーズな進行のために20分の１に間引いていますが、本番では"skiprows=lambda x: x%20!=0"を削除して用いてください。

submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv', index_col=0)
submission.info()
len(submission.loan_condition)
len(y_pred)
submission.loan_condition = y_pred
submission.to_csv('submission.csv')
TXT_train.fillna('#', inplace=True)

# TXT_train = TXT_train.fillna('#')と同じ

TXT_test.fillna('#', inplace=True)
tfidf = TfidfVectorizer(max_features=1000, use_idf=True)
tfidf
TXT_train = tfidf.fit_transform(TXT_train)

TXT_test = tfidf.fit_transform(TXT_test)
#疎行列が帰ってきます。

TXT_train
TXT_train.shape
# todenseで密行列に変換できますが、ほどんどゼロであることがみて取れます。

TXT_train.todense()
X_train.values
sp.sparse.hstack([X_train.values, TXT_train])
sp.sparse.hstack([X_train.values, TXT_train]).todense()
df_train2 = pd.read_csv('../input/homework-for-students2/train_small.csv', index_col=0, parse_dates=['issue_d'])
df_train2['year'] = df_train2.issue_d.dt.year

df_train2['month'] = df_train2.issue_d.dt.month
df_spi = pd.read_csv('../input/homework-for-students2/spi.csv', parse_dates=['date'] )
df_spi
df_spi['year'] = df_spi.date.dt.year

df_spi['month'] = df_spi.date.dt.month
df_spi.groupby(['year', 'month'])['close'].mean()
df_temp = df_spi.groupby(['year', 'month'], as_index=False)['close'].mean()
df_train2 =df_train2.merge(df_temp, on=['year', 'month'], how='left')
df_train2.head()
df_temp = df_spi.groupby(['year', 'month'], as_index=False)['close'].mean()
df_train.emp_title.isnull().astype(int) #欠損フラグも情報を持っているので残す

# df['col_name'] = df_train.emp_title.isnull().astype(int) #col_nameのところにカラム名を入れる