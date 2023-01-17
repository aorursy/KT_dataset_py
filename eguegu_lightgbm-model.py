# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# ライブラリ

import numpy as np

import pandas as pd

import os

import json

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

%matplotlib inline
def read_data():

    print(f'Read data')

    train_df = pd.read_csv('../input/data-science-bowl-2019/train.csv')

    test_df = pd.read_csv('../input/data-science-bowl-2019/test.csv')

    train_labels_df = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

    specs_df = pd.read_csv('../input/data-science-bowl-2019/specs.csv')

    sample_submission_df = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')

    print(f"train shape: {train_df.shape}")

    print(f"test shape: {test_df.shape}")

    print(f"train labels shape: {train_labels_df.shape}")

    print(f"specs shape: {specs_df.shape}")

    print(f"sample submission shape: {sample_submission_df.shape}")

    return train_df, test_df, train_labels_df, specs_df, sample_submission_df
train_df, test_df, train_labels_df, specs_df, sample_submission_df = read_data()
train_df.head()
test_df.head()
train_labels_df.head()
specs_df.head()
sample_submission_df.head()
print(f"train installation id: {train_df.installation_id.nunique()}")

print(f"test installation id: {test_df.installation_id.nunique()}")

print(f"train label installation id: {train_labels_df.installation_id.nunique()}")
# train dataのそれぞれのカラムのユニークな値の合計

for column in train_df.columns.values:

    print(f"[train] Unique values of `{column}` : {train_df[column].nunique()}")
# test dataのそれぞれのカラムのユニークな値の合計

for column in test_df.columns.values:

    print(f"[test] Unique values of `{column}`: {test_df[column].nunique()}")
# train dataと test dataのそれぞれの特徴量の散らばり(上位20)をグラフで表示する関数

def plot_count(feature, title, df, size=1):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()  
# titleのtrain dataの特徴量の種類の散らばり

plot_count('title', 'title (first most frequent 20 values - train)', train_df, size=4)
# titleのtest dataの特徴量の種類の散らばり

plot_count('title', 'title (first most frequent 20 values - test)', test_df, size=4)
# typeのtrain dataの特徴量の散らばり

plot_count('type', 'type - train', train_df, size=2)
# typeのtest dataの特徴量の散らばり

plot_count('type', 'type - test', test_df, size=2)
# train label dataのそれぞれのカラムのユニークな値の合計

for column in train_labels_df.columns.values:

    print(f"[train_labels] Unique values of {column} : {train_labels_df[column].nunique()}")
# titleのtrain label dataの特徴量の種類の散らばり

plot_count('title', 'title - train_labels', train_labels_df, size=3)
train_df['title'].unique()
plot_count('accuracy_group', 'accuracy_group - train_labels', train_labels_df, size=2)
# spec data のそれぞれのカラムのユニークな値の合計

for column in specs_df.columns.values:

    print(f"[specs] Unique values of `{column}`: {specs_df[column].nunique()}")
# Todo 

# event_dataの中身を分析する　→ https://www.kaggle.com/gpreda/2019-data-science-bowl-eda

train_df['event_data'].iloc[3]
# Todo 

# spec dataのargsの中身を分析する
# train data とtrain label data のtimestampに関するデータを生成して、加算したdfを作る

def extract_time_features(df):

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['date'] = df['timestamp'].dt.date

    df['month'] = df['timestamp'].dt.month

    df['hour'] = df['timestamp'].dt.hour

    df['year'] = df['timestamp'].dt.year

    df['dayofweek'] = df['timestamp'].dt.dayofweek

    df['weekofyear'] = df['timestamp'].dt.weekofyear

    df['dayofyear'] = df['timestamp'].dt.dayofyear

    df['quarter'] = df['timestamp'].dt.quarter

    df['is_month_start'] = df['timestamp'].dt.is_month_start

    print(f"shape: {df.shape}")

    return df
train_df = extract_time_features(train_df)

test_df = extract_time_features(test_df)
train_df.head()
test_df.head()
# Todo

# timestampを使った分析をする
# 数値データと分類データに分けるためのカラム

numerical_columns = ['game_time', 'month', 'dayofweek', 'hour']

categorical_columns = ['type', 'world']



# installation_idをkeyとしたdfを作る

comp_train_df = pd.DataFrame({'installation_id': train_df['installation_id'].unique()})

comp_train_df.set_index('installation_id', inplace = True)
def get_numeric_columns(df, column):

    df = df.groupby('installation_id').agg({f'{column}': ['mean', 'sum', 'min', 'max', 'std', 'skew']})

    df[column].fillna(df[column].mean(), inplace = True)

    df.columns = [f'{column}_mean', f'{column}_sum', f'{column}_min', f'{column}_max', f'{column}_std', f'{column}_skew']

    return df
for i in numerical_columns:

    comp_train_df = comp_train_df.merge(get_numeric_columns(train_df, i), left_index = True, right_index = True)
print(f"comp_train shape: {comp_train_df.shape}")
pd.get_option("display.max_columns")

pd.set_option('display.max_columns', 50)

comp_train_df.head()
# get the mode of the title

labels_map = dict(train_labels_df.groupby('title')['accuracy_group'].agg(lambda x:x.value_counts().index[0]))

# merge target

labels = train_labels_df[['installation_id', 'title', 'accuracy_group']]

# replace title with the mode

labels['title'] = labels['title'].map(labels_map)

# join train with labels

comp_train_df = labels.merge(comp_train_df, on = 'installation_id', how = 'left')

print('We have {} training rows'.format(comp_train_df.shape[0]))
print(f"comp_train shape: {comp_train_df.shape}")
comp_train_df.head()
print(f"comp_train_df shape: {comp_train_df.shape}")

for feature in comp_train_df.columns.values[3:20]:

    print(f"{feature} unique values: {comp_train_df[feature].nunique()}")
plt.figure(figsize=(16,6))

_accuracy_groups = comp_train_df.accuracy_group.unique()

plt.title("Distribution of log(`game time mean`) values (grouped by accuracy group) in the comp train")

for _accuracy_group in _accuracy_groups:

    red_comp_train_df = comp_train_df.loc[comp_train_df.accuracy_group == _accuracy_group]

    sns.distplot(np.log(red_comp_train_df['game_time_mean']), kde=True, label=f'accuracy group= {_accuracy_group}')

plt.legend()

plt.show()
# com_test_data を生成する

# installation_idをkeyとしたdfを作る

comp_test_df = pd.DataFrame({'installation_id': test_df['installation_id'].unique()})

comp_test_df.set_index('installation_id', inplace = True)



for i in numerical_columns:

    comp_test_df = comp_test_df.merge(get_numeric_columns(test_df, i), left_index = True, right_index = True)
print(f"comp_test shape: {comp_test_df.shape}")

comp_test_df.head()
a = test_df

b = a.drop_duplicates(subset='installation_id')

b
test_title = b[['installation_id','title']]







# join train with labels

comp_test_df = test_title.merge(comp_test_df, on = 'installation_id', how = 'left')

print('We have {} testing rows'.format(comp_test_df.shape[0]))
print(f"comp_test shape: {comp_test_df.shape}")

comp_test_df.head()
train = comp_train_df.drop(['installation_id', 'title'], axis=1)

train_x = train.drop(['accuracy_group'], axis=1)

train_y = train['accuracy_group']

test_x = comp_test_df.drop(['installation_id', 'title'], axis=1)
train_x.head()
test_x.head()
# train_xは学習データ、train_yは目的変数、test_xはテストデータ

# pandasのDataFrame, Seriesで保持します。（numpyのarrayで保持することもあります）



# installation_idを削除する



train = comp_train_df.drop(['installation_id', 'title'], axis=1)

train_x = train.drop(['accuracy_group'], axis=1)

train_y = train['accuracy_group']

test_x = comp_test_df.drop(['installation_id', 'title'], axis=1)





# 学習データを学習データとバリデーションデータに分ける

from sklearn.model_selection import KFold



kf = KFold(n_splits=4, shuffle=True, random_state=71)

tr_idx, va_idx = list(kf.split(train_x))[0]

tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]

tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]



# -----------------------------------

# lightgbmの実装

# -----------------------------------

import lightgbm as lgb

from sklearn.metrics import log_loss



# 特徴量と目的変数をlightgbmのデータ構造に変換する

lgb_train = lgb.Dataset(tr_x, tr_y)

lgb_eval = lgb.Dataset(va_x, va_y)



# ハイパーパラメータの設定

params = {'objective': 'regression', 'seed': 71, 'verbose': 0}

num_round = 100



# 学習の実行

# カテゴリ変数をパラメータで指定している

# バリデーションデータもモデルに渡し、学習の進行とともにスコアがどう変わるかモニタリングする

#categorical_features = ['accuracy_group', 'medical_info_b2', 'medical_info_b3']

model = lgb.train(params, lgb_train, num_boost_round=num_round,

                  # categorical_feature=categorical_features,

                  valid_names=['train', 'valid'], 

                  valid_sets=[lgb_train, lgb_eval])



# バリデーションデータでのスコアの確認

#va_pred = model.predict(va_x)

#score = log_loss(va_y, va_pred)

#print(f'logloss: {score:.4f}')



# 予測

pr1 = model.predict(test_x)





pr1[pr1 <= 1.56] = 0

pr1[np.where(np.logical_and(pr1 > 1.56, pr1 <= 1.77))] = 1

pr1[np.where(np.logical_and(pr1 > 1.77, pr1 <= 2.025))] = 2

pr1[pr1 > 2.025] = 3





print(pr1)
sample_submission_df['accuracy_group'] = pr1.astype(int)

sample_submission_df.to_csv('submission.csv', index=False)
sample_submission_df['accuracy_group'].value_counts(normalize=True)