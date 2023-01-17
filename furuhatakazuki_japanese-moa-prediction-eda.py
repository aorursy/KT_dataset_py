# この Python 3 環境には、多くの便利な分析ライブラリがインストールされています。

# kaggle/pythonのDockerイメージで定義されています。: https://github.com/kaggle/docker-python

# 例えば、ロードするのに便利なパッケージは以下の通りです。



import numpy as np # 線形代数

import pandas as pd # データ処理, CSVファイル I/O (例. pd.read_csv)



# 入力データファイルは、読み取り専用です "../input/" directory

# 例えば、これを実行すると (run をクリックするか Shift+Enter キーを押して)、入力ディレクトリの下にあるすべてのファイルが一覧表示されます。



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# カレントディレクトリ(/kaggle/working/)に最大5GBまで書き込むことができ、"Save & Run All "を使ってバージョンを作成したときに出力として保存されます。 

# /kaggle/temp/ に一時ファイルを書き込むこともできますが、現在のセッションの外には保存されません。



import plotly.express as px

from IPython.display import display

pd.options.display.max_columns = None

import random

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train['dataset'] = 'train'

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

test['dataset'] = 'test'

df = pd.concat([train, test])
train.head()
test.head()
print('Number of rows in training set: ', train.shape[0])

print('Number of columns in training set: ', train.shape[1]-1)

print('Number of rows in test set: ', test.shape[0])

print('Number of columns in test set: ', test.shape[1]-1)
df.info()
ds = df.groupby(['cp_type', 'dataset'])['sig_id'].count().reset_index()

ds.columns = ['cp_type', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='cp_type', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='cp_type train/test counts', 

    width=600,

    height=500

)

fig.show()
ds = df.groupby(['cp_time', 'dataset'])['sig_id'].count().reset_index()

ds.columns = ['cp_time', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='cp_time', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='cp_time train/test counts', 

    width=600,

    height=500

)

fig.show()
ds = df.groupby(['cp_dose', 'dataset'])['sig_id'].count().reset_index()

ds.columns = ['cp_dose', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='cp_dose', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='cp_dose train/test counts', 

    width=600,

    height=500

)

fig.show()
train_columns = train.columns.to_list()

g_list = [i for i in train_columns if i.startswith('g-')]

c_list = [i for i in train_columns if i.startswith('c-')]
plot_list = [g_list[random.randint(0, len(g_list)-1)] for i in range(12)]



fig = make_subplots(rows=4, cols=3)



trace0 = go.Histogram(x=train[plot_list[0]], nbinsx=20, name=plot_list[0])

trace1 = go.Histogram(x=train[plot_list[1]], nbinsx=20, name=plot_list[1])

trace2 = go.Histogram(x=train[plot_list[2]], nbinsx=20, name=plot_list[2])

trace3 = go.Histogram(x=train[plot_list[3]], nbinsx=20, name=plot_list[3])

trace4 = go.Histogram(x=train[plot_list[4]], nbinsx=20, name=plot_list[4])

trace5 = go.Histogram(x=train[plot_list[5]], nbinsx=20, name=plot_list[5])

trace6 = go.Histogram(x=train[plot_list[6]], nbinsx=20, name=plot_list[6])

trace7 = go.Histogram(x=train[plot_list[7]], nbinsx=20, name=plot_list[7])

trace8 = go.Histogram(x=train[plot_list[8]], nbinsx=20, name=plot_list[8])

trace9 = go.Histogram(x=train[plot_list[9]], nbinsx=20, name=plot_list[9])

trace10 = go.Histogram(x=train[plot_list[10]], nbinsx=20, name=plot_list[10])

trace11 = go.Histogram(x=train[plot_list[11]], nbinsx=20, name=plot_list[11])



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 1, 3)

fig.append_trace(trace3, 2, 1)

fig.append_trace(trace4, 2, 2)

fig.append_trace(trace5, 2, 3)

fig.append_trace(trace6, 3, 1)

fig.append_trace(trace7, 3, 2)

fig.append_trace(trace8, 3, 3)

fig.append_trace(trace9, 4, 1)

fig.append_trace(trace10, 4, 2)

fig.append_trace(trace11, 4, 3)



fig.update_layout(

    title_text='Randomly selected gene expression features distributions'

)

fig.show()
plot_list = [c_list[random.randint(0, len(c_list)-1)] for i in range(12)]



fig = make_subplots(rows=4, cols=3)



trace0 = go.Histogram(x=train[plot_list[0]], nbinsx=20, name=plot_list[0])

trace1 = go.Histogram(x=train[plot_list[1]], nbinsx=20, name=plot_list[1])

trace2 = go.Histogram(x=train[plot_list[2]], nbinsx=20, name=plot_list[2])

trace3 = go.Histogram(x=train[plot_list[3]], nbinsx=20, name=plot_list[3])

trace4 = go.Histogram(x=train[plot_list[4]], nbinsx=20, name=plot_list[4])

trace5 = go.Histogram(x=train[plot_list[5]], nbinsx=20, name=plot_list[5])

trace6 = go.Histogram(x=train[plot_list[6]], nbinsx=20, name=plot_list[6])

trace7 = go.Histogram(x=train[plot_list[7]], nbinsx=20, name=plot_list[7])

trace8 = go.Histogram(x=train[plot_list[8]], nbinsx=20, name=plot_list[8])

trace9 = go.Histogram(x=train[plot_list[9]], nbinsx=20, name=plot_list[9])

trace10 = go.Histogram(x=train[plot_list[10]], nbinsx=20, name=plot_list[10])

trace11 = go.Histogram(x=train[plot_list[11]], nbinsx=20, name=plot_list[11])



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)

fig.append_trace(trace2, 1, 3)

fig.append_trace(trace3, 2, 1)

fig.append_trace(trace4, 2, 2)

fig.append_trace(trace5, 2, 3)

fig.append_trace(trace6, 3, 1)

fig.append_trace(trace7, 3, 2)

fig.append_trace(trace8, 3, 3)

fig.append_trace(trace9, 4, 1)

fig.append_trace(trace10, 4, 2)

fig.append_trace(trace11, 4, 3)



fig.update_layout(

    title_text='Randomly selected cell viability features distributions'

)

fig.show()
columns = g_list + c_list

for_correlation = [columns[random.randint(0, len(columns)-1)] for i in range(40)]

data = df[for_correlation]



f = plt.figure(figsize=(19, 15))

plt.matshow(data.corr(), fignum=f.number)

plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=45)

plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
import time



start = time.time()

cols = ['cp_time'] + columns

all_columns = []

for i in range(0, len(cols)):

    for j in range(i+1, len(cols)):

        if abs(train[cols[i]].corr(train[cols[j]])) > 0.9:

            all_columns.append(cols[i])

            all_columns.append(cols[j])



print(time.time()-start)
all_columns = list(set(all_columns))
len(all_columns)
data = df[all_columns]



f = plt.figure(figsize=(19, 15))

plt.matshow(data.corr(), fignum=f.number)

plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=45)

plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
fig = make_subplots(rows=12, cols=3)



traces = [go.Histogram(x=train[col], nbinsx=20, name=col) for col in all_columns]



i=1

j=1



for trace in traces:

    fig.append_trace(trace, i, j)

    if j==3:

        j=1

        i+=1

    else:

        j+=1



fig.update_layout(

    title_text='Highly correlated features',

    height=1200

)

fig.show()
train_target = pd.read_csv("../input/lish-moa/train_targets_scored.csv")



print('Number of rows : ', train_target.shape[0])

print('Number of cols : ', train_target.shape[1])

train_target.head()
x = train_target.drop(['sig_id'], axis=1).sum(axis=0).sort_values().reset_index()

x.columns = ['column', 'nonzero_records']



fig = px.bar(

    x.tail(50), 

    x='nonzero_records', 

    y='column', 

    orientation='h', 

    title='Columns with the higher number of positive samples (top 50)', 

    height=1000, 

    width=800

)

fig.show()
x = train_target.drop(['sig_id'], axis=1).sum(axis=0).sort_values(ascending=False).reset_index()

x.columns = ['column', 'nonzero_records']



fig = px.bar(

    x.tail(50), 

    x='nonzero_records', 

    y='column', 

    orientation='h', 

    title='Columns with the lowest number of positive samples (top 50)', 

    height=1000, 

    width=800

)

fig.show()
x = train_target.drop(['sig_id'], axis=1).sum(axis=0).sort_values(ascending=False).reset_index()

x.columns = ['column', 'count']

x['count'] = x['count'] * 100 / len(train_target)

fig = px.bar(

    x, 

    x='column', 

    y='count', 

    orientation='v', 

    title='Percent of positive records for every column in target', 

    height=800, 

    width=1200

)

fig.show()
data = train_target.drop(['sig_id'], axis=1).astype(bool).sum(axis=1).reset_index()

data.columns = ['row', 'count']

data = data.groupby(['count'])['row'].count().reset_index()

fig = px.bar(

    data, 

    y=data['row'], 

    x="count", 

    title='Number of activations in targets for every sample', 

    width=800, 

    height=500

)

fig.show()
data = train_target.drop(['sig_id'], axis=1).astype(bool).sum(axis=1).reset_index()

data.columns = ['row', 'count']

data = data.groupby(['count'])['row'].count().reset_index()

fig = px.pie(

    data, 

    values=100 * data['row']/len(train_target), 

    names="count", 

    title='Number of activations in targets for every sample (Percent)', 

    width=800, 

    height=500

)

fig.show()
train_target.describe()
start = time.time()



correlation_matrix = pd.DataFrame()

for t_col in train_target.columns:

    corr_list = list()

    if t_col == 'sig_id':

        continue

    for col in columns:

        res = train[col].corr(train_target[t_col])

        corr_list.append(res)

    correlation_matrix[t_col] = corr_list

    

print(time.time()-start)
correlation_matrix['train_features'] = columns

correlation_matrix = correlation_matrix.set_index('train_features')

correlation_matrix
maxCol=lambda x: max(x.min(), x.max(), key=abs)

high_scores = correlation_matrix.apply(maxCol, axis=0).reset_index()

high_scores.columns = ['column', 'best_correlation']



fig = px.bar(

    high_scores, 

    x='column', 

    y="best_correlation", 

    orientation='v', 

    title='Best correlation with train columns for every target column', 

    width=1200,

    height=800

)

fig.show()
col_df = pd.DataFrame()

tr_cols = list()

tar_cols = list()

for col in correlation_matrix.columns:

    tar_cols.append(col)

    tr_cols.append(correlation_matrix[col].abs().sort_values(ascending=False).reset_index()['train_features'].head(1).values[0])



col_df['column'] = tar_cols

col_df['train_best_column'] = tr_cols



total_scores = pd.merge(high_scores, col_df)

total_scores
count_features = total_scores['train_best_column'].value_counts().reset_index().sort_values('train_best_column')

count_features.columns = ['column', 'count']

fig = px.bar(

    count_features.tail(33), 

    x='count', 

    y="column", 

    orientation='h', 

    title='Columns from training set with number of high correlations with target columns', 

    width=800,

    height=700

)

fig.show()