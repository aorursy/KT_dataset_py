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
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train['dataset'] = 'train'

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

test['dataset'] = 'test'

df = pd.concat([train, test])
train.head()

test.head()
print("train",train.shape)

print("test",test.shape)
train.isnull().sum()
test.isnull().sum()
df.info()
import plotly.express as px

import plotly.offline as pf

from plotly.subplots import make_subplots
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
ratio1 = train.groupby('cp_type').count()

ratio2 = test.groupby('cp_type').count()

#groupby(['cp_type', 'dataset'])['sig_id'].count().reset_index()
ratio1
ratio2
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
g_list
import plotly.graph_objs as go

import random



plot_list = [g_list[random.randint(0, len(g_list)-1)] for i in range(12)]

#[g_list[random.randint(0, len(g_list)-1)] for i in range(12)]





fig = make_subplots(rows=4, cols=3)



#trace0 = px.histogram(train, x=plot_list[0], color='cp_type',opacity=0.4, marginal='box')

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
fig = px.histogram(df, x='g-0', color='cp_type',opacity=0.4, marginal='box')

                           #nbins=19, range_x=[4,8], width=600, height=350,

                           #opacity=0.4, marginal='box')

fig.update_layout(barmode='overlay')

fig.update_yaxes(range=[0,20],row=1, col=1)
import matplotlib.pyplot as plt



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
target_columns = train_target.columns.tolist()

target_columns.remove('sig_id')

for_analysis = [target_columns[random.randint(0, len(target_columns)-1)] for i in range(5)]

current_corr = correlation_matrix[for_analysis]
target_columns
col_df = pd.DataFrame()

tr_first_cols = list()

tr_second_cols = list()

tar_cols = list()

for col in current_corr.columns:

    tar_cols.append(col)

    tr_first_cols.append(current_corr[col].abs().sort_values(ascending=False).reset_index()['train_features'].head(2).values[0])

    tr_second_cols.append(current_corr[col].abs().sort_values(ascending=False).reset_index()['train_features'].head(2).values[1])



col_df['column'] = tar_cols

col_df['train_1_column'] = tr_first_cols

col_df['train_2_column'] = tr_second_cols

col_df
def plot_scatter(col_df, index):

    analysis = pd.DataFrame()

    analysis['color'] = train_target[col_df.iloc[index]['column']]

    analysis['x'] = train[col_df.iloc[index]['train_1_column']]

    analysis['y'] = train[col_df.iloc[index]['train_2_column']]

    analysis.columns = ['color', col_df.iloc[index]['train_1_column'], col_df.iloc[index]['train_2_column']]

    analysis['size'] = 1

    analysis.loc[analysis['color'] == 1, 'size'] = 10



    fig = px.scatter(

        analysis, 

        x=col_df.iloc[index]['train_1_column'], 

        y=col_df.iloc[index]['train_2_column'], 

        color="color", 

        size='size', 

        height=800,

        title='Scatter plot for ' + col_df.iloc[index]['column']

    )

    fig.show()
plot_scatter(col_df, 0)
plot_scatter(col_df, 1)
plot_scatter(col_df, 2)
plot_scatter(col_df, 3)
for_analysis = [target_columns[random.randint(0, len(target_columns)-1)] for i in range(5)]

current_corr = correlation_matrix[for_analysis]



col_df = pd.DataFrame()

tr_first_cols = list()

tr_second_cols = list()

tr_third_cols = list()

tar_cols = list()

for col in current_corr.columns:

    tar_cols.append(col)

    tr_first_cols.append(current_corr[col].abs().sort_values(ascending=False).reset_index()['train_features'].head(3).values[0])

    tr_second_cols.append(current_corr[col].abs().sort_values(ascending=False).reset_index()['train_features'].head(3).values[1])

    tr_third_cols.append(current_corr[col].abs().sort_values(ascending=False).reset_index()['train_features'].head(3).values[2])





col_df['column'] = tar_cols

col_df['train_1_column'] = tr_first_cols

col_df['train_2_column'] = tr_second_cols

col_df['train_3_column'] = tr_third_cols

col_df
#ターゲット名から最後の特徴文字を抽出し、カウントする。

last_term = dict()

for item in target_columns:

    try:

        last_term[item.split('_')[-1]] += 1

    except:

        last_term[item.split('_')[-1]] = 1



#1よりも多くあったところを

last_term = pd.DataFrame(last_term.items(), columns=['group', 'count'])

last_term = last_term.sort_values('count')

last_term = last_term[last_term['count']>1]

last_term['count'] = last_term['count'] * 100 / 206





fig = px.bar(

    last_term, 

    x='count', 

    y="group", 

    orientation='h', 

    title='Groups in target columns (Percent from all target columns)', 

    width=800,

    height=500

)

fig.show()