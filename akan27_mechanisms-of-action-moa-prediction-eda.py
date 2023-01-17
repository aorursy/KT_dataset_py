import numpy as np

import pandas as pd

import plotly.express as px

from IPython.display import display

import random

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

import time



pd.options.display.max_columns = None
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')



train['dataset'] = 'train'

test['dataset'] = 'test'



df = pd.concat([train, test])
train.head()
test.head()
print('Number of rows in training set: ', train.shape[0])

print('Number of columns in training set: ', train.shape[1] - 1)

print('Number of rows in test set: ', test.shape[0])

print('Number of columns in test set: ', test.shape[1] - 1)
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

    width=500,

    height=400

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

    width=500,

    height=400

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

    width=500,

    height=400

)



fig.show()
ds = df[df['dataset']=='train']

ds = ds.groupby(['cp_type', 'cp_time', 'cp_dose'])['sig_id'].count().reset_index()

ds.columns = ['cp_type', 'cp_time', 'cp_dose', 'count']



fig = px.sunburst(

    ds, 

    path=[

        'cp_type',

        'cp_time',

        'cp_dose' 

    ], 

    values='count', 

    title='Sunburst chart for all cp_type/cp_time/cp_dose',

    width=500,

    height=500

)

fig.show()
train_columns = train.columns.to_list()

g_list = [i for i in train_columns if i.startswith('g-')]

c_list = [i for i in train_columns if i.startswith('c-')]

#g_list

#c_list
def plot_set_histograms(plot_list, title):

    fig = make_subplots(rows=4, cols=3)

    traces = [go.Histogram(x=train[col], nbinsx=20, name=col) for col in plot_list]



    for i in range(len(traces)):

        fig.append_trace(traces[i], (i // 3) + 1, (i % 3) + 1)



    fig.update_layout(

        title_text=title,

        height=1000,

        width=1000

    )

    fig.show()
plot_list = [g_list[random.randint(0, len(g_list)-1)] for i in range(50)]

plot_list = list(set(plot_list))[:12]



plot_set_histograms(plot_list, 'Randomly selected gene expression features distributions')
plot_list = [c_list[random.randint(0, len(c_list)-1)] for i in range(50)]

plot_list = list(set(plot_list))[:12]

plot_set_histograms(plot_list, 'Randomly selected cell expression features distributions')
columns = g_list + c_list

for_correlation = list(set([columns[random.randint(0, len(columns)-1)] for i in range(200)]))[:40]

data = df[for_correlation]



f = plt.figure(figsize=(19, 17))

plt.matshow(data.corr(), fignum=f.number)

plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=50)

plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=13)
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

print('Number of columns: ', len(all_columns))
data = df[all_columns]



f = plt.figure(figsize=(19, 15))

plt.matshow(data.corr(), fignum=f.number)

plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=50)

plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
fig = make_subplots(rows=12, cols=3)

traces = [go.Histogram(x=train[col], nbinsx=20, name=col) for col in all_columns]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 3) + 1, (i % 3)+1)



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
plot_scatter(col_df, 4)
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
def plot_3dscatter(col_df, index):

    analysis = pd.DataFrame()

    analysis['color'] = train_target[col_df.iloc[index]['column']]

    analysis['x'] = train[col_df.iloc[index]['train_1_column']]

    analysis['y'] = train[col_df.iloc[index]['train_2_column']]

    analysis['z'] = train[col_df.iloc[index]['train_3_column']]

    analysis.columns = ['color', col_df.iloc[index]['train_1_column'], col_df.iloc[index]['train_2_column'], col_df.iloc[index]['train_3_column']]

    analysis['size'] = 1

    analysis.loc[analysis['color'] == 1, 'size'] = 20



    fig = px.scatter_3d(

        analysis, 

        x=col_df.iloc[index]['train_1_column'], 

        y=col_df.iloc[index]['train_2_column'],

        z=col_df.iloc[index]['train_3_column'], 

        color="color", 

        size='size', 

        height=800,

        width=800,

        title='Scatter plot for ' + col_df.iloc[index]['column']

    )

    fig.show()
plot_3dscatter(col_df, 0)
plot_3dscatter(col_df, 1)
plot_3dscatter(col_df, 2)
plot_3dscatter(col_df, 3)
plot_3dscatter(col_df, 4)
last_term = dict()

for item in target_columns:

    try:

        last_term[item.split('_')[-1]] += 1

    except:

        last_term[item.split('_')[-1]] = 1



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
answer = list()

for group in last_term.group.tolist():

    agent_list = list()

    for item in target_columns:

        if item.split('_')[-1] == group:

            agent_list.append(item)

    agent_df = train_target[agent_list]

    data = agent_df.astype(bool).sum(axis=1).reset_index()

    answer.append(data[0].max())
ds = pd.DataFrame()

ds['group'] = last_term.group.tolist()

ds['max_value'] = answer



fig = px.bar(

    ds, 

    x='max_value', 

    y="group", 

    orientation='h', 

    title='Maximum number of active columns in 1 sample for every group', 

    width=800,

    height=500

)

fig.show()
categories = train[['cp_type', 'cp_time', 'cp_dose']]

tar = train_target.copy()

tar = tar.drop(['sig_id'], axis=1)

analysis = pd.concat([categories, tar], axis=1)
for category in analysis['cp_dose'].unique().tolist():

    number = 0

    cols = []

    for col in analysis.columns:

        if col in ['cp_type', 'cp_time', 'cp_dose']:

            continue

        if len(analysis[analysis['cp_dose'] == category][col].value_counts()) == 1:

            number += 1

            cols.append(col)



    print(category, '. Number of columns with 1 unique value: ', number, '. Columns: ', cols)
analysis[analysis['cp_dose'] == 'D2']['atp-sensitive_potassium_channel_antagonist'].value_counts()
analysis[analysis['cp_dose']=='D2']['erbb2_inhibitor'].value_counts()
for category in analysis['cp_time'].unique().tolist():

    number = 0

    cols = []

    for col in analysis.columns:

        if col in ['cp_type', 'cp_time', 'cp_dose']:

            continue

        if len(analysis[analysis['cp_time']==category][col].value_counts()) == 1:

            number += 1

            cols.append(col)



    print(category, '. Number of columns with 1 unique value: ', number, '. Columns: ', cols)
analysis[analysis['cp_time'] == 24]['erbb2_inhibitor'].value_counts()
analysis[analysis['cp_time'] == 72]['erbb2_inhibitor'].value_counts()
analysis[analysis['cp_time'] == 24]['atp-sensitive_potassium_channel_antagonist'].value_counts()
analysis[analysis['cp_time'] == 72]['atp-sensitive_potassium_channel_antagonist'].value_counts()
for category in analysis['cp_type'].unique().tolist():

    number = 0

    cols = []

    for col in analysis.columns:

        if col in ['cp_type', 'cp_time', 'cp_dose']:

            continue

        if len(analysis[analysis['cp_type']==category][col].value_counts()) == 1:

            number += 1

            cols.append(col)



    print(category, '. Number of columns with 1 unique value: ', number, '. Columns: ', cols)
analysis[analysis['cp_type']=='ctl_vehicle']['igf-1_inhibitor'].value_counts()