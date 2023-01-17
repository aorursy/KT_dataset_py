import math

import os

import csv

import json

import gc

from functools import partial

from pathlib import Path

from multiprocessing import Pool



import pandas as pd

import numpy as np

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

import matplotlib.patheffects as pe

from tqdm import tqdm_notebook

from pandas_summary import DataFrameSummary

import seaborn as sns

from pandas.io.json import json_normalize

from hashlib import md5



tqdm.pandas()
def compare_cat_frequency(df1, df2, column, df1_name='train', df2_name='test', top_n=25, figsize=(20, 16)):

    fig, (ax, ax2) = plt.subplots(ncols=2, sharey=True, figsize=figsize)

    

    ax.yaxis.tick_right()



    df1_values = df1[column].value_counts()[:25]

    df1_values.plot.barh(ax=ax, title=f'{df1_name} {column} distribution')



    test_df[column].value_counts()[df1_values.keys()].plot.barh(ax=ax2, title=f'{df2_name} {column} distribution')



    plt.tight_layout()

    plt.show()





def plot_grid_hist(df, columns, test_df=None, ncols=3, figsize=(20, 12)):

    sns.set_palette("Spectral_r")



    fig, axes = plt.subplots(nrows=len(columns) // ncols, ncols=ncols, figsize=figsize)



    count = 0

    for ax_row in axes:

        for ax in ax_row:

            count += 1

            try:

                key = columns[count]

                print(key)

                ax.hist(df[key], label='Train', edgecolor='black', linewidth=0, bins=100, histtype='stepfilled', density=True)

                if test_df is not None:

                    ax.hist(test_df[key], label='Test', bins=100, linewidth=1, linestyle='dashed', alpha = 0.5, histtype='stepfilled', density=True)

                    ax.legend()

                ax.set_title(f'Distribution of {key}')

            except IndexError:

                continue
DATA_PATH = Path('/kaggle/input/data-science-bowl-2019/')

OUTPUT_PATH = Path('/kaggle/working/')
TRAIN_DTYPES = {

    'event_count': np.uint16,

    'event_code': np.uint16,

    'game_type': np.uint32

}
%%time



train_df = pd.read_csv(

    DATA_PATH/'train.csv', parse_dates=['timestamp'],

    dtype=TRAIN_DTYPES

).sort_values('timestamp')
%%time



test_df = pd.read_csv(

    DATA_PATH/'test.csv',

    parse_dates=['timestamp'],

    dtype=TRAIN_DTYPES

).sort_values('timestamp')
DataFrameSummary(train_df).columns_stats
DataFrameSummary(test_df).columns_stats
sns.set_palette("Spectral_r")



plt.figure(figsize=(14, 4))

plt.title("timestamp frequency")



plt.hist(train_df.timestamp, edgecolor='black', linewidth=1.2, label='Train', histtype='stepfilled', density=True)

plt.hist(test_df.timestamp, edgecolor='black', linewidth=1.2, linestyle='dashed', label='Test', alpha = 0.5, histtype='stepfilled', density=True)



plt.xticks(rotation=70)

plt.legend()

plt.show()
import matplotlib.patheffects as pe



sns.set_palette("RdBu_r")



plt.figure(figsize=(14, 4))

plt.title("event_time frequency")

plt.hist(train_df.game_time, label='Train', bins=100, path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])

plt.hist(test_df.game_time, label='Test', bins=100, path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])



plt.legend()

plt.show()
compare_cat_frequency(train_df, test_df, column='title')
for column in ['event_code', 'event_id', 'world', 'type']:

    compare_cat_frequency(train_df, test_df, column=column)
all_df = pd.concat([train_df, test_df], axis=0)



WORLD_VALS = all_df.world.unique()

TITLE_VALS = all_df.title.unique()

TYPE_VALS = all_df.type.unique()

EVENT_CODE = all_df.event_code.unique()

EVENT_ID = all_df.event_id.unique()
def set_categorical(df):

    df.world = pd.Categorical(df.world, categories=WORLD_VALS)

    df.title = pd.Categorical(df.title, categories=TITLE_VALS)

    df.type = pd.Categorical(df.type, categories=TYPE_VALS)

    df.event_code = pd.Categorical(df.event_code, categories=EVENT_CODE)

    df.event_id = pd.Categorical(df.event_id, categories=EVENT_ID)

    return df
train_df = set_categorical(train_df)

test_df = set_categorical(test_df)
train_df.dtypes
del all_df

gc.collect()
def flatten_json(nested_json):

    nested_json = json.loads(nested_json)



    out = {}



    def _flatten(x, name=''):

        if type(x) is dict:

            for a in x: _flatten(x[a], name + a + '_')

        elif type(x) is list:

            i = 0

            for a in x:

                _flatten(a, name + str(i) + '_')

                i += 1

        else:

            out[name[:-1]] = x



    _flatten(nested_json)



    return out
train_df_sample = train_df.sample(n=100_000)
%%time



train_event_data_norm_sample = json_normalize(train_df_sample.event_data.progress_apply(flatten_json))
train_event_na_perc = (

    train_event_data_norm_sample.isna().sum().sort_values() /

     len(train_event_data_norm_sample))
train_event_na_perc
columns_to_include = train_event_na_perc[train_event_na_perc <= 0.95].keys()

columns_to_include = [c for c in columns_to_include if c not in ('event_count', 'event_code', 'game_time')]
DataFrameSummary(train_event_data_norm_sample[columns_to_include]).columns_stats
ed_summary = DataFrameSummary(train_event_data_norm_sample[columns_to_include]).summary(); ed_summary
numeric_cols = ed_summary.T[ed_summary.T.types == 'numeric'].index

path_effects = [pe.Stroke(linewidth=1, foreground='black'), pe.Normal()]
plot_grid_hist(train_event_data_norm_sample, columns=list(numeric_cols)[:16])
columns_to_include
DESCRIPTIONS = []

SOURCE_CATS = []

IDENTIFIER_CATS = set([])

MEDIA_TYPE_CATS = set([])

COORD_STAGE_HEIGHT = set([])

COORD_STAGE_WIDTH = set([])





def do_event_data(event_data: dict, output_file: str):

    csv_file = open(f'{OUTPUT_PATH}/{output_file}', 'w')

    csv_writer = csv.writer(csv_file, delimiter=',')



    for data in tqdm(event_data.values, total=len(event_data)):

        row_flattened = flatten_json(data)



        # map description to its hash.

        desc = row_flattened.get('description')

        if desc:

            if desc not in DESCRIPTIONS:

                DESCRIPTIONS.append(desc)

            row_flattened['description'] = DESCRIPTIONS.index(desc)

            

        source = row_flattened.get('source')

        if source:

            source = str(source).lower()

            if source not in SOURCE_CATS:

                SOURCE_CATS.append(source)

            row_flattened['source'] = SOURCE_CATS.index(source)

            

        for col, l in [

            ('identifier', IDENTIFIER_CATS),

            ('media_type', MEDIA_TYPE_CATS),

            ('coordinates_stage_height', COORD_STAGE_HEIGHT),

            ('coordinates_stage_width', COORD_STAGE_WIDTH)

        ]:

            value = row_flattened.get(col)

            if value: l.add(value)



        csv_writer.writerow(row_flattened.get(k, None) for k in columns_to_include)
do_event_data(train_df.event_data, 'train_event_data.csv')
do_event_data(test_df.event_data, 'test_event_data.csv')
dtypes = dict(

    source=pd.CategoricalDtype(list(range(len(SOURCE_CATS)))),

    media_type=pd.CategoricalDtype(MEDIA_TYPE_CATS),

    identifier=pd.CategoricalDtype(IDENTIFIER_CATS),

    description=pd.CategoricalDtype(list(range(len(DESCRIPTIONS)))),

    coordinates_stage_height=pd.CategoricalDtype(list(range(len(COORD_STAGE_HEIGHT)))),

    coordinates_stage_width=pd.CategoricalDtype(list(range(len(COORD_STAGE_WIDTH))))

)
%%time



train_event_data = pd.read_csv(

    OUTPUT_PATH/'train_event_data.csv', names=columns_to_include, header=None, dtype=dtypes)
%%time



test_event_data = pd.read_csv(OUTPUT_PATH/'test_event_data.csv', names=columns_to_include, header=None, dtype=dtypes)
numeric_cols_revised = ['round', 'coordinates_x', 'coordinates_y', 'duration', 'total_duration', 'level', 'size', 'weight']
plot_grid_hist(

    train_event_data.sample(n=500_000),

    test_df=test_event_data,

    columns=list(numeric_cols_revised),

    ncols=2, figsize=(16, 20))
def join_event_data(df, df_event):

    return pd.concat([

        df[[i for i in df.columns if i != 'event_data']].reset_index(drop=True),

        df_event.reset_index(drop=True)], axis=1).reset_index(drop=True)
train_df_comb = join_event_data(train_df, train_event_data)

test_df_comb = join_event_data(test_df, test_event_data)
train_df_comb = train_df_comb.sort_values(['installation_id', 'timestamp']).reset_index(drop=True)

test_df_comb = test_df_comb.sort_values(['installation_id', 'timestamp']).reset_index(drop=True)
train_df_comb.to_feather(OUTPUT_PATH/'train.fth')

test_df_comb.to_feather(OUTPUT_PATH/'test.fth')
del train_df_comb

del test_df_comb

del train_df

del test_df

del train_event_data

del test_event_data

del train_event_data_norm_sample

del train_df_sample



gc.collect()
train_df_comb = pd.read_feather(OUTPUT_PATH/'train.fth')

test_df_comb = pd.read_feather(OUTPUT_PATH/'test.fth')
train_labels = pd.read_csv(DATA_PATH/'train_labels.csv')
# thanks to https://www.kaggle.com/artgor/oop-approach-to-fe-and-models



def set_attempt_label(df):

    df['attempt'] = 0

    df.loc[

        (df['title'] == 'Bird Measurer (Assessment)') &

        (df['event_code'] == 4110), 'attempt'] = 1



    df.loc[

        (df['title'] != 'Bird Measurer (Assessment)') &

        (df['event_code'] == 4100) & (df['type'] == 'Assessment'), 'attempt'] = 1



    return df

    

train_df_comb = set_attempt_label(train_df_comb)

test_df_comb = set_attempt_label(test_df_comb)
def get_accuracy_group(row):

    if row.correct == 0:

        return 0

    

    if row.attempt > 2:

        return 1

    

    if row.attempt == 2:

        return 2

    

    if row.attempt == 1:

        return 3





def get_labels(df):

    num_correct = df[df.attempt == 1].groupby(['game_session', 'installation_id']).correct.sum().astype(int)

    num_attempts = df[df.attempt == 1].groupby(['game_session', 'installation_id']).attempt.sum().astype(int)

    titles = df[df.attempt == 1].groupby(['game_session', 'installation_id']).title.agg(lambda x: x.iloc[0])

    labels_joined = num_correct.to_frame().join(num_attempts).join(titles).reset_index()

    labels_joined['accuracy_group'] = labels_joined.apply(get_accuracy_group, axis=1)

    return labels_joined
train_labels_joined = get_labels(train_df_comb)

test_labels_joined = get_labels(test_df_comb)
train_labels_joined.accuracy_group.value_counts().plot.bar(title='Train labels dist')
test_labels_joined.accuracy_group.value_counts().plot.bar(title='Test labels dist')
def _do_installation_id(inp, df):

    (installation_id, row) = inp



    game_sessions = row.game_session.unique()



    filtered_rows = df[df.installation_id == installation_id]



    start_idx = filtered_rows.head(1).index[0]



    output = []

    for game_session in game_sessions:

        assessment_row = filtered_rows[(filtered_rows.game_session == game_session) & (filtered_rows.event_code == 2000)]

        output.append((installation_id, game_session, start_idx, assessment_row.index[0]))



    return output





def add_start_and_end_pos(labels, df):

    labels_grouped = labels.groupby('installation_id')

    

    labels['start_idx'] = -1

    labels['end_idx'] = -1

    

    for row in tqdm(labels_grouped, total=len(labels_grouped)):

        results = _do_installation_id(row, df=df)



        for (installation_id, game_session, start_pos, end_pos) in results:

            filt = (labels.installation_id == installation_id) & (labels.game_session == game_session)



            labels.loc[filt, 'start_idx'] = start_pos

            labels.loc[filt, 'end_idx'] = end_pos
add_start_and_end_pos(train_labels_joined, train_df_comb)

add_start_and_end_pos(test_labels_joined, test_df_comb)
train_labels_joined.to_feather(OUTPUT_PATH/'train_labels.fth')

test_labels_joined.to_feather(OUTPUT_PATH/'test_labels.fth')