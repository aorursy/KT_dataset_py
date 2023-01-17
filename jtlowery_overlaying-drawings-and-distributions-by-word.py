import numpy as np
import pandas as pd
import os
import glob
import ast
import datetime
import random
import joblib
from dask import delayed
import dask.dataframe as dd

import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
np.warnings.filterwarnings('ignore')
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
train_filepaths = glob.glob('../input/train_simplified/*.csv')
train_filepaths
# change this set to compare different classes
selected_classes = {'sleeping bag', 'house', 'dragon', 'The Mona Lisa', 'star', 'lightning'}
selected_fps = [f'../input/train_simplified/{x}.csv' for x in selected_classes]
def make_df(fp):
    dtypes = {
    'countrycode': 'category',
    'drawing': np.str,
    'key_id': np.uint64,
    'recognized': np.bool_,
    'timestamp': np.str,
    'word': 'category',
    }
    df = pd.read_csv(fp, dtype=dtypes)
    df['word'] = df['word'].replace(' ', '_', regex=True)
    df["drawing"] = df["drawing"].apply(lambda x: ast.literal_eval(x))
    df["timestamp"] = df["timestamp"].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
    return df
def plot_drawing(drawing, ax, **kwargs):
    for line_set in drawing:
        xs, ys = line_set
        ax.plot(xs, ys, **kwargs)

def overlay_drawings(drawings, ax, **kwargs):
    for drawing in drawings:
        plot_drawing(drawing, ax, **kwargs)
def plot_overlays(df, sample_size=100, **kwargs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    n_rec = df['recognized'].sum()
    n_not_rec = (df['recognized'] == False).sum()
    n_rec = n_rec if sample_size > n_rec else sample_size
    n_not_rec = n_not_rec if sample_size > n_not_rec else sample_size
    
    overlay_drawings(df[df['recognized']]['drawing'].sample(n_rec).tolist(), axes[0], **kwargs)
    axes[0].set_title('recognized', fontsize=14)
    overlay_drawings(df[~df['recognized']]['drawing'].sample(n_not_rec).tolist(), axes[1], **kwargs)
    axes[1].set_title('not recognized', fontsize=14)
    
    fig.suptitle(df.word[0], y=1.02, fontsize=14)
    fig.tight_layout()
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    axes[0].axis('off')
    axes[1].axis('off')
selected_dfs = joblib.Parallel(n_jobs=4)(joblib.delayed(make_df)(fp) for fp in selected_fps)
# plot some sample drawings for each word
n_samples = 4
for df in selected_dfs:
    
    fig, axes = plt.subplots(1, n_samples, figsize=(6*n_samples, 6))
    for ax in axes:
        sample_df = df.sample(1)
        plot_drawing(sample_df.drawing.tolist()[0], ax)
        ax.invert_yaxis()
        ax.axis('off')
        rec = sample_df.recognized.tolist()[0]
        rec_text = "recognized" if sample_df.recognized.tolist()[0] else "not recognized"
        ax.set_title(rec_text, fontsize=12)
    fig.suptitle(df.word[0], y=1.02, fontsize=14)
    fig.tight_layout()
# plot many samples overlaid ontop of each other for each word (separating recognized vs not recognized into two plots)
n_samples = 500
for df in selected_dfs:
    plot_overlays(df, n_samples, linewidth=5, alpha=.01, color='blue')

def make_df_summary(fp):
    dtypes = {
    'countrycode': 'category',
    'drawing': np.str,
    'key_id': np.uint64,
    'recognized': np.bool_,
    'timestamp': np.str,
    'word': 'category',
    }
    df = pd.read_csv(fp, dtype=dtypes)
    df['word'] = df['word'].replace(' ', '_', regex=True).astype('category')
    df["drawing"] = df["drawing"].apply(lambda x: ast.literal_eval(x))
    df["number_lines"] = df["drawing"].apply(lambda x: len(x)).astype(np.uint16)
    df["mean_points_in_line"] = df["drawing"].apply(lambda x: np.mean([len(line[0]) for line in x])).astype(np.float32)
    df["timestamp"] = df["timestamp"].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
    df = df.drop(["drawing", "key_id"], axis='columns')
    return df
dfs = joblib.Parallel(n_jobs=4)(joblib.delayed(make_df_summary)(fp) for fp in selected_fps)
df = dd.concat(dfs, axis=0, interleave_partitions=True)
df = df.compute()
df.head()
def plot_by_word_dists(df, feature):
    words = [word for word in df.word.unique()]
    hist_data = [df[df['word']==word][feature] for word in words]
    fig = ff.create_distplot(hist_data, words, show_curve=False, show_rug=False)
    fig['layout'].update(xaxis=dict(title=feature), yaxis=dict(title='Frequency'))
    iplot(fig, filename=f'{feature}_plot')
plot_by_word_dists(df, feature='number_lines')
plot_by_word_dists(df, feature='mean_points_in_line')
def plot_recognized_vs_not_recognized_by_word(df):
    agg_df = df.groupby('word').agg({'recognized': ['sum', 'size']})
    agg_df.columns = agg_df.columns.map('_'.join)
    agg_df = agg_df.rename(columns={'recognized_sum': 'recognized', 'recognized_size': 'size'})
    agg_df['not_recognized'] = agg_df['size'] - agg_df['recognized']
    agg_df['not_recognized'] /= agg_df['size']
    agg_df['recognized'] /= agg_df['size']
    agg_df.index = agg_df.index.astype(np.str)
    print(agg_df.head())
    agg_df.iplot(kind='bar', keys=['recognized', 'not_recognized'], filename='recognized_vs_not', xTitle='Word', yTitle='Frequency', title='Recognized vs Not Recognized by Word')
plot_recognized_vs_not_recognized_by_word(df)
def plot_count_by_word(df):
    agg_df = df.groupby('word').agg({'recognized': ['sum', 'size']})
    agg_df.columns = agg_df.columns.map('_'.join)
    agg_df = agg_df.rename(columns={'recognized_sum': 'recognized', 'recognized_size': 'size'})
    agg_df.index = agg_df.index.astype(np.str)
    agg_df['size'].iplot(kind='bar', filename='samples_per_word', xTitle='Word', yTitle='Count', title='Number of Samples per Word')
plot_count_by_word(df)
def plot_count_by_word_by_country(df, country_cols=None):
    if not country_cols:
        country_cols = ['US', 'GB', 'CA', 'DE', 'AU', 'RU', 'FI', 'BR', 'KR']
    
    agg_df = df.groupby(['word', 'countrycode']).size()
    agg_df.name = 'count'
    agg_df = agg_df.reset_index(drop=False)

    words = list(agg_df.word.unique())
    countrycodes = list(agg_df.countrycode.unique())

    records = []
    for word in agg_df.word.unique():
        counts_for_countries = {x: 0 for x in countrycodes}
        word_df = agg_df[agg_df.word == word]
        dicts = word_df[['countrycode', 'count']].to_dict(orient='records')
        present_counts_for_countries = {d['countrycode']: d['count'] for d in dicts}
        counts_for_countries.update(present_counts_for_countries)
        counts_for_countries['word'] = word
        records.append(counts_for_countries)
        
    df = pd.DataFrame(records)
    df.index = df.pop('word')
    
    df[country_cols].iplot(kind='bar', filename='samples_per_word_per_country', xTitle='Word', yTitle='Count')
plot_count_by_word_by_country(df)
