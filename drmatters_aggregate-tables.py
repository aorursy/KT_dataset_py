import ast

import os

import glob

import pathlib

import pandas as pd

import numpy as np

import seaborn as sns



INPUT_PATH = pathlib.Path('../input')
def get_submission_params(fpath):

    def get_param(key, fname):

        start_index = fname.find(key)

        value = fname[start_index + len(key):].split('(')[1].split(')')[0]

        return value

    fname = os.path.basename(fpath)

    return {

        'model_name': get_param('model', fname),

        'data_split': True if get_param('data', fname) == 'split' else False,

        'metric': int(get_param('metric', fname)),

        'split_articles': bool(get_param('SplitArticles', fname)),

        'additional': get_param('additional', fname),

    }





def unpack_lists(col):

    is_scalar = max(col.apply(lambda s: len(ast.literal_eval(s)))) == 1

    if is_scalar:

        res = col.apply(lambda s: ast.literal_eval(s)[0])

    else:

        res = col.apply(ast.literal_eval)

    return res





def get_submission(name: str, read_texts=False):

    folder = INPUT_PATH / name

    fpath = glob.glob(str(folder) + '/sub_*')[0]

    

    cols = {'id', 'probas_title', 'probas_text'}

    if read_texts:

        cols = cols | {'title', 'text'}

    df = pd.read_csv(fpath, usecols=cols)

    df['probas_title'] = unpack_lists(df['probas_title'])

    df['probas_text'] = unpack_lists(df['probas_text'])

    params = get_submission_params(fpath)

    return df, params
print('Reading submissions')

ru_short_orig, p_ru_short = get_submission('rubert-short-v1', True)

ru_full_orig, p_ru_full = get_submission('rubert-full-v1')

mul_short_orig, p_mul_short = get_submission('bert-multi-uncased')
ru_short = ru_short_orig.rename(columns={'probas_title': 'ru_short_title', 'probas_text': 'ru_short_text'})

ru_full = ru_full_orig.rename(columns={'probas_title': 'ru_full_title', 'probas_text': 'ru_full_text'})

mul_short = mul_short_orig.rename(columns={'probas_title': 'mul_short_title', 'probas_text': 'mul_short_text'})
ru_short['ru_short_max_text'] = ru_short['ru_short_text'].apply(max)

mul_short['mul_short_max_text'] = mul_short['mul_short_text'].apply(max)
agg = pd.concat([

    ru_short.drop(['ru_short_text'], axis=1),

    ru_full.drop(['id'], axis=1),

    mul_short.drop(['id', 'mul_short_text'], axis=1)

], axis=1)



agg['weighted_60_25_15'] = agg['ru_short_max_text'] * 0.6 + agg['ru_full_text'] * 0.25 + agg['mul_short_max_text'] * 0.15

agg['avg_ru_short_ru_full'] = (agg['ru_short_max_text'] + agg['ru_full_text']) / 2

agg['avg_minus_multi'] = agg['avg_ru_short_ru_full'] - agg['mul_short_max_text']

agg['ru_short_minus_ru_full'] = agg['ru_short_max_text'] - agg['ru_full_text']

agg['ru_short_minus_multi'] = agg['ru_short_max_text'] - agg['mul_short_max_text']

agg.head(3)
sns.distplot(agg['ru_short_max_text'], axlabel='Model 1')
sns.distplot(agg['ru_full_text'], axlabel='Model 2')
sns.distplot(agg['mul_short_max_text'], axlabel='Model 3')
agg_no_titles = agg[[col for col in agg.columns if '_title' not in col]]

agg_no_titles.head(2)
agg_no_titles.to_csv('agg_no_titles.csv', index=False)

agg.to_csv('agg.csv', index=False)
submission = agg[['id', 'title', 'text', 'weighted_60_25_15', 'avg_ru_short_ru_full',

                  'ru_short_max_text', 'ru_full_text', 'mul_short_max_text']]

submission = submission.sort_values('weighted_60_25_15', ascending=False)
submission.to_csv('submission.csv', index=False)