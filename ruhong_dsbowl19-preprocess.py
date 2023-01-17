import os

import re

import random

import gc

import warnings

from collections import namedtuple

from math import sqrt

import json

import numpy as np

import pandas as pd

import lightgbm as lgb

import sklearn

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.pipeline import Pipeline

from sklearn.metrics import cohen_kappa_score, make_scorer, mean_squared_error, accuracy_score

from sklearn.model_selection import GridSearchCV, PredefinedSplit

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm

from scipy.stats import ks_2samp



pd.show_versions()
def _log(str):

    os.system(f'echo \"{str}\"')

    print(str)
INPUT_ROOT = '../input/data-science-bowl-2019'

JOIN_KEY = ['installation_id', 'game_session', 'title']

TARGET = 'accuracy_group'

FEATURES = {

    'event_id', 

    'game_session', 

    'timestamp', 

    'installation_id', 

    'event_count',

    'event_code', 

    'game_time', 

    'title', 

    'type', 

    'world',

    'event_data'

}

EVENT_CODES = ['2000', '2010', '2020', '2025', '2030', '2035', '2040', '2050', '2060', '2070', '2075', '2080', '2081', '2083', '3010', '3020', '3021', '3110', '3120', '3121', '4010', '4020', '4021', '4022', '4025', '4030', '4031', '4035', '4040', '4045', '4050', '4070', '4080', '4090', '4095', '4100', '4110', '4220', '4230', '4235', '5000', '5010']

SEED = 31
def _init():

    # Characters such as empty strings '' or numpy.inf are considered NA values

    pd.set_option('use_inf_as_na', True)

    pd.set_option('display.max_columns', 999)

    pd.set_option('display.max_rows', 999)

    

    

_init()
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)





seed_everything(SEED)
RANDOM_VALUES = []

for _ in range(4000):

    RANDOM_VALUES.append(random.random())

    

_log(f'RANDOM_VALUES={RANDOM_VALUES[:20]}')
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



for dirname, _, filenames in os.walk(INPUT_ROOT):

    for filename in filenames:

        _log(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%%time

train_raw = pd.read_csv(f'{INPUT_ROOT}/train.csv', usecols=FEATURES)

train_labels = pd.read_csv(f'{INPUT_ROOT}/train_labels.csv', usecols=JOIN_KEY + [TARGET])

test_raw = pd.read_csv(f'{INPUT_ROOT}/test.csv', usecols=FEATURES)

train_labels.info()
def _remove_unlabelled_data(train_raw, train_labels):

    return train_raw[train_raw['installation_id'].isin(train_labels['installation_id'].unique())]





train_raw = _remove_unlabelled_data(train_raw, train_labels)
%%time

def _add_labels(train_raw, train_labels, on):

    return pd.merge(train_raw, train_labels, on=on, how='left')





train_raw = _add_labels(train_raw, train_labels, on=JOIN_KEY)

del train_labels
def _concat_columns(df1, df2):

    """Concatenate the columns of two pandas dataframes in the order of the operands.

    Both dataframes must have the same number of rows.

    """

    assert len(df1) == len(df2)

    res = pd.concat([df1, df2.reindex(df1.index)], axis=1, join='inner')

    assert len(res) == len(df1)

    return res

    



def _extract_event_data(df, keep_cols, chunk_size=1000000):

    res = pd.DataFrame()

    _len = len(df)

    for i in tqdm(range(0, _len, chunk_size)):

        if i + chunk_size < _len:

            chunk = df[i:i + chunk_size].copy()

        else:

            chunk = df[i:].copy()

        ed = pd.io.json.json_normalize(chunk['event_data'].apply(json.loads)).add_prefix('ed.')

        ed = ed[keep_cols].astype(np.float32)

        chunk = _concat_columns(chunk, ed)

        # sort=False because not all rows have same fields in event_data

        res = pd.concat([res, chunk], ignore_index=True, sort=False)

    # this line is too slow and OOM error!

    #res[keep_cols] = res[keep_cols].fillna(-1).astype(np.float32)

    assert len(df) == len(res)

    return res





#keep_cols = ['ed.duration', 'ed.level', 'ed.round', 'ed.correct', 'ed.misses','ed.weight', 'ed.total_duration']

#keep_cols = ['ed.identifier', 'ed.duration', 'ed.level', 'ed.round', 'ed.correct', 'ed.misses','ed.weight', 'ed.total_duration', 'ed.source']

#train_raw = _extract_event_data(train_raw, keep_cols)

#test_raw = _extract_event_data(test_raw, keep_cols)
test_raw.info(max_cols=999)
train_raw.info(max_cols=999)
# All event ids in test set also exist in train set

#test_set = set(test_raw['event_id'])

#train_set = set(train_raw['event_id'])

#vs = test_set - train_set

#_log(f'{len(vs)} event_ids exist in test set but not train set.')
EVENT_IDS = sorted(list(set(train_raw['event_id']) | set(test_raw['event_id'])))

_log(f'{len(EVENT_IDS)} EVENT_IDS={EVENT_IDS}')
TITLES = test_raw['title'].unique()

test_raw['title'].value_counts()
TYPES = test_raw['type'].unique()

test_raw['type'].value_counts()
WORLDS = test_raw['world'].unique()

test_raw['world'].value_counts()
#test_raw['ed.source'].value_counts()
#test_raw['ed.identifier'].value_counts()
vs = sorted(train_raw['event_code'].unique())

_log(f'{len(vs)} train_raw type={vs}')
%%time

_prev = pd.to_datetime(pd.Series(['2019-08-06T05:22:41.147000000'])).astype(np.int64).values[0]

_next = pd.to_datetime(pd.Series(['2019-08-06T05:22:41.147000001'])).astype(np.int64).values[0]

assert _next - _prev == 1





def _transform_timestamp(df):

    vs = pd.to_datetime(df['timestamp'])

    df['timestamp'] = vs

    assert df['timestamp'].notna().all()

    df['timestamp_int'] = vs.astype(np.int64)

    assert df['timestamp_int'].notna().all()





_transform_timestamp(train_raw)

_transform_timestamp(test_raw)
%%time

def _set_string_type(df, cols):

    df[cols] = df[cols].astype(str)

    return df





cols = ['event_code', 'timestamp']

train_raw = _set_string_type(train_raw, cols=cols)

test_raw = _set_string_type(test_raw, cols=cols)
%%time

def _sort_it(df):

    return df.sort_values(by=['installation_id', 'timestamp'])





#train_raw = _sort_it(train_raw)

#test_raw = _sort_it(test_raw)
vs = train_raw[train_raw[TARGET].notna()].groupby('installation_id', as_index=False)[TARGET].nunique()

vs
#_log(f'train_raw[timestamp] is from {train_raw.timestamp.min()} to {train_raw.timestamp.max()}')

#_log(f'test_raw[timestamp] is from {test_raw.timestamp.min()} to {test_raw.timestamp.max()}')
#vs = train_raw[(train_raw['installation_id'] == '0006a69f') & (train_raw[TARGET].notna())].groupby(['game_session', 'title', TARGET], as_index=False)['timestamp'].max()

#vs = sorted(vs['timestamp'].values) 

#vs
#tmp = test_raw.groupby(['world', 'type'], as_index=False)['game_time'].quantile([.25, .5, .75], interpolation='lower')

#tmp.head()
train_raw.info(max_cols=999)
test_raw.info(max_cols=999)
def _key(s):

    return re.sub(r'[\W\s]', '', s).lower()





def _timestamp_cutoffs(df, TARGET):

    res = df[df[TARGET].notna()].copy().groupby(['game_session', 'title', TARGET], as_index=False)['timestamp_int'].max()

    res = sorted(res['timestamp_int'].values)

    return res



    

def _target_variable(df, TARGET):

    vs = df[TARGET].copy().dropna().unique()

    assert len(set(vs)) == 1

    return vs[0]

    



def _group_stats(df, col, titles, types, worlds, suffix):

    """Get percentile stats"""

    res = {}

    _percentiles = [0.25, 0.5, 0.75]

    defaults = {}

    qs = df[col].quantile(_percentiles, interpolation='lower').to_numpy()

    for i, q in enumerate(qs):

        percentile = f'p{int(_percentiles[i] * 100)}'

        defaults[percentile] = q

        k = f'{col}_{percentile}{suffix}'

        res[k] = np.float32([q])

    

    # initialize values

    for p in _percentiles:

        percentile = f'p{int(p * 100)}'

        for w in worlds:

            for t in types:

                k = f'{col}_{percentile}_{_key(w)}_{_key(t)}{suffix}'

                res[k] = np.float32([defaults[percentile]])



        for t in titles:

            k = f'{col}_{percentile}_{_key(t)}{suffix}' 

            res[k] = np.float32([defaults[percentile]])



    if len(worlds) != 0 and len(types) != 0:

        tmp = df.groupby(['world', 'type'], as_index=False)[col].quantile(_percentiles, interpolation='lower')

        i = 0

        for row in tmp.itertuples(index=False):

            percentile = f'p{int(_percentiles[i % 3] * 100)}'

            k = f'{col}_{percentile}_{_key(row[0])}_{_key(row[1])}{suffix}'

            i += 1

            if k in res and not np.isnan(row[2]):

                res[k] = np.float32([row[2]])

    

    if len(titles) != 0:

        tmp = df.groupby(['title'], as_index=False)[col].quantile(_percentiles, interpolation='lower')

        i = 0

        for row in tmp.itertuples(index=False):

            percentile = f'p{int(_percentiles[i % 3] * 100)}'

            k = f'{col}_{percentile}_{_key(row[0])}{suffix}'

            i += 1

            if k in res and not np.isnan(row[1]):

                res[k] = np.float32([row[1]])

    return res    

    



def _game_session_stats(df, col, titles, types, worlds, suffix):

    """Deprecated."""

    res = {}

    _default = -1

    # initialize values

    k = f'{col}_gamesession_p50{suffix}'

    res[k] = np.float32([_default])

    for w in worlds:

        for t in types:

            k = f'{col}_gamesession_{_key(w)}_{_key(t)}_p50{suffix}'

            res[k] = np.float32([_default])



    for t in titles:

        k = f'{col}_gamesession_{_key(t)}_p50{suffix}' 

        res[k] = np.float32([_default])



    tmp = df.groupby(['game_session'], as_index=False)[col].quantile(.75, interpolation='lower')

    k = f'{col}_gamesession_p50{suffix}'

    if k in res and not tmp[col].isna().all():

        v = tmp[col].median()

        res[k] = np.float32([v])



    if len(worlds) != 0 and len(types) != 0:

        tmp = df.groupby(['game_session', 'world', 'type'], as_index=False)[col].quantile(.75, interpolation='lower')

        tmp.dropna(subset=[col], inplace=True)

        tmp = tmp.groupby(['world', 'type'], as_index=False)[col].median()

        for row in tmp.itertuples(index=False):

            k = f'{col}_gamesession_{_key(row[0])}_{_key(row[1])}_p50{suffix}'

            if k in res:

                res[k] = np.float32([row[2]])



    if len(titles) != 0:

        tmp = df.groupby(['game_session', 'title'], as_index=False)[col].quantile(.75, interpolation='lower')

        tmp.dropna(subset=[col], inplace=True)

        tmp = tmp.groupby(['title'], as_index=False)[col].median()

        for row in tmp.itertuples(index=False):

            k = f'{col}_gamesession_{_key(row[0])}_p50{suffix}'

            if k in res:

                res[k] = np.float32([row[1]])

    

    #qs = vs.quantile([0.25, 0.5, 0.75], interpolation='lower').to_numpy()    

    return res





def _count(df, col, values, suffix):

    res = {}

    for v in values:

        res[f'{col}_{_key(v)}{suffix}'] = np.int32([0])

    

    if len(values) != 0:

        tmp = df.groupby([col], as_index=False).count()

        for row in tmp.itertuples(index=False):

            res[f'{col}_{_key(row[0])}{suffix}'] = np.int32([row[1]])

    

    return res

    



def _event_id_features(df, event_ids, titles, types, worlds, suffix):

    res = {}

    # initialize counts

    for eid in event_ids:

        res[f'eid_{eid}{suffix}'] = np.int32([0])      

        for w in worlds:

            for t in types:

                res[f'eid_{eid}_{_key(w)}_{_key(t)}{suffix}'] = np.int32([0])

            

        for t in titles:

            res[f'eid_{eid}_{_key(t)}{suffix}'] = np.int32([0])

                      

    tmp = df.groupby(['event_id'], as_index=False).count()

    for row in tmp.itertuples(index=False):

        res[f'eid_{row[0]}{suffix}'] = np.int32([row[1]])

        

    if len(worlds) != 0 and len(types) != 0:

        tmp = df.groupby(['event_id', 'world', 'type'], as_index=False).count()

        for row in tmp.itertuples(index=False):

            k = f'eid_{row[0]}_{_key(row[1])}_{_key(row[2])}{suffix}'

            if k in res:

                res[k] = np.int32([row[3]])



    if len(titles) != 0:

        tmp = df.groupby(['event_id', 'title'], as_index=False).count()

        for row in tmp.itertuples(index=False):

            k = f'eid_{row[0]}_{_key(row[1])}{suffix}'

            if k in res:

                res[k] = np.int32([row[2]])

        

    return res





def _event_code_features(df, event_codes, titles, types, worlds, suffix):

    res = {}

    # initialize counts

    for code in event_codes:

        res[f'event_{code}{suffix}'] = np.int32([0])

        for w in worlds:

            for t in types:

                res[f'event_{code}_{_key(w)}_{_key(t)}{suffix}'] = np.int32([0])

            

        for t in titles:

            res[f'event_{code}_{_key(t)}{suffix}'] = np.int32([0])

        

    tmp = df.groupby(['event_code'], as_index=False).count()

    for row in tmp.itertuples(index=False):

        res[f'event_{row[0]}{suffix}'] = np.int32([row[1]])

        

    if len(worlds) != 0 and len(types) != 0:

        tmp = df.groupby(['event_code', 'world', 'type'], as_index=False).count()

        for row in tmp.itertuples(index=False):

            k = f'event_{row[0]}_{_key(row[1])}_{_key(row[2])}{suffix}'

            if k in res:

                res[k] = np.int32([row[3]])



    if len(titles) != 0:

        tmp = df.groupby(['event_code', 'title'], as_index=False).count()

        for row in tmp.itertuples(index=False):

            k = f'event_{row[0]}_{_key(row[1])}{suffix}'

            if k in res:

                res[k] = np.int32([row[2]])

        

    return res





def _event_data_features(df, suffix):

    res = {}

    res[f'ed_duration{suffix}'] = np.int32(df['ed.duration'].fillna(0).max())

    res[f'ed_total_duration{suffix}'] = np.int32(df['ed.total_duration'].fillna(0).max())

    res[f'ed_level{suffix}'] = np.int32(df['ed.level'].fillna(0).max())

    res[f'ed_round{suffix}'] = np.int32(df['ed.round'].fillna(0).max())

    res[f'ed_correct{suffix}'] = np.int32(df['ed.correct'].fillna(0).max())

    res[f'ed_misses{suffix}'] = np.int32(df['ed.misses'].fillna(0).max())

    res[f'ed_weight{suffix}'] = np.int32(df['ed.weight'].fillna(0).max())

    res[f'ed_source_resources{suffix}'] = np.int32([sum(df['ed.source'] == 'resources')])

    res[f'ed_source_right{suffix}'] = np.int32([sum(df['ed.source'] == 'right')])

    res[f'ed_source_left{suffix}'] = np.int32([sum(df['ed.source'] == 'left')])

    res[f'ed_source_scale{suffix}'] = np.int32([sum(df['ed.source'] == 'scale')])

    res[f'ed_source_middle{suffix}'] = np.int32([sum(df['ed.source'] == 'middle')])

    res[f'ed_source_heaviest{suffix}'] = np.int32([sum(df['ed.source'] == 'Heaviest')])

    res[f'ed_source_heavy{suffix}'] = np.int32([sum(df['ed.source'] == 'Heavy')])

    res[f'ed_source_lightest{suffix}'] = np.int32([sum(df['ed.source'] == 'Lightest')])

    n = 0

    for i in range(1, 13):

        n += sum(df['ed.source'] == str(i))

    res[f'ed_source_numbered{suffix}'] = np.int32([n])

    res[f'ed_id_dot{suffix}'] = np.int32([sum(df['ed.identifier'].str.contains('Dot_', regex=False))])

    res[f'ed_id_buddy{suffix}'] = np.int32([sum(df['ed.identifier'].str.contains('Buddy_', regex=False))])

    res[f'ed_id_cleo{suffix}'] = np.int32([sum(df['ed.identifier'].str.contains('Cleo_', regex=False))])

    res[f'ed_id_mom{suffix}'] = np.int32([sum(df['ed.identifier'].str.contains('Mom_', regex=False))])

    res[f'ed_id_sid{suffix}'] = np.int32([sum(df['ed.identifier'].str.contains('sid_', regex=False))])

    positives = {'Dot_SoCool', 'Dot_GreatJob', 'ohWow', 'wowSoCool', 'thatLooksSoCool', 'tub_success', 

                 'water_success', 'soap_success', 'Dot_Amazing', 'Dot_WhoaSoCool', 'Dot_ThatsIt', 'youDidIt_1305',

                 'SFX_completedtask', 'Cleo_AmazingPowers', 'RIGHTANSWER1', 'Dot_Awesome', 'greatJob_1306', 'YouDidIt',

                 'RIGHTANSWER3', 'RIGHTANSWER2', 'INSTRCOMPLETE', 'AWESOME', 'WayToGoTeam', 'Dot_NiceWorkAllMatch',

                 'GreatFlying', 'WeDidItOneRoundLeft', 'Cleo_AweOfYourSkills', 'Dot_NiceWork'}

    n_pos = 0

    for p in positives:

        n_pos += sum(df['ed.identifier'].str.contains(p, regex=False))

    res[f'ed_id_positive{suffix}'] = np.int32([n_pos])

    negatives = {'Dot_Uhoh', 'Dot_UhOh', 'Dot_NeedTryAgain', 'IncorrectTooHeavy', 'Dot_GoLower', 'Buddy_TryDifferentNest',

                 'Cleo_BowlTooLight', 'Dot_GoHigher', 'Dot_SoLow', 'Dot_SoHigh', 'Dot_WhoopsTooShort', 'IncorrectTooLight',

                 'NOT_THAT_HEAVY', 'Dot_UhOhTooTall', 'ADD_MORE_WEIGHT', 'wrong1', 'tryAgain1', 'Dot_TryWeighingAgain',

                 'Cleo_RememberHeavierBowl', 'Dot_Whoops', 'Dot_NotBalanced', 'Mom_TooManyContainers',

                 'WrongOver', 'Mom_TooMuchWater', 'Dot_ThatBucketNotRight', 'Dot_TryAgain', 'wrongFewer', 'WrongBetweenCliff',

                 'Mom_NeedMoreContainers', 'Dot_Try', 'Dot_HmTooSmall'}

    n_neg = 1

    for ne in negatives:

        n_neg += sum(df['ed.identifier'].str.contains(ne, regex=False))

    res[f'ed_id_negative{suffix}'] = np.int32([n_neg])

    res[f'ed_id_positive_ratio{suffix}'] = np.float32([n_pos / n_neg])

    return res

    

    

def _worlds_picked():

    return ['MAGMAPEAK', 'TREETOPCITY', 'CRYSTALCAVES']





def _titles_picked():

    return ['Cauldron Filler (Assessment)', 'Mushroom Sorter (Assessment)', 'Bird Measurer (Assessment)',

            'Cart Balancer (Assessment)', 'Chest Sorter (Assessment)'

           ]

    



def _types_picked():

    return ['Assessment', 'Game']

    

        

def _features_map(df, EVENT_CODES, EVENT_IDS, TITLES, TYPES, WORLDS, suffix=''):

    res = {}

    worlds = _worlds_picked()

    titles = _titles_picked()

    types = _types_picked()

    #cols = ['game_time', 'event_count', 'ed.duration', 'ed.level', 'ed.round','ed.correct','ed.misses','ed.weight','ed.total_duration']

    #for col in cols:

     #   res.update(_group_stats(df, col, titles=titles, types=types, worlds=worlds, suffix=suffix))

    

    res.update(_count(df, col='type', values=TYPES, suffix=suffix))

    res.update(_count(df, col='world', values=WORLDS, suffix=suffix))

    res.update(_count(df, col='title', values=TITLES, suffix=suffix))

    res.update(_event_code_features(df, EVENT_CODES, titles=titles, types=types, worlds=worlds, suffix=suffix))

    res.update(_event_id_features(df, EVENT_IDS, titles=titles, types=types, worlds=worlds, suffix=suffix))

    #res.update(_event_data_features(df, suffix))

    return res





def _features(df, installation_id, EVENT_CODES, EVENT_IDS, TITLES, TYPES, WORLDS):

    res = {}

    if TARGET in df.columns:

        res[TARGET] = np.int16([_target_variable(df, TARGET)])

    res['installation_id'] = [installation_id]    

    res.update(_features_map(df, EVENT_CODES, EVENT_IDS, TITLES, TYPES, WORLDS))

    return pd.DataFrame.from_dict(res)





def _preprocess(raw, EVENT_CODES, EVENT_IDS, TITLES, TYPES, WORLDS):

    res = pd.DataFrame()

    raw = raw.set_index('installation_id', drop=False)

    iids = raw['installation_id'].unique()

    prev_len = None

    prev_cols = None

    rv = 0

    for iid in tqdm(iids):

        whole = raw.loc[[iid]].copy()  # double square brackets return a Dataframe!

        whole = whole.set_index('timestamp_int', drop=False)

        dfs = []

        if TARGET in whole.columns:

            # train set: each installation id may contribute one or more examples.

            _prev = pd.to_datetime(pd.Series(['1999-01-01T05:22:41.147000000'])).astype(np.int64).values[0]

            for _curr in _timestamp_cutoffs(whole, TARGET):

                df = whole.loc[_prev + 1:_curr]

                dfs.append(df)

                _prev = _curr

        else:

            # test set: each installation id contributes one example.

            dfs.append(whole)

        j = -1

        if len(dfs) > 1:

            j = int(RANDOM_VALUES[rv])

            rv += 1

        for i, df in enumerate(dfs):

            if TARGET in df.columns:

                installation_id = f'{iid}_{i + 1}'

            else:

                installation_id = iid

            ex = _features(df, installation_id, EVENT_CODES, EVENT_IDS, TITLES, TYPES, WORLDS)

            if TARGET in df.columns:

                if i == j:

                    ex['_is_val'] = 0  # validation set

                else:

                    ex['_is_val'] = -1

            prev_len = len(ex.columns) if prev_len is None else prev_len

            prev_cols = set(ex.columns) if prev_cols is None else prev_cols

            if len(ex.columns) != prev_len:

                _diff = set(ex.columns) - prev_cols

                raise ValueError(f'Number of columns must be the same. Difference found={_diff}')

            prev_len = len(ex.columns)

            res = pd.concat([res, ex], ignore_index=True)

    return res



test = _preprocess(test_raw, EVENT_CODES, EVENT_IDS, TITLES, TYPES, WORLDS)
test.info(max_cols=9999)
assert test.notna().all(axis=None)

del test_raw

gc.collect()

test.head(20)
# budget of 4 seconds per iteration, or 4 hours total.

train = _preprocess(train_raw, EVENT_CODES, EVENT_IDS, TITLES, TYPES, WORLDS)
train.info(max_cols=9999)
assert train.notna().all(axis=None)

tmp = train.groupby(['_is_val'], as_index=False)['installation_id'].count()

assert tmp.iloc[1]['installation_id'] >= 2000

tmp.head()
del train_raw, tmp

gc.collect()

train.head(10)
train.to_parquet('train.parquet')

test.to_parquet('test.parquet')

_log(os.listdir("."))
%%time

train = pd.read_parquet('train.parquet')

test = pd.read_parquet('test.parquet')
def _log_transform(df, cols):

    df[cols] = np.float32(np.log(df[cols] + 1))





#cols = list(set(test.columns.values) - {'installation_id'})

#_log_transform(train, cols)

#_log_transform(test, cols)

#train.head()
%%time

def _scaling(dfs, cols, scaler=None):

    scaler = sklearn.preprocessing.RobustScaler() if scaler is None else scaler

    scaler.fit(dfs[0][cols])

    for df in dfs:

        df[cols] = np.float32(scaler.transform(df[cols]))

        assert df.notna().all(axis=None)





#scaler = sklearn.preprocessing.PowerTransformer()

cols = list(set(test.columns.values) - {'installation_id', '_is_val'})

_scaling([train, test], cols)

train.head()
train.to_parquet('train_scaled.parquet')

test.to_parquet('test_scaled.parquet')

_log(os.listdir("."))
%%time

train = pd.read_parquet('train_scaled.parquet')

test = pd.read_parquet('test_scaled.parquet')
def _select_features(df1, df2, features, alpha):

    res = []

    for f in tqdm(features):

        if ks_2samp(df1[f], df2[f]).pvalue > alpha:

            res.append(f)

    return res





ALPHA = 0.2

features = set(test.columns.values) - {'installation_id', '_is_val'}

PREDICTORS = _select_features(train, test, features, ALPHA)
dropped = sorted(list(features - set(PREDICTORS)))

PREDICTORS = sorted(PREDICTORS)

_log(f'alpha={ALPHA}, keep {len(PREDICTORS)}/{len(features)} features, drop {len(dropped)} features.\nkeep={PREDICTORS}')
_log(f'drop={dropped}')
train['is_solved'] = -1

train['solved_attempts'] = -1

train.loc[train[TARGET] == 0, ['is_solved']] = 0

train.loc[train[TARGET] != 0, ['is_solved']] = 1

train.loc[train[TARGET] == 3, ['solved_attempts']] = 1

train.loc[train[TARGET] == 2, ['solved_attempts']] = 2

train.loc[train[TARGET] == 1, ['solved_attempts']] = 3
%%time

cls_name = 'c/lgb/gbdt'

y_train_cls = train['is_solved']

x_train_cls = train[PREDICTORS]

p_split = PredefinedSplit(test_fold=train['_is_val'].values)

model = lgb.LGBMClassifier(n_estimators=10000, reg_alpha=1, objective='binary', boosting_type='gbdt')

pipe = Pipeline([('model', model)])

param_grid = {

    'model__learning_rate': [0.001],

    'model__min_child_samples': [100],

    'model__colsample_bytree': [0.01]

}

cls = GridSearchCV(pipe, cv=p_split, param_grid=param_grid, scoring='f1')

#cv.fit(x_train, y_train, model__early_stopping_rounds=200, model__verbose=500)

cls.fit(x_train_cls, y_train_cls)

assert cls.best_estimator_['model'].n_classes_ == 2

_log(f"""F1 {cls_name}

best_score_={cls.best_score_:.5f}

best_params_={cls.best_params_}

n_features={cls.best_estimator_['model'].n_features_}

""")
lgb.plot_importance(cls.best_estimator_['model'], max_num_features=100, figsize=(10, 30), title=f'{cls_name} feature importance')
%%time

clsd_name = 'c/lgb/dart'

model = lgb.LGBMClassifier(n_estimators=10000, reg_alpha=1, objective='binary', boosting_type='dart')

pipe = Pipeline([('model', model)])

param_grid = {

    'model__learning_rate': [0.001],

    'model__min_child_samples': [100],

    'model__colsample_bytree': [0.5]

}

clsd = GridSearchCV(pipe, cv=p_split, param_grid=param_grid, scoring='f1')

#cv.fit(x_train, y_train, model__early_stopping_rounds=200, model__verbose=500)

clsd.fit(x_train_cls, y_train_cls)

assert clsd.best_estimator_['model'].n_classes_ == 2

_log(f"""F1 {clsd_name}

best_score_={clsd.best_score_:.5f}

best_params_={clsd.best_params_}

n_features={clsd.best_estimator_['model'].n_features_}

""")
lgb.plot_importance(clsd.best_estimator_['model'], max_num_features=100, figsize=(10, 30), title=f'{clsd_name} feature importance')
%%time

def _random_forest_classifier(x_train_cls, y_train_cls):

    model = RandomForestClassifier(n_estimators=4000, max_features='log2')

    pipe = Pipeline([('model', model)])

    param_grid = {

        'model__max_depth': [4],

        'model__min_samples_leaf': [40]

    }

    rfc = GridSearchCV(pipe, cv=FOLDS, param_grid=param_grid, scoring='f1')

    rfc.fit(x_train_cls, y_train_cls)

    assert rfc.best_estimator_['model'].n_classes_ == 2

    return rfc





#rfc = _random_forest_classifier(x_train_cls, y_train_cls)

#_log(f"""F1 RandomForestClassifier

#best_score_={rfc.best_score_:.5f}

#best_params_={rfc.best_params_}

#n_features={rfc.best_estimator_['model'].n_features_}

#""")
def _rmse(y, y_pred):

    return sqrt(mean_squared_error(y, y_pred))





SCORING = make_scorer(_rmse, greater_is_better = False)
tmp = train[train['is_solved'] == 1]

y_train = tmp['solved_attempts']

x_train = tmp[PREDICTORS]

p_split = PredefinedSplit(test_fold=tmp['_is_val'].values)



split_df = tmp.groupby(['_is_val'], as_index=False)['installation_id'].count()

assert split_df.iloc[1]['installation_id'] >= 1500

split_df.head()
%%time

cv_name = 'r/lgb/gbdt'

model = lgb.LGBMRegressor(n_estimators=10000, reg_alpha=1, boosting_type='gbdt')

pipe = Pipeline([('model', model)])

param_grid = {

    'model__learning_rate': [0.001],

    'model__min_child_samples': [100],

    'model__colsample_bytree': [0.1]

}

cv = GridSearchCV(pipe, cv=p_split, param_grid=param_grid, scoring=SCORING)

#cv.fit(x_train, y_train, model__early_stopping_rounds=200, model__verbose=500)

cv.fit(x_train, y_train)

_log(f"""RMSE {cv_name}

best_score_={cv.best_score_:.5f}

best_params_={cv.best_params_}

n_features={cv.best_estimator_['model'].n_features_}

""")
# plot_metric only works with early stopping rounds

#lgb.plot_metric(cv.best_estimator_['model'])
lgb.plot_importance(cv.best_estimator_['model'], max_num_features=100, figsize=(10, 30), title=f'{cv_name} feature importance')
%%time

cvd_name = 'r/lgb/dart'

model = lgb.LGBMRegressor(n_estimators=20000, reg_alpha=1, boosting_type='dart')

pipe = Pipeline([('model', model)])

param_grid = {

    'model__learning_rate': [0.001],

    'model__min_child_samples': [100],

    'model__colsample_bytree': [0.5]

}

cvd = GridSearchCV(pipe, cv=p_split, param_grid=param_grid, scoring=SCORING)

#cv.fit(x_train, y_train, model__early_stopping_rounds=200, model__verbose=500)

cvd.fit(x_train, y_train)

_log(f"""RMSE {cvd_name}

best_score_={cvd.best_score_:.5f}

best_params_={cvd.best_params_}

n_features={cvd.best_estimator_['model'].n_features_}

""")
lgb.plot_importance(cvd.best_estimator_['model'], max_num_features=100, figsize=(10, 30), title=f'{cvd_name} feature importance')
%%time

def _random_forest_regressor(x_train, y_train):

    model = RandomForestRegressor(n_estimators=4000, max_features='log2')

    pipe = Pipeline([('model', model)])

    param_grid = {

        'model__max_depth': [4],

        'model__min_samples_leaf': [40]

    }

    rfr = GridSearchCV(pipe, cv=FOLDS, param_grid=param_grid, scoring=SCORING)

    rfr.fit(x_train, y_train)

    return rfr





#rfr = _random_forest_regressor(x_train, y_train)

#_log(f"""RMSE RandomForestRegressor

#best_score_={rfr.best_score_:.5f}

#best_params_={rfr.best_params_}

#n_features={rfr.best_estimator_['model'].n_features_}

#""")
%%time

BlendModel = namedtuple('BlendModel', ['model', 'name', 'weight'])



def _is_solved(score):

    if score >= 0.755:

        return 1

    return 0





def _solved_attempts(score):

    if score >= 2.2:

        return 3

    if score >= 1.35:

        return 2

    return 1





def _predict(df, classifiers, regressors):

    res = df[['installation_id']].copy()

    res[TARGET] = np.nan

    x_cls = df[PREDICTORS]

    res['is_solved'] = 0

    for m in classifiers:

        col = f'is_solved_{m.name}'

        res[col] = m.model.predict_proba(x_cls)[:,1]

        res['is_solved'] += res[col] * m.weight

    

    res['is_solved'] = np.int16(res['is_solved'].map(_is_solved))

    iids = set(res[res['is_solved'] == 1]['installation_id'].values)

    cols = ['installation_id'] + PREDICTORS

    tmp = df[df['installation_id'].isin(iids)][cols].copy()

    x = tmp[PREDICTORS]

    cols = ['installation_id', 'solved_attempts_raw', 'solved_attempts']

    tmp['solved_attempts_raw'] = 0

    for m in regressors:

        col = f'solved_attempts_{m.name}'

        cols.append(col)

        tmp[col] = m.model.predict(x)

        tmp['solved_attempts_raw'] += tmp[col] * m.weight

        

    tmp['solved_attempts'] = np.int16(tmp['solved_attempts_raw'].map(_solved_attempts))

    tmp = tmp[cols]

    res = res.merge(tmp, on='installation_id', how='left')

    res.loc[res['is_solved'] == 0, [TARGET]] = 0

    res.loc[(res['is_solved'] == 1) & (res['solved_attempts'] >= 3), [TARGET]] = 1

    res.loc[(res['is_solved'] == 1) & (res['solved_attempts'] == 2), [TARGET]] = 2

    res.loc[(res['is_solved'] == 1) & (res['solved_attempts'] <= 1), [TARGET]] = 3

    assert res[TARGET].notna().all(axis=None)

    res[TARGET] = np.int16(res[TARGET])

    return res





classifiers=[

    BlendModel(model=cls, weight=0.5, name=cls_name),

    BlendModel(model=clsd, weight=0.5, name=clsd_name)

]

regressors=[

    BlendModel(model=cv, weight=0.5, name=cv_name),

    BlendModel(model=cvd, weight=0.5, name=cvd_name)

]

oof = _predict(train, classifiers=classifiers, regressors=regressors)

oof.head(10)
plt.subplot(1, 2, 1)

plt.title(f'is_solved_{cls_name}')

oof[f'is_solved_{cls_name}'].plot(kind='hist')

plt.subplot(1, 2, 2)

plt.title(f'is_solved_{clsd_name}')

oof[f'is_solved_{clsd_name}'].plot(kind='hist')
plt.subplot(1, 2, 1)

plt.title(f'solved_attempts_{cv_name}')

oof[f'solved_attempts_{cv_name}'].plot(kind='hist')

plt.subplot(1, 2, 2)

plt.title(f'solved_attempts_{cvd_name}')

oof[f'solved_attempts_{cvd_name}'].plot(kind='hist')
oof.sort_values(by=['installation_id'], inplace=True)

train.sort_values(by=['installation_id'], inplace=True)

y_true = train[TARGET]

y_pred = oof[TARGET]

kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')

acc = accuracy_score(y_true, y_pred)

_log(f'oof kappa={kappa:.5f}, acc={acc:.5f}')
%%time

sub = _predict(test, classifiers=classifiers, regressors=regressors)

sub = sub[['installation_id', TARGET]]

sample_sub = pd.read_csv(f'{INPUT_ROOT}/sample_submission.csv')

assert sub['installation_id'].equals(sample_sub['installation_id'])

sub.head()
plt.subplot(1, 3, 1)

plt.title('test predict')

sub[TARGET].plot(kind='hist')

plt.subplot(1, 3, 2)

plt.title('oof predict')

oof[TARGET].plot(kind='hist')

plt.subplot(1, 3, 3)

plt.title('oof truth')

tmp = train[TARGET].copy()

tmp = tmp.astype(int)

tmp.plot(kind='hist')

plt.tight_layout()
sub.to_csv('submission.csv', index=False)

_log(os.listdir("."))