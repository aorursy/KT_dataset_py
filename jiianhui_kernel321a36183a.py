%matplotlib inline



def split_val(a, n):

    return a[:n].copy(), a[n:].copy()



# these are retrieved from Jeremy Howard's fastai library

from IPython.display import display

from pandas_summary import DataFrameSummary

import cv2, sklearn_pandas, sklearn

from sklearn import metrics, ensemble, preprocessing

from sklearn.ensemble import RandomForestClassifier

from IPython.lib.deepreload import reload as dreload

import PIL, os, numpy as np, math, collections, threading, json, bcolz, random, scipy

import pandas as pd, pickle, sys, itertools, string, sys, re, datetime, time, shutil, copy

import IPython, warnings, pdb

import contextlib

from abc import abstractmethod

from glob import glob, iglob

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from itertools import chain

from functools import partial

from collections import Iterable, Counter, OrderedDict

from IPython.lib.display import FileLink

from PIL import Image, ImageEnhance, ImageOps

from operator import itemgetter, attrgetter

from pathlib import Path

from distutils.version import LooseVersion

from matplotlib import pyplot as plt, rcParams, animation

matplotlib.rc('animation', html='html5')

np.set_printoptions(precision=5, linewidth=110, suppress=True)

from ipykernel.kernelapp import IPKernelApp

import tqdm as tq

from tqdm import tqdm_notebook, tnrange

from sklearn_pandas import DataFrameMapper

from sklearn.ensemble import forest

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.impute._base import SimpleImputer as Imputer

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype





def in_notebook(): return IPKernelApp.initialized()



def in_ipynb():

    try:

        cls = get_ipython().__class__.__name__

        return cls == 'ZMQInteractiveShell'

    except NameError:

        return False



def clear_tqdm():

    inst = getattr(tq.tqdm, '_instances', None)

    if not inst: return

    try:

        for i in range(len(inst)): inst.pop().close()

    except Exception:

        pass



if in_notebook():

    def tqdm(*args, **kwargs):

        clear_tqdm()

        return tq.tqdm(*args, file=sys.stdout, **kwargs)

    def trange(*args, **kwargs):

        clear_tqdm()

        return tq.trange(*args, file=sys.stdout, **kwargs)

else:

    from tqdm import tqdm, trange

    tnrange=trange

    tqdm_notebook=tqdm

    

def is_date(x): return np.issubdtype(x.dtype, np.datetime64)



def train_cats(df):

    for n,c in df.items():

        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()



def apply_cats(df, trn):

    for n,c in df.items():

        if (n in trn.columns) and (trn[n].dtype.name=='category'):

            df[n] = c.astype('category').cat.as_ordered()

            df[n].cat.set_categories(trn[n].cat.categories, ordered=True, inplace=True)



def fix_missing(df, col, name, na_dict):

    if is_numeric_dtype(col):

        if pd.isnull(col).sum() or (name in na_dict):

            df[name+'_na'] = pd.isnull(col)

            filler = na_dict[name] if name in na_dict else col.median()

            df[name] = col.fillna(filler)

            na_dict[name] = filler

    return na_dict



def numericalize(df, col, name, max_n_cat):

    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):

        df[name] = pd.Categorical(col).codes+1



def scale_vars(df, mapper):

    warnings.filterwarnings('ignore', category=sklearn.exceptions.DataConversionWarning)

    if mapper is None:

        map_f = [([n],StandardScaler()) for n in df.columns if is_numeric_dtype(df[n])]

        mapper = DataFrameMapper(map_f).fit(df)

    df[mapper.transformed_names_] = mapper.transform(df)

    return mapper



def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,

            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):

    if not ignore_flds: ignore_flds=[]

    if not skip_flds: skip_flds=[]

    if subset: df = get_sample(df,subset)

    else: df = df.copy()

    ignored_flds = df.loc[:, ignore_flds]

    df.drop(ignore_flds, axis=1, inplace=True)

    if preproc_fn: preproc_fn(df)

    if y_fld is None: y = None

    else:

        if not is_numeric_dtype(df[y_fld]): df[y_fld] = pd.Categorical(df[y_fld]).codes

        y = df[y_fld].values

        skip_flds += [y_fld]

    df.drop(skip_flds, axis=1, inplace=True)



    if na_dict is None: na_dict = {}

    else: na_dict = na_dict.copy()

    na_dict_initial = na_dict.copy()

    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)

    if len(na_dict_initial.keys()) > 0:

        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)

    if do_scale: mapper = scale_vars(df, mapper)

    for n,c in df.items(): numericalize(df, c, n, max_n_cat)

    df = pd.get_dummies(df, dummy_na=True)

    df = pd.concat([ignored_flds, df], axis=1)

    res = [df, y, na_dict]

    if do_scale: res = res + [mapper]

    return res



def get_nn_mappers(df, cat_vars, contin_vars):

    # Replace nulls with 0 for continuous, "" for categorical.

    for v in contin_vars: df[v] = df[v].fillna(df[v].max()+100,)

    for v in cat_vars: df[v].fillna('#NA#', inplace=True)



    # list of tuples, containing variable and instance of a transformer for that variable

    # for categoricals, use LabelEncoder to map to integers. For continuous, standardize

    cat_maps = [(o, LabelEncoder()) for o in cat_vars]

    contin_maps = [([o], StandardScaler()) for o in contin_vars]

left = pd.read_csv("../input/data-science-bowl-2019/train.csv")

right = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")

test = pd.read_csv("../input/data-science-bowl-2019/test.csv")
train = pd.merge(left, right, on=["installation_id", "game_session"])
left = right = None
train.drop(["title_y", "num_correct", "num_incorrect", "accuracy"], axis=1, inplace=True)
train.rename(columns={"title_x":"title"}, inplace=True)
tmp0 = pd.json_normalize(train.event_data.apply(json.loads))
tmp0.drop(["event_code", "event_count", "game_time"],axis=1, inplace=True)
tmp1 = pd.concat([tmp0, train], axis=1)
tmp1.drop("event_data", axis=1, inplace=True)
tmp1 = tmp1.explode("stumps")

tmp1 = tmp1.explode("hats")

tmp1 = tmp1.explode("caterpillars")

tmp1 = tmp1.explode("hats_placed")

tmp1 = tmp1.explode("buckets")

tmp1 = tmp1.explode("buckets_placed")

tmp1 = tmp1.explode("pillars")

tmp9 = tmp1.explode("chests")
tmp9.drop(["crystals", "left", "right"], inplace=True, axis = 1)
train = tmp9.copy()
tmp9 = tmp1 = tmp0 = None
train.drop("timestamp", axis=1, inplace=True)
train.drop("installation_id", axis=1, inplace=True)
train_cats(train)
train.event_code = train.event_code.astype("category", copy=False)

train.accuracy_group = train.accuracy_group.astype("category", copy=False)
full_X, full_y, nas = proc_df(train, "accuracy_group", max_n_cat = 7)
train_size = int((len(full_X)*0.8) // 1)

train_X, valid_X = split_val(full_X, train_size)

train_y, valid_y = split_val(full_y, train_size)
full_X = full_y = valid_X = valid_y = None
# histogram problem: we temporarily add 1 to label

train_y = train_y + 1
keep_cols = np.array(['game_session', 'event_count', 'game_time', 'coordinates.y', 'coordinates.x',

       'title_Chest Sorter (Assessment)', 'duration', 'hat', 'title_Bird Measurer (Assessment)',

       'correct_True', 'source', 'correct_False', 'event_id'])
train_X = train_X[keep_cols].copy()
m = RandomForestClassifier(n_jobs=-1, oob_score=False,

                          n_estimators = 100, min_samples_leaf = 10, max_features = "sqrt")

m.fit(train_X, train_y)
tes0 = pd.json_normalize(test.event_data.apply(json.loads))

tes0.drop(["event_code", "event_count", "game_time"],axis=1, inplace=True)

tes1 = pd.concat([tes0, test], axis=1)

tes1.drop("event_data", axis=1, inplace=True)



tes1 = tes1.explode("stumps")

tes1 = tes1.explode("hats")

tes1 = tes1.explode("caterpillars")

tes1 = tes1.explode("hats_placed")

tes1 = tes1.explode("buckets")

tes1 = tes1.explode("buckets_placed")

tes1 = tes1.explode("pillars")

tes9 = tes1.explode("chests")



test = tes9.drop(["crystals", "left", "right", "timestamp"], axis = 1)
tes0 = tes1 = tes9 = None
ind = test.loc[:,"installation_id"]
test.drop("installation_id", axis=1, inplace=True)
apply_cats(test, train)
test.event_code = test.event_code.astype("category", copy=False)
test2 = test[train.drop(["accuracy_group"], axis=1).columns]
test_X, _, _ = proc_df(test2, na_dict = nas, max_n_cat = 7)
test_X = test_X[keep_cols]
_ = train = train_X = train_y = None
pred = m.predict(test_X)
comb = np.array([ind, pred])
submit = pd.DataFrame(data=comb)
keep_cols = pred = comb = ind = m = None
submit2 = submit.T
submit3 = submit2.rename(columns={0:"installation_id", 1:"accuracy_group"})
submit2 = submit = None
submit3.accuracy_group = submit3.accuracy_group - 1
submit3.accuracy_group = submit3.accuracy_group.astype("int8", copy=False)
submit4 = submit3.groupby("installation_id").median()
submit4.accuracy_group = submit4.accuracy_group.astype(int)
submit4.to_csv("submission.csv")