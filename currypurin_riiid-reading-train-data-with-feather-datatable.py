!pip install ../input/python-datatable/datatable-0.11.0-cp37-cp37m-manylinux2010_x86_64.whl

!mkdir ../tmp/
from time import time

from contextlib import contextmanager

import pandas as pd

from tqdm.auto import tqdm

import gc

import pickle

import datatable as dt



gc.enable()



@contextmanager

def timer(name):

    t0 = time()

    yield

    print(f'[{name}] done in {time() - t0:.2f} s')



def sizeof_fmt(num, suffix='B'):

    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:

        if abs(num) < 1024.0:

            return "%3.1f%s%s" % (num, unit, suffix)

        num /= 1024.0

    return "%.1f%s%s" % (num, 'Yi', suffix)
!du -h ../input/riiid-feather-format/train.feather
# read the feather format 10 times.

with timer('feather'):

    for _ in tqdm(range(10)):

        train_df = pd.read_feather('../input/riiid-feather-format/train.feather')
print(sizeof_fmt(train_df.memory_usage().sum()))
with timer('feather save'):

    train_df.to_feather('../tmp/train.feather')
with timer('pickle save'):

    with open('../tmp/train.pickle', 'wb') as f:

        pickle.dump(train_df, f)
!du -h ../tmp/train.pickle
with timer('pickle load'):

    for _ in tqdm(range(10)):

        with open('../tmp/train.pickle', 'rb') as f:

            train_df = pickle.load(f)
with timer('DataFrame save'):

    dt.Frame(train_df).to_jay("train.jay")
del train_df
with timer('Datatable'):

    for _ in tqdm(range(10)):

        train_dt = dt.fread('train.jay')
with timer('Datatable.Frame save'):

    train_dt.to_jay('train.jay')
type(train_dt)
!du -h train.jay
import sys

print(sizeof_fmt(sys.getsizeof(train_dt)))

del train_dt
with timer('Datatable to pd.DataFrame'):

    for _ in tqdm(range(10)):

        train_df = dt.fread('train.jay').to_pandas()
type(train_df)
train_df.dtypes