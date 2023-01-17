# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numba import jit

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import warnings

warnings.filterwarnings('ignore')

print(os.listdir("../input"))

import re

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
from datetime import datetime, timedelta



def calculateDate(ordinal, _epoch0=datetime(1899, 12, 31)):

    ordinal = (ordinal*10000)-7000

    if ordinal > 59:

        ordinal -= 1  # Excel leap year bug, 1900 is not a leap year!

    return (_epoch0 + timedelta(days=ordinal)).replace(microsecond=0)

def add_datepart(df, fldname, drop=True, time=False, errors="raise"):

    fld = df[fldname]

    fld_dtype = fld.dtype

    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):

        fld_dtype = np.datetime64



    if not np.issubdtype(fld_dtype, np.datetime64):

        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)

    targ_pre = re.sub('[Dd]ate$', '', fldname)

    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',

            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']

    if time: attr = attr + ['Hour', 'Minute', 'Second']

    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())

    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9

    if drop: df.drop(fldname, axis=1, inplace=True)

dates = []

for i in range(len(train_df)):

    dates.append(calculateDate(train_df["var_68"][i]))

testdates = []

for i in range(len(test_df)):

    testdates.append(calculateDate(test_df["var_68"][i]))
train_df["date"]=dates

test_df["date"]=testdates

add_datepart(train_df,"date")

add_datepart(test_df,"date")
train_df[['Month','Week','Day','Dayofweek','Dayofyear','Is_month_end','Is_month_start','Is_quarter_end','Is_quarter_start','Is_year_end','Is_year_start']].head()