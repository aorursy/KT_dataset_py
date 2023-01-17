import os

import re

import pandas as pd

import numpy as np

from pathlib import Path
%reload_ext autoreload

%autoreload 2
from fastai import *

#from fastai.basics import *
PATH=Path('../input/')

table_names = ['train', 'store', 'store_states', 'state_names', 'googletrend', 'weather', 'test']

tables = [pd.read_csv(PATH/f'{fname}.csv', low_memory=False) for fname in table_names]

train, store, store_states, state_names, googletrend, weather, test = tables

len(train),len(test)
train.StateHoliday = train.StateHoliday!='0'

test.StateHoliday = test.StateHoliday!='0'
def join_df(left, right, left_on, right_on=None, suffix='_y'):

    if right_on is None: right_on = left_on

    return left.merge(right, how='left', left_on=left_on, right_on=right_on, 

                      suffixes=("", suffix))
weather = join_df(weather, state_names, "file", "StateName")
googletrend['Date'] = googletrend.week.str.split(' - ', expand=True)[0]

googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]

googletrend.loc[googletrend.State=='NI', "State"] = 'HB,NI'
def add_datepart(df, fldname, drop=True, time=False):

    "Helper function that adds columns relevant to a date."

    fld = df[fldname]

    fld_dtype = fld.dtype

    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):

        fld_dtype = np.datetime64



    if not np.issubdtype(fld_dtype, np.datetime64):

        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)

    targ_pre = re.sub('[Dd]ate$', '', fldname)

    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',

            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']

    if time: attr = attr + ['Hour', 'Minute', 'Second']

    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())

    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9

    if drop: df.drop(fldname, axis=1, inplace=True)
add_datepart(weather, "Date", drop=False)

add_datepart(googletrend, "Date", drop=False)

add_datepart(train, "Date", drop=False)

add_datepart(test, "Date", drop=False)
trend_de = googletrend[googletrend.file == 'Rossmann_DE']
store = join_df(store, store_states, "Store")

len(store[store.State.isnull()])
joined = join_df(train, store, "Store")

joined_test = join_df(test, store, "Store")

len(joined[joined.StoreType.isnull()]),len(joined_test[joined_test.StoreType.isnull()])
joined = join_df(joined, googletrend, ["State","Year", "Week"])

joined_test = join_df(joined_test, googletrend, ["State","Year", "Week"])

len(joined[joined.trend.isnull()]),len(joined_test[joined_test.trend.isnull()])
joined = joined.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))

joined_test = joined_test.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))

len(joined[joined.trend_DE.isnull()]),len(joined_test[joined_test.trend_DE.isnull()])
joined = join_df(joined, weather, ["State","Date"])

joined_test = join_df(joined_test, weather, ["State","Date"])

len(joined[joined.Mean_TemperatureC.isnull()]),len(joined_test[joined_test.Mean_TemperatureC.isnull()])
for df in (joined, joined_test):

    for c in df.columns:

        if c.endswith('_y'):

            if c in df.columns: df.drop(c, inplace=True, axis=1)
for df in (joined,joined_test):

    df['CompetitionOpenSinceYear'] = df.CompetitionOpenSinceYear.fillna(1900).astype(np.int32)

    df['CompetitionOpenSinceMonth'] = df.CompetitionOpenSinceMonth.fillna(1).astype(np.int32)

    df['Promo2SinceYear'] = df.Promo2SinceYear.fillna(1900).astype(np.int32)

    df['Promo2SinceWeek'] = df.Promo2SinceWeek.fillna(1).astype(np.int32)
for df in (joined,joined_test):

    df["CompetitionOpenSince"] = pd.to_datetime(dict(year=df.CompetitionOpenSinceYear, 

                                                     month=df.CompetitionOpenSinceMonth, day=15))

    df["CompetitionDaysOpen"] = df.Date.subtract(df.CompetitionOpenSince).dt.days
for df in (joined,joined_test):

    df.loc[df.CompetitionDaysOpen<0, "CompetitionDaysOpen"] = 0

    df.loc[df.CompetitionOpenSinceYear<1990, "CompetitionDaysOpen"] = 0
for df in (joined,joined_test):

    df["CompetitionMonthsOpen"] = df["CompetitionDaysOpen"]//30

    df.loc[df.CompetitionMonthsOpen>24, "CompetitionMonthsOpen"] = 24

joined.CompetitionMonthsOpen.unique()
# If needed, uncomment:

# ! pip install isoweek
from isoweek import Week

for df in (joined,joined_test):

    df["Promo2Since"] = pd.to_datetime(df.apply(lambda x: Week(

        x.Promo2SinceYear, x.Promo2SinceWeek).monday(), axis=1).astype(pd.datetime))

    df["Promo2Days"] = df.Date.subtract(df["Promo2Since"]).dt.days
for df in (joined,joined_test):

    df.loc[df.Promo2Days<0, "Promo2Days"] = 0

    df.loc[df.Promo2SinceYear<1990, "Promo2Days"] = 0

    df["Promo2Weeks"] = df["Promo2Days"]//7

    df.loc[df.Promo2Weeks<0, "Promo2Weeks"] = 0

    df.loc[df.Promo2Weeks>25, "Promo2Weeks"] = 25

    df.Promo2Weeks.unique()
joined.to_pickle('joined')

joined_test.to_pickle('joined_test')
def get_elapsed(fld, pre):

    day1 = np.timedelta64(1, 'D')

    last_date = np.datetime64()

    last_store = 0

    res = []



    for s,v,d in zip(df.Store.values,df[fld].values, df.Date.values):

        if s != last_store:

            last_date = np.datetime64()

            last_store = s

        if v: last_date = d

        res.append(((d-last_date).astype('timedelta64[D]') / day1))

    df[pre+fld] = res
columns = ["Date", "Store", "Promo", "StateHoliday", "SchoolHoliday"]
#df = train[columns]

df = train[columns].append(test[columns])
fld = 'SchoolHoliday'

df = df.sort_values(['Store', 'Date'])

get_elapsed(fld, 'After')

df = df.sort_values(['Store', 'Date'], ascending=[True, False])

get_elapsed(fld, 'Before')
fld = 'StateHoliday'

df = df.sort_values(['Store', 'Date'])

get_elapsed(fld, 'After')

df = df.sort_values(['Store', 'Date'], ascending=[True, False])

get_elapsed(fld, 'Before')
fld = 'Promo'

df = df.sort_values(['Store', 'Date'])

get_elapsed(fld, 'After')

df = df.sort_values(['Store', 'Date'], ascending=[True, False])

get_elapsed(fld, 'Before')
df = df.set_index("Date")
columns = ['SchoolHoliday', 'StateHoliday', 'Promo']
for o in ['Before', 'After']:

    for p in columns:

        a = o+p

        df[a] = df[a].fillna(0).astype(int)
bwd = df[['Store']+columns].sort_index().groupby("Store").rolling(7, min_periods=1).sum()
fwd = df[['Store']+columns].sort_index(ascending=False

                                      ).groupby("Store").rolling(7, min_periods=1).sum()
bwd.drop('Store',1,inplace=True)

bwd.reset_index(inplace=True)
fwd.drop('Store',1,inplace=True)

fwd.reset_index(inplace=True)
df.reset_index(inplace=True)
df = df.merge(bwd, 'left', ['Date', 'Store'], suffixes=['', '_bw'])

df = df.merge(fwd, 'left', ['Date', 'Store'], suffixes=['', '_fw'])
df.drop(columns,1,inplace=True)
df.head()
df.to_pickle('df')
df["Date"] = pd.to_datetime(df.Date)
df.columns
joined = pd.read_pickle('joined')

joined_test = pd.read_pickle('joined_test')
joined = join_df(joined, df, ['Store', 'Date'])
joined_test = join_df(joined_test, df, ['Store', 'Date'])
joined = joined[joined.Sales!=0]
joined.reset_index(inplace=True)

joined_test.reset_index(inplace=True)
joined.to_pickle('train_clean')

joined_test.to_pickle('test_clean')