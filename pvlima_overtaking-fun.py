%matplotlib inline

import math

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
races = pd.read_csv('../input/all_races.csv', nrows=146928)
def clean_data(df):

    # fix one missing net time

    flt = (df.place==1)&(df.event=='family-race')&(df.event_year==2015)

    df.loc[flt, 'net_time'] = df[flt].official_time

    df.official_time = pd.to_timedelta(df.official_time)

    # many cases of missing net time

    df.net_time = np.where(df.net_time=='-', np.nan, df.net_time)

    df.net_time = pd.to_timedelta(df.net_time)

    # extract the date

    df['birth_year'] = df['birth_date'].str[6:].astype(int)

    return df

    

def input_missing_event_net_time(df):

    # if the runner is on top10, set missing net time equal to the official time

    df.net_time = np.where((df.net_time.isnull())&(df.place <= 10), 

                            df.official_time, df.net_time)

    df['delay_time'] = df['official_time'].dt.seconds - df['net_time'].dt.seconds

    df['delay_time_mean'] = df.delay_time.rolling(window=10, min_periods=5).mean()

    df['net_time_mean_sec'] = df['official_time'].dt.seconds - df['delay_time_mean']

    df['net_time'] = np.where(df.net_time.isnull(),

                            pd.to_timedelta(df.net_time_mean_sec, unit='s'), 

                            df.net_time)

    df = df.drop(['net_time_mean_sec','delay_time_mean','delay_time'], axis=1)

    assert not (df.official_time < df.net_time).any() 

    assert not df.net_time.isnull().any() 

    return df



def count_overtaking(s, dd):

    start_before_me = np.where(dd.start_position < s.start_position, True, False)

    end_before_me = np.where(dd.place < s.place, True, False)

    overtake = np.sum((start_before_me)&(~end_before_me))

    was_overtaken = np.sum((~start_before_me)&(end_before_me))

    return pd.Series([s.place, overtake, was_overtaken])



def calculate_overtaking(df):

    # check we only get a single event as input

    assert df.event_year.nunique() == 1 and df.event.nunique() == 1

    df['start_position'] = df['delay_time_min'].rank(method='first')

    res = df.apply(count_overtaking, args=(df,), axis=1)

    res.columns = ['place','overtake','was_overtaken']

    df = pd.merge(df, res, on='place', how='left')

    df['rank_overtaking'] = df['overtake'].rank(method='first', ascending=False)

    df['rank_was_overtaken'] = df['was_overtaken'].rank(method='first', ascending=False)

    return df



def add_features(df):

    df['pace'] = df.net_time / df.distance

    df['net_time_min'] = df.net_time.dt.seconds/60

    df['official_time_min'] = df.official_time.dt.seconds/60

    df['delay_time_min'] = df['official_time_min'] - df['net_time_min']

    return df
races = clean_data(races)

races = races.groupby(['event','event_year']).apply(input_missing_event_net_time)

races = add_features(races)

races = races.groupby(['event','event_year']).apply(calculate_overtaking)
races[races.rank_overtaking==1][['name','place','official_time','net_time','overtake','rank_overtaking']].reset_index().sort_values(['event','event_year','rank_overtaking'])
odf = races[races.rank_overtaking<=100].groupby(['name','birth_year']).agg({'place':'count', 'overtake':'sum'})

odf['overtake_avg'] = odf.overtake / odf.place

odf.sort_values('place', ascending=False).head()
races[races.rank_was_overtaken==1][['name','place','official_time','net_time','was_overtaken','rank_was_overtaken']].reset_index().sort_values(['event','event_year'])