# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import datetime as dt 


df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv", index_col='SNo')

def clean_dataset(df):
    df.columns = df.columns.str.lower().str.replace('/','_')
    df.columns = df.columns.str.lower().str.replace(' ','_')
    df['last_update'] = pd.to_datetime(df.last_update)
    df['observationdate'] = pd.to_datetime(df.observationdate)
    return df

df = clean_dataset(df)

def state_added(df, state, metric, col):
    df = df[df.province_state.str.lower() == state][['observationdate', metric]]
    df = df.groupby(by='observationdate').sum()
    df['daily_added'] = df[metric].diff(periods=1)
    df['10_day_avg'] = df[metric].rolling(10, min_periods=1).mean()
    df['5_day_avg'] = df[metric].rolling(5, min_periods=1).mean()
    return df[col]

def region_added(df, region, metric, col):
    df = df[df.country_region.str.lower() == region][['observationdate', metric]]
    df = df.groupby(by='observationdate').sum()
    df['daily_added'] = df[metric].diff(periods=1)
    df['10_day_avg'] = df[metric].rolling(10, min_periods=1).mean()
    df['5_day_avg'] = df[metric].rolling(5, min_periods=1).mean()
    return df[col]

def generate_graph(df, metric, col, countries,states, horizon):
    graphs = {}
    if countries is not None:
        for country in countries: 
            graphs[country] = region_added(df, country, metric, col)
    if states is not None:
        for state in states:
            graphs[state]=state_added(df, state, metric, col)

    graphs = pd.DataFrame(graphs, index=graphs[countries[0]].index.date)
    graphs = graphs[-horizon:]
    
    if graphs.empty:
        print("no data to graph")
        return None
    else:
        return graphs.plot.bar(title='Regional deaths added daily')

def handle_input(df, metric, col, horizon):
    c_list = input("Please enter selected countries seperated by commas:").lower().split(",")
    c_list = [c.strip() for c in c_list]
    if len(c_list) == 0:
        c_list == None
    s_list = input("Please enter selected states seperated by commas:").lower().split(",")
    s_list = [s.strip() for s in s_list]
    if len(s_list) == 0:
        s_list == None
    return generate_graph(df, metric, col, countries=c_list, states=s_list, horizon=horizon)

# s_list = ['New York', 'Florida']
# r_list = ['Netherlands', 'Mainland China']
handle_input(df=df,metric='deaths', col='daily_added', horizon=30)




