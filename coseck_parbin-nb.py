from argparse import Namespace
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from sklearn import model_selection
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
args = Namespace(
    data_csv = '/kaggle/input/parbin/data.csv',
    data_2_csv = '/kaggle/input/parbin/data_2.csv',
    data_3_csv = '/kaggle/input/parbin/data_3.csv',


    truth_csv = '/kaggle/input/parbin/truth.csv',
    truth_2_csv = '/kaggle/input/parbin/truth.csv',
    truth_3_csv = '/kaggle/input/parbin/truth.csv',
)
reading_df = pd.read_csv(args.data_csv)
del reading_df['Unnamed: 12']
reading_2_df = pd.read_csv(args.data_2_csv)


truth_df = pd.read_csv(args.truth_csv)
truth_2_df = pd.read_csv(args.truth_2_csv)
reading_df
def take_n_sec_samples(
    sensor_df:pd.DataFrame, label_df:pd.DataFrame, n:int, head:bool = True
):
    """
    Args:
        sensor_df: sensor reading df
        label_df: df containing label for every minute
        n: number of seconds to take, if not enough, takes as much as possible
        head: Take n rows from the head, else take n rows from the tail
    Returns:
        df: pd.DataFrame with every n seconds labelled samples. 
            Uses sql join, hence contains all columns from both df. 
            Unrequired columns should be dropped manually.
    """
    sensor_df['time'] = sensor_df.Time.map(lambda x: pd.to_datetime(x).strftime("%H:%M"))
    label_df['time'] = label_df.Time.map( lambda x: pd.to_datetime(x).strftime("%H:%M"))
    
    merge_df = pd.merge(sensor_df,label_df, on='time')    
    minutes = sorted(list(set(sensor_df['time'])))
    
    df = None
    
    for minute in minutes:
        if not head:
            _temp = merge_df[merge_df.time == minute].tail(n).copy()
        else:
            _temp = merge_df[merge_df.time == minute].head(n).copy()
        
        if not isinstance(df,pd.DataFrame):
            df = _temp.copy()
        else:
            df = pd.concat([df,_temp],axis=0)
    df =  df.reset_index()
    return df
    
x = take_n_sec_samples(reading_df, truth_df, 2, True)
x
sns.heatmap(x.corr())
#select whatever features that you want
_df = x.copy()
_df = _df[['time','22_Temp','22_humidity','21_Temp','21_humidity']].copy()
_df

# df1 = df.groupby('a')['b'].apply(list).reset_index(name='new')
cols = list(_df.columns)
cols.remove('time') # we are gonna group everything else on 'time'

group_cols = []
grouped_df = None

for col in cols:
    c = _df.groupby('time')[col].apply(list).reset_index(name=col)
    if not isinstance(grouped_df,pd.DataFrame):
        grouped_df = c
    else:
        grouped_df = pd.merge(grouped_df, c, on='time')
grouped_df #grouped for every 2 seconds
_df.groupby('time')['22_Temp'].apply(list)
#whatever ground truth you want, i am using number of people
_t = x.drop_duplicates(subset=['time'], keep='first')[['time','number of people']]
pd.merge(grouped_df, _t)
