import gc

import os

import numpy as np

import pandas as pd

import subprocess

import seaborn as sns

import matplotlib.pyplot as plt

import xgboost as xgb
def check_fsize(dpath,s=30):

    """check file size

    Args:

    dpath: file directory

    s: string length in total after padding

    

    Returns:

    None

    """

    for f in os.listdir(dpath):

        print(f.ljust(s) + str(round(os.path.getsize(dpath+'/' + f) / 1000000, 2)) + 'MB')
check_fsize('../input')
def check_fline(fpath):

    """check total number of lines of file for large files

    

    Args:

    fpath: string. file path

    

    Returns:

    None

    

    """

    lines = subprocess.run(['wc', '-l', fpath], stdout=subprocess.PIPE).stdout.decode('utf-8')

    print(lines, end='', flush=True)
fs=['../input/train.csv', '../input/test.csv', '../input/train_sample.csv']

[check_fline(s) for s in fs]
# Load sample training data

df_train = pd.read_csv('../input/train.csv', nrows=1000000, parse_dates=['click_time'])

df_test = pd.read_csv('../input/test.csv', nrows=1000000, parse_dates=['click_time'])



# Show head

print(df_train.head())



# show shape

print(df_test.head())
def check_cunique(df,cols):

    """check unique values for each column

    df: data frame. 

    cols: list. The columns of data frame to be counted

    """

    df_nunique = df[cols].nunique().to_frame()

    df_nunique = df_nunique.reset_index().rename(columns={'index': 'feat',0:'nunique'})

    return df_nunique
df_nunique = check_cunique(df_train,['ip', 'app', 'device', 'os', 'channel'])

df_nunique
plt.figure(figsize=(15, 8))

sns.set(font_scale=1.2)

sns.barplot(x="feat" ,y="nunique", data=df_nunique,log=True)
def feat_value_count(df,colname):

    """value count of each feature

    

    Args

    df: data frame.

    colname: string. Name of to be valued column

    

    Returns

    df_count: data frame.

    """

    df_count = df[colname].value_counts().to_frame().reset_index()

    df_count = df_count.rename(columns={'index':colname+'_values',colname:'counts'})

    return df_count
feat_value_count(df_train,'is_attributed')
def check_missing(df,cols=None,axis=0):

    """check data frame column missing situation

    Args

    df: data frame.

    cols: list. List of column names

    

    Returns

    missing_info: data frame. 

    """

    if cols != None:

        df = df[cols]

    missing_num = df.isnull().sum(axis).to_frame().rename(columns={0:'missing_num'})

    missing_num['minssing_percent'] = df.isnull().mean(axis)*100

    return missing_num.sort_values(by='minssing_percent',ascending = False) 
print(check_missing(df_train))

print(check_missing(df_train,axis=1).head())