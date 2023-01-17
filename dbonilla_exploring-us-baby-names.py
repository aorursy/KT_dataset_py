%matplotlib inline
%%sh
# location of data files
ls /kaggle/input
# imports
%matplotlib inline
import warnings
warnings.filterwarnings("ignore", message="axes.color_cycle is deprecated")
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import sqlite3
# explore sqlite contents
con = sqlite3.connect('../input/database.sqlite')
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())
# helper method to load the data
def load(what='NationalNames'):
    assert what in ('NationalNames', 'StateNames')
    cols = ['Name', 'Year', 'Gender', 'Count']
    if what == 'StateNames':
        cols.append('State')
    df = pd.read_sql_query("SELECT {} from {}".format(','.join(cols), what),
                           con)
    return df
df = load(what='NationalNames')
df.head(5)
df.query('Name == "Andrea"')[['Year', 'Count']].groupby('Year').sum().plot()
df.query('Name == "Daniel"')[['Year', 'Count']].groupby('Year').sum().plot()

df.query('Name == "Tania"')[['Year', 'Count']].groupby('Year').sum().plot()
