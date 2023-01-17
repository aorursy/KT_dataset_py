# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import warnings

warnings.filterwarnings("ignore", message="axes.color_cycle is deprecated")

import numpy as np

import pandas as pd

import scipy as sp

import seaborn as sns

import sqlite3
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

df.query('Name=="Alice"')[['Year', 'Count']].groupby('Year').sum().plot()


df2 = load(what='StateNames')

tmp = df2.groupby(['Year', 'State']).agg({'Count': 'sum'}).reset_index()

largest_states = (tmp.groupby('State')

                  .agg({'Count': 'sum'})

                  .sort_values('Count', ascending=False)

                  .index[:5].tolist())

tmp.pivot(index='Year', columns='State', values='Count')[largest_states].plot()