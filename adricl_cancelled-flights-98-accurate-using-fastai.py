# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from fastai.tabular import *
path='../input/flight-delay-prediction/'
outpath='../working/'
df19=pd.read_csv(path + 'Jan_2019_ontime.csv')
df20=pd.read_csv(path + 'Jan_2020_ontime.csv')
df19.head().T
df20.head().T
# Appending the two files together
df = pd.concat([df20, df19])
df.reset_index(drop=True, inplace=True)
df.head()
# removing the pointless columns and removing the columns that are null when the flight is canceled 
df.drop('Unnamed: 21',axis=1,inplace=True)

# continus 'DEP_TIME', 'ARR_TIME',  data not specified if canceled
# cataogrical vals 'DEP_DEL15', 'ARR_DEL15'

df.drop('DEP_TIME',axis=1,inplace=True)
df.drop('ARR_TIME',axis=1,inplace=True)

df.drop('DEP_DEL15',axis=1,inplace=True)
df.drop('ARR_DEL15',axis=1,inplace=True)
binary = lambda a : a > 0
# Making these binary True False as I dont really like the 1, 0
df['CANCELLED'] = df['CANCELLED'].apply(binary)
df['DIVERTED'] = df['DIVERTED'].apply(binary)
# Showing all the retsults
df[df.select_dtypes([bool]).any(1)]
# Setting up the continuous and categorical vars
procs=[FillMissing, Categorify, Normalize]

cat_vars=[ 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_UNIQUE_CARRIER', 'OP_CARRIER_AIRLINE_ID', 'OP_CARRIER', 'TAIL_NUM',
     'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN', 'DEST_AIRPORT_ID', 'DEST_AIRPORT_ID',
    'DEST_AIRPORT_SEQ_ID', 'DEST', 'DIVERTED']

cont_vars = ['DISTANCE', 'OP_CARRIER_FL_NUM']
dep_var = 'CANCELLED'
test = TabularList.from_df(df.iloc[800:2000].copy(), path=outpath, cat_names=cat_vars, cont_names=cont_vars)
data = (TabularList.from_df(df, path=outpath, cat_names=cat_vars, cont_names=cont_vars, procs=procs)
                    .split_by_rand_pct(valid_pct=0.3, seed=42)
                    .label_from_df(cols=dep_var)
                    .add_test(test)
                    .databunch())
data.show_batch(rows=10)
learn = tabular_learner(data, layers=[20,20], metrics=accuracy)
learn.lr_find()
learn.recorder.plot()
print(learn.model)
learn.fit_one_cycle(3, 2e-2)