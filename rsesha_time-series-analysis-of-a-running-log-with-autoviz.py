import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Install Viz Library
!pip install autoviz
from autoviz.AutoViz_Class import AutoViz_Class
df = pd.read_csv('/kaggle/input/running-log-insight/activity_log.csv')
print(df.shape)
df.head()
df['Date'] = pd.to_datetime(df['Date'])
df.dtypes
time_cols = ['Time','Avg Pace','Best Pace','Best Lap Time']
for time_loc in time_cols:
    df.loc[df[time_loc]=='--',time_loc] = '0.0'
import tqdm
from tqdm import tqdm
for time_col in time_cols:
    df[time_col] = df[time_col].map(lambda x: int(x.split(":")[1])+int(x.split(":")[2])/60 if len(
        x.split(":"))==3 else (int(x.split(":")[0])+int(x.split(":")[1].split(".")[0]) if len(x.split(":"))==2 else int(x.split(".")[0])))
df.dtypes
int_cols = ['Calories','Avg HR','Max HR','Avg Run Cadence','Max Run Cadence','Elev Gain','Elev Loss']
for int_loc in int_cols:
    df.loc[df[int_loc]=='--',int_loc] = '0'
for int_col in int_cols:
    df[int_col] = df[int_col].map(lambda x: int(x) if len(x.split(","))==1 else int(x.split(",")[0]))
df.dtypes
AV = AutoViz_Class()
target = 'Best Lap Time'
dft = AV.AutoViz(filename='', sep=',', depVar=target, dfte=df, header=0, verbose=1)
