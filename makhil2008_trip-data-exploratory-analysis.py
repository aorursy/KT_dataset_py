import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import seaborn as sns

import plotly.graph_objs as go

import plotly.plotly as py

sns.set(style="darkgrid")

import matplotlib.pyplot as plt

%matplotlib inline

import os
path2 = r'../input/input_data/trip' # use your path

all_files2 = glob.glob(path2 + "/*.parquet")



li2 = []



for filename in all_files2:

    df2 = pd.read_parquet(filename, engine='auto')

    li2.append(df2)



tripframe = pd.concat(li2, axis=0, ignore_index=True)
tripframe.describe().T
trip_start=tripframe.sort_values(['vehicle_id','trip_id','datetime']).groupby('trip_id').first().reset_index()
import IPython

IPython.display.IFrame("https://www.google.com/maps/d/u/0/embed?mid=16gmX85ju4_-sa2IkT0VALv8eDS1nWKj3" ,width="900", height="350")

trip_end=tripframe.sort_values(['vehicle_id','trip_id','datetime']).groupby('trip_id').last().reset_index()
import IPython

IPython.display.IFrame("https://www.google.com/maps/d/u/0/embed?mid=12lW58NTLQOspTpQYckqejsLNBuDanNxc" ,width="900", height="350")
tripframe.groupby('trip_id').agg({'lat':['mean','var'],'long':['mean','var']})
tripframe['time2']=tripframe.datetime.dt.tz_localize('UTC').dt.tz_convert('America/Los_Angeles').dt.tz_localize(None)

tripframe['week_start']=(tripframe['datetime']- tripframe['datetime'].dt.weekday.astype('timedelta64[D]')).dt.strftime('%Y-%m-%d')

tripframe.groupby('vehicle_id').agg({'week_start':['min','max']})