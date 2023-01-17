# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_profiling import ProfileReport



import matplotlib.pyplot as plt

from matplotlib import dates as md

import plotly.graph_objs as go

import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

cf.set_config_file(offline=True)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_meta = pd.read_csv('/kaggle/input/building-data-genome-project-v1/meta_open.csv')

df_meta
profile = ProfileReport(df_meta, title="Pandas Profiling Report")
profile
df_powerMeter = pd.read_csv('/kaggle/input/building-data-genome-project-v1/temp_open_utc_complete.csv', index_col='timestamp', parse_dates=True)

df_powerMeter
df_powerMeter['Office_Abbey'].dropna().iplot()
df_weather0 = pd.read_csv('/kaggle/input/building-data-genome-project-v1/weather0.csv', index_col='timestamp', parse_dates=True)

df_weather0
df_plot = df_weather0.select_dtypes(['int', 'float']).copy()

df_plot = df_plot[df_plot>-100]

df_plot.iplot()
df_schedule2 = pd.read_csv('/kaggle/input/building-data-genome-project-v1/schedule2.csv')

df_schedule2