# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plotting graphs
plt.close('all')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
df
df['Active'] = df['Confirmed'] - df['Deaths'] - df['Cured']

df
from plotly import __version__
%matplotlib inline
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
init_notebook_mode(connected=True)

init_notebook_mode(connected=True)
cf.go_offline()

def state_df(state_name):
    state_df = df[df['State/UnionTerritory'] == state_name][['Date', 'Deaths', 'Confirmed', 'Active']]
    state_df['Daily Increase(Confirmed)x10'] = (state_df['Confirmed'] - ([0] + list(state_df['Confirmed'])[:-1])) * 10
    state_df['Daily Increase(Active)'] = state_df['Active'] - ([0] + list(state_df['Active'])[:-1])
    state_df['Deathsx10'] = (state_df['Deaths'] - ([0] + list(state_df['Deaths'])[:-1])) * 1000
    return state_df
state_df('Delhi').iplot(kind='line', x='Date')
state_df('Maharashtra').iplot(kind='line', x='Date')
state_df('Karnataka').iplot(kind='line', x='Date')
state_df('Tamil Nadu').iplot(kind='line', x='Date')
state_df('Uttar Pradesh').iplot(kind='line', x='Date')
state_df('Kerala').iplot(kind='line', x='Date')
state_df('Rajasthan').iplot(kind='line', x='Date')
state_df('Madhya Pradesh').iplot(kind='line', x='Date')
state_df('Bihar').iplot(kind='line', x='Date')
state_df('Jammu and Kashmir').iplot(kind='line', x='Date')
state_df('Assam').iplot(kind='line', x='Date')
state_df('Goa').iplot(kind='line', x='Date')
state_df('Jharkhand').iplot(kind='line', x='Date')