# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(rc={'figure.figsize':(20,20)})
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset=pd.read_csv('../input/data.csv',encoding="ISO-8859-1")
dataset.head()
dataset.describe(include='all')
dataset.location.value_counts()
dataset.state.value_counts()
dataset.type.value_counts()
dataset.agency.value_counts()
dataset.location_monitoring_station.value_counts()
dataset.sampling_date.value_counts()
dataset.date.value_counts()
dataset.describe()
so2_levels_by_state = dataset.groupby('state').mean().reset_index()
sns.barplot(x='so2',y='state',data=so2_levels_by_state)
sns.barplot(x='no2',y='state',data=so2_levels_by_state)
sns.barplot(x='rspm',y='state',data=so2_levels_by_state)
sns.barplot(x='spm',y='state',data=so2_levels_by_state)
sns.barplot(x='pm2_5',y='state',data=so2_levels_by_state)