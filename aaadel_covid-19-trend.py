# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

!pip install tinydb

from tinydb.storages import Storage, JSONStorage
import json

class JSONStorageReadOnly(Storage):
    def __init__(self, filename):
        self.filename = filename

    def read(self):
        with open(self.filename, 'r') as handle:
            try:
                data = json.load(handle)
                return data
            except json.JSONDecodeError:
                return None

    def write(self, data):
        raise RuntimeError('JSONStorageReadOnly cannot write data')

    def close(self):
        pass
    
from tinydb import TinyDB, Query
db = TinyDB('/kaggle/input/covid19-trend/trend.json', storage=JSONStorageReadOnly)

# Any results you write to the current directory are saved as output.
import dateutil.parser

a = np.array([])
t = np.array([])
for row in db:
    a = np.append(a, (row['vc_i'] - row['vp_i']) / row['vp_i'])
    t = np.append(t, dateutil.parser.parse(row['timestamp_dt']))

df = pd.DataFrame(a, t, ['trend'])
df
fig_dims = (18, 4)
fig, ax = plt.subplots(figsize=fig_dims)
ax.set(ylim=(-0.5, 0.5))
fig.add_gridspec(1, 1)

lineplot = sns.lineplot(data=df, ax=ax)
plt.xticks(rotation=45)