# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

import time



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/lish-moa/train_features.csv").sort_values(by='sig_id')

train_targets = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv").sort_values(by='sig_id')

test = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")
train.head()
print('Rows: ', train.shape[0])

print('Columns: ', train.shape[1] - 1)



print('Rows test: ', test.shape[0])

print('Columns test: ', test.shape[1] - 1)
train_columns = train.columns.to_list()

list_of_g_index = [i for i in train_columns if i.startswith('g-')]

list_of_c_index = [i for i in train_columns if i.startswith('c-')]



plot_list = [list_of_g_index[random.randint(0, len(list_of_g_index)-1)] for i in range(50)]

plot_list = list(set(plot_list))[:12]



plot_list_1 = [list_of_c_index[random.randint(0, len(list_of_c_index)-1)] for i in range(50)]

plot_list_1 = list(set(plot_list_1))[:12]
fig = make_subplots(rows=4, cols=3)

traces = [go.Histogram(x=train[col], nbinsx=20, name=col) for col in plot_list]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 3) + 1, (i % 3) + 1)



fig.update_layout(title_text="Watch on g_index graph", height=1000, width=1000)
fig = make_subplots(rows=4, cols=3)

traces = [go.Histogram(x=train[col], nbinsx=20, name=col) for col in plot_list_1]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 3) + 1, (i % 3) + 1)



fig.update_layout(title_text="Watch on c_index graph", height=1000, width=1000)