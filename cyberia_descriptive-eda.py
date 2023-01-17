from IPython.display import Image

Image(url='https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/3-phase_flow.gif/357px-3-phase_flow.gif')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd
metadata_train_ = pd.read_csv("../input/metadata_train.csv")

metadata_test_ = pd.read_csv("../input/metadata_test.csv")
metadata_test_.head()
metadata_train_.head()
metadata_train_.dtypes
metadata_train_.shape
for col in ['id_measurement', 'phase', 'target']:

    metadata_train_[col] = metadata_train_[col].astype('category')

    

    

for col in ['id_measurement', 'phase']:

    metadata_test_[col] = metadata_test_[col].astype('category')
metadata_train_.dtypes
metadata_test_.dtypes
stats = []

for col in metadata_train_.columns:

    stats.append((col, metadata_train_[col].nunique(), metadata_train_[col].isnull().sum() * 100 / metadata_train_.shape[0], metadata_train_[col].value_counts(normalize=True, dropna=False).values[0] * 100, metadata_train_[col].dtype))

    

stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values', 'Percentage of values in the biggest category', 'type'])

stats_df.sort_values('Percentage of missing values', ascending=False)
import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)
data = [go.Bar(x=stats_df.Feature,

            y=stats_df.Unique_values)]



iplot(data, filename='jupyter-basic_bar')
stats_ = []

for col in metadata_test_.columns:

    stats_.append((col, metadata_test_[col].nunique(), metadata_test_[col].isnull().sum() * 100 / metadata_test_.shape[0], metadata_test_[col].value_counts(normalize=True, dropna=False).values[0] * 100, metadata_test_[col].dtype))

    

stats_df_ = pd.DataFrame(stats_, columns=['Feature', 'Unique_values', 'Percentage of missing values', 'Percentage of values in the biggest category', 'type'])

stats_df_.sort_values('Percentage of missing values', ascending=False)
data_ = [go.Bar(x=stats_df_.Feature,

            y=stats_df_.Unique_values)]



iplot(data_, filename='jupyter-basic_bar')