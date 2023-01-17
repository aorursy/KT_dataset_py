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
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

train = pd.read_csv('../input/train.tsv', sep='\t')
rows = train.shape[0]

cols = train.shape[1]

print("Number of Rows %d and Columns %d" % (rows, cols))
train.head(10)
train.describe()
sns.countplot(x="item_condition_id", data=train)
sns.countplot(x="shipping", data=train)
sns.distplot(train.price)
train.price_log = np.log(train.price)
sns.distplot(train.price_log[~np.isinf(train.price_log)])
import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
x = train.category_name.value_counts().index[:100]

y = train.category_name.value_counts().values[:100]
trace = go.Bar(

        x = x,

        y = y,

        marker=dict( color=y, colorscale='Viridis', reversescale=True ),

        name = "Category name distribution",

    )

layout = dict(title='Category name distributin', height=1500, width=900)

data = go.Figure(data=[trace], layout=layout)

py.iplot(data, filename='feature-importance-bar')
np.sum(y[:30])