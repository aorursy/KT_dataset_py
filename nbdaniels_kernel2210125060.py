# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import plotly_express as px

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import plotly.offline as py

py.init_notebook_mode(connected=True)
num = 10

matrix = np.random.choice([x for x in range(0,num)],num*num)

matrix.resize(num,num)

df = pd.DataFrame()

count = 0

for x in range(0,num):

    df[str(count)] = list(matrix[x])

    count+=1

df.head(10)
data = []

for col in range(1,len(df.columns)):

    data.append(go.Scatter(x = df.iloc[:,0],

                    y = df.iloc[:,col],

                    marker= dict(colorscale='Jet',

                                 color = df.iloc[:,1]

                                ),

                    text=df.columns[col],

                    name=df.columns[col]

                ))

layout = go.Layout(title="Test Visualization", hovermode='x')

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='basic-bar')