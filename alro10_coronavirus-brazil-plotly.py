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
# Dataset from Brazil

covid = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19.csv')

covid.head()
# Dataset from states

states = pd.read_csv('/kaggle/input/brazilianstates/states.csv')

states.head()
# Merge data from analysis

corona = covid.merge(states, on='state')

corona['potentials'] = corona['population'] * 0.3

corona['potentials'] = corona['potentials'].astype(int)

corona['date'] = pd.to_datetime(corona['date'])

corona.head()
group_uf = corona.groupby('state')

uf = group_uf.tail(1).sort_values('cases', ascending=False).drop(columns=['date']).set_index('state')

uf.style.background_gradient(cmap='Reds', subset=['suspects','refuses','cases','deaths'])
cumulated = corona.groupby('date').sum()

cumulated.head()
# Plot confirmed cases

import plotly.graph_objs as go



fig = go.Figure(

    [

        go.Scatter(x = cumulated.index, y = cumulated['cases'], mode = 'markers+lines', name="Cases Brazil")

    ]

)



fig.update_layout(title='Daily Total Confirmed',

                   plot_bgcolor='rgb(230, 230,230)',

                   showlegend=True, template="plotly_white")



fig.show()
from plotly.subplots import make_subplots



fig = make_subplots(specs=[[{"secondary_y": False}]])



for s in uf.index[:10]:

    data = corona[(corona['state'] == s)]

    fig.add_trace(go.Scatter(y = data['cases'],x = data['date'], mode = 'markers+lines', name = s), secondary_y=False)

                  

fig.update_layout(title='Daily Total Confirmed by State',

                   plot_bgcolor='rgb(230, 230,230)',

                   showlegend=True, template="plotly_white")



fig.show()