# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objs as go



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/cusersmarildownloadsfuneralscsv/funerals.csv', delimiter=';', encoding = "ISO-8859-1")
df.describe()
set(df["cost_recovered"])
pending = df[df['cost_recovered'] == 'Pending']

not_pending = df[df['cost_recovered'] != 'Pending']
fig = go.Figure(data=[go.Pie(labels=['Pending', 'Not Pending'], values=[len(pending), len(not_pending)])])

fig.show()
float_val = []                            # This list with contain the values in float type

for idx, row in not_pending.iterrows():   # We loop on the not_pending DataFrame

    

    ''' There are potentially two char to delete: £ symbol, and the coma between thousands and hundreds.

        The way to do it is to access the value by row['cost_recovered'],

        then we replace those chars by nothing, which is the same as deleting them.

        And we append the results in our correction list, that will become the new "cost_recovered" column later '''

    

    float_val.append(row['cost_recovered'].replace('£','').replace(',',''))

    

not_pending = not_pending.drop('cost_recovered', axis=1)    # We delete the old cost recovered column

not_pending['cost_recovered'] = float_val                   # And replace it by the new one
not_pending.head(5)
float_fun = []

for idx, row in not_pending.iterrows():   # We loop on the not_pending DataFrame

    

    ''' There are potentially two char to delete: £ symbol, and the coma between thousands and hundreds.

        The way to do it is to access the value by row['cost_of_funeral'],

        then we replace those chars by nothing, which is the same as deleting them.

        And we append the results in our correction list, that will become the new "cost_of_funeral" column later '''

    

    float_fun.append(row['cost_of_funeral'].replace('£','').replace(',',''))

    

not_pending = not_pending.drop('cost_of_funeral', axis=1)    # We delete the old cost recovered column

not_pending['cost_of_funeral'] = float_fun                   # And replace it by the new one
not_pending.head(5)
px.scatter(not_pending, x='cost_of_funeral',y='cost_recovered')