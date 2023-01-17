# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

from datetime import date

import plotly.graph_objects as go

from plotly.subplots import make_subplots





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/nba2k20-player-dataset/nba2k20-full.csv",parse_dates=True)

data
data = data.drop(['college', 'full_name', 'jersey', 'draft_peak'], axis=1)
data[data.isnull().any(axis=1)]
data['draft_round'] = data['draft_round'].apply(lambda x: 0 if x=='Undrafted' else int(x)) 

data['team'] = data['team'].fillna('No team')



data['position'] = data['position'].apply(lambda x: 'F-C' if x=='C-F' else x)

data['position'] = data['position'].apply(lambda x: 'F-G' if x=='G-F' else x)
data['weight'] = [float(data['weight'][i].split()[3]) for i in range(len(data))]

data['height'] = [float(data['height'][i].split()[-1]) for i in range(len(data))]

data['salary'] = [int(data['salary'][i].split('$')[1]) for i in range(len(data))]
data.head()
data.dtypes
data['b_day'] = data['b_day'].apply(lambda x: datetime.strptime(x, '%m/%d/%y').date())

data['age'] = (datetime.today().date() - data['b_day']).astype('<m8[Y]').astype('int64')
data = data.drop(['b_day'], axis=1)
data.head()
fig = make_subplots(

    rows=2, cols=2,

    specs=[[{"type": "box"}, {"type": "box"}],

           [{"type": "box"}, {"type": "box"}]],

)



fig.add_trace(go.Box(y=data['age'], boxpoints='all', name='age'),

              row=1, col=1)



fig.add_trace(go.Box(y=data['weight'], boxpoints='all', name='weight'),

              row=1, col=2)



fig.add_trace(go.Box(y=data['height'], boxpoints='all', name='height'),

              row=2, col=1)



fig.add_trace(go.Box(y=data['rating'], boxpoints='all', name='rating'),

              row=2, col=2)

fig.update_layout(height=1000)

fig.show()

for column in ['weight', 'height']:

    upper_lim = data[column].quantile(.95)

    lower_lim = data[column].quantile(.05)

    data.loc[(data[column] > upper_lim),column] = upper_lim

    data.loc[(data[column] < lower_lim),column] = lower_lim

for column in ['age', 'rating']:

    upper_lim = data[column].quantile(.95)

    lower_lim = data[column].quantile(.05)

    data.loc[(data[column] > upper_lim),column] = int(upper_lim)

    data.loc[(data[column] < lower_lim),column] = int(lower_lim)
for column in ['team', 'country', 'position', 'draft_round']:

    encoded_columns = pd.get_dummies(data[column])

    data = data.join(encoded_columns).drop(column, axis=1)
data