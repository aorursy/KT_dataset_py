# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import plotly.graph_objs as go

import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')

df.head()
df['Date'] = df['Date'].apply(pd.to_datetime)

df['Last Update'] = df['Last Update'].apply(pd.to_datetime)

df = df.rename(columns={'Last Update':'Last_Update'})

df.drop(columns='Sno',axis=1,inplace=True)

df.head()
df['ev_month'] = [df.Date[i].month for i in range(df.shape[0])]

df['ev_day']   = [df.Date[i].day   for i in range(df.shape[0])]

df.drop(columns='Date',axis=1,inplace=True)



df['ls_month'] = [df.Last_Update[i].month for i in range(df.shape[0])]

df['ls_day']   = [df.Last_Update[i].day   for i in range(df.shape[0])]

df.drop(columns='Last_Update',axis=1,inplace=True)



df.head()
df.describe(include='all')
df.Country[df.Country=='Mainland China']='China'

df.groupby('Country').sum()[['Confirmed','Deaths','Recovered']]
df_wc = df[df.Country!='China']

g= df_wc.groupby('Country').sum()[['Confirmed','Deaths','Recovered']]



fig = make_subplots(rows=3, cols=1,subplot_titles=("Confirmed", "Deaths", "Recovered"))

fig.add_trace(go.Bar(x=g.index, y=g.Confirmed),row=1, col=1)

fig.add_trace(go.Bar(x=g.index, y=g.Deaths   ),row=2, col=1)

fig.add_trace(go.Bar(x=g.index, y=g.Recovered),row=3, col=1)

fig.update_layout(height=700, width=1000, title_text="Corona Virus Report (Except China)")

fig.show()
g = df[df.Country=='China'].groupby('Province/State').sum()[['Confirmed','Deaths','Recovered']]

fig = make_subplots(rows=3, cols=1,subplot_titles=("Confirmed", "Deaths", "Recovered"))

fig.add_trace(go.Bar(x=g.index, y=g.Confirmed),row=1, col=1)

fig.add_trace(go.Bar(x=g.index, y=g.Deaths   ),row=2, col=1)

fig.add_trace(go.Bar(x=g.index, y=g.Recovered),row=3, col=1)

fig.update_layout(height=800, width=1000, title_text="Corona Virus Report (In States of China)")

fig.show()
g = g[g.Confirmed<max(g.Confirmed)]

fig = make_subplots(rows=3, cols=1,subplot_titles=("Confirmed", "Deaths", "Recovered"))

fig.add_trace(go.Bar(x=g.index, y=g.Confirmed),row=1, col=1)

fig.add_trace(go.Bar(x=g.index, y=g.Deaths   ),row=2, col=1)

fig.add_trace(go.Bar(x=g.index, y=g.Recovered),row=3, col=1)

fig.update_layout(height=700, width=1000, title_text="Corona Virus Report (In States of China)")

fig.show()
print("Granular view for following nations were available\n")

g4 = df[df.Country=='Australia'].groupby('Province/State').sum()[['Confirmed','Deaths','Recovered']]

print("\nStats for Australia\n",'_'*50,'\n',g4)

g4 = df[df.Country=='US'].groupby('Province/State').sum()[['Confirmed','Deaths','Recovered']]

print("\nStats for United States of America\n",'_'*50,'\n',g4)
dft = df[df.Country=='China']

g1 = pd.DataFrame(dft[['Country','ev_day','ev_month','Confirmed']].groupby(['ev_month','ev_day']).sum()['Confirmed'])



a=[i for i in range(g1.shape[0])]



fig = px.bar(x=a, y=g1.Confirmed)

fig.update_layout(height=300, width=800, title_text="Corona Virus (In China)")



fig.update_layout(

    xaxis = dict(

        tickmode = 'array',

        tickvals = [i for i in range(g1.shape[0]+1)],

        ticktext = g1.index

    )

)

fig.show()
dft = df[df.Country!='China']

g2 = pd.DataFrame(dft[['Country','ev_day','ev_month','Confirmed']].groupby(['ev_month','ev_day']).sum()['Confirmed'])



a=[i for i in range(g1.shape[0])]



fig = px.bar(x=a, y=g2.Confirmed)

fig.update_layout(height=300, width=800, title_text="Corona Virus (Rest of the World)")

fig.update_layout(

    xaxis = dict(

        tickmode = 'array',

        tickvals = [i for i in range(g1.shape[0]+1)],

        ticktext = g1.index

    )

)

fig.show()