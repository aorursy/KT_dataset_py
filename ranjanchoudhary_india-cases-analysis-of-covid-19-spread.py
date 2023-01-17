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
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

pd.set_option('display.max_rows', None)
data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data.head()
indiadata=data[data['Country/Region']== 'India']

indiadata.head()
indiadata[["Confirmed","Deaths","Recovered"]] =indiadata[["Confirmed","Deaths","Recovered"]].astype(int)

indiadata.head()
indiadata['Active_case'] = indiadata['Confirmed'] - indiadata['Deaths'] - indiadata['Recovered']

indiadata.head()
fig = go.Figure()

fig.add_trace(go.Scatter(x=indiadata['ObservationDate'], y=indiadata['Confirmed'],

                    mode='lines',

                    name='Confirmed cases'))



fig.add_trace(go.Scatter(x=indiadata['ObservationDate'], y=indiadata['Active_case'],

                    mode='lines',

                    name='Active cases',line=dict( dash='dot')))

fig.add_trace(go.Scatter(x=indiadata['ObservationDate'], y=indiadata['Deaths'],name='Deaths',

                                   marker_color='black',mode='lines',line=dict( dash='dot') ))

fig.add_trace(go.Scatter(x=indiadata['ObservationDate'], y=indiadata['Recovered'],

                    mode='lines',

                    name='Recovered cases',marker_color='green'))

fig.update_layout(

    title='Evolution of cases over time in India',

)



fig.show()