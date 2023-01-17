# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import plotly.express as px
import plotly as py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
url = '../input/corona-virus-outbreak-daily-cases/time_series_covid19_confirmed_global.csv'
df = pd.read_csv(url, delimiter=',', header='infer')
df.head()

df_interest = df.loc[
    df['Country/Region'].isin(['Pakistan', 'US', 'Italy', 'Brazil', 'India'])
    & df['Province/State'].isna()]
df_interest.rename(
    index=lambda x: df_interest.at[x, 'Country/Region'], inplace=True)
df2 = df_interest.transpose()
df2 = df2.drop(['Province/State', 'Country/Region', 'Lat', 'Long'])
df2 = df2.loc[(df2 != 0).any(1)]
df2.index = pd.to_datetime(df2.index)
df2 = df2.diff()     #day on day changes                                                        #28, 131, 137, 223, 225 rows were of selected countries
df2
fig = px.line()
for i,n in enumerate(df2.columns):
    fig.add_scatter(x=df2.index, y= df2[df2.columns[i]], name= df2.columns[i])
    
fig.update_layout(
    title = 'Daily cases of COVID-19'
    ,xaxis_title = 'Dates'
    ,yaxis_title = 'Number of Confirmed cases'
    ,font = dict(size = 20)
    ,template = 'plotly_dark' #"plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"
)

# All plots are interactive, for seeing individual country at a time double click on that country
fig = px.bar(x=df2.index, y= df2[df2.columns[0]])
for i,n in enumerate(df2.columns):
    fig.add_bar(x=df2.index, y= df2[df2.columns[i]], name= df2.columns[i])
fig.update_layout(
    title = 'Daily cases of COVID-19'
    ,xaxis_title = 'Dates'
    ,yaxis_title = 'Number of Confirmed cases'
    ,font = dict(size = 25)
    ,template = 'plotly_dark' #"plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"
)
fig.show()
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.7,
            y=1.03,
            buttons=list([
                dict(label=df2.columns[0],
                     method="update",
                     args=[ {"visible": [False,True, False, False, False, False]},
                            {'showlegend' : True}
                        ]),
                dict(label=df2.columns[1],
                     method="update",
                     args=[ {"visible": [False, False, True, False, False,False]},
                            {'showlegend' : True}
                     ]),
                dict(label=df2.columns[2],
                     method="update",
                     args=[ {"visible": [False, False, False,True, False,False]},
                            {'showlegend' : True}
                        ]),
                dict(label=df2.columns[3],
                     method="update",
                     args=[ {"visible": [False, False, False, False, True, False]},
                            {'showlegend' : True}
                     ]),
                dict(label=df2.columns[4],
                     method="update",
                     args=[ {"visible": [False, False, False, False, False, True,]},
                            {'showlegend' : True}
                           ]),
                dict(label='All',
                     method="update",
                     args=[ {"visible": [True, True, True, True, True, True]},
                            {'showlegend' : True}
                           ]),
            ]),
        )
    ]
)
