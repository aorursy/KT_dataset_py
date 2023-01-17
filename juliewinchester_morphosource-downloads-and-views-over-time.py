import pandas as pd

import numpy as np

import plotly.express as px



dl = pd.read_csv('../input/morphosource-monthly-downloads/dl_number_monthly.csv')



dl2 = pd.DataFrame(columns=['time', 'downloads', 'views'])

dl2['time'] = ['-'.join(t.split('_')) for t in list(dl)[:-8]]

dl2['downloads'] = dl.iloc[0].values[:-8]

dl2['downloads'] = dl2['downloads'].astype(int)

dl2['views'] = dl.iloc[1].values[:-8]

dl2['views'] = dl2['views'].astype(int)



dl3 = dl2.loc[dl2.index > 74]
fig = px.line(dl2, x='time', y='downloads')

fig.update_layout(

    title="Downloads Per Month 2013 to Present",

    xaxis_title="Time",

    yaxis_title="Downloads"

)



fig.show()
fig = px.line(dl3, x='time', y='downloads')

fig.update_layout(

    title="Downloads Per Month April 2019 to April 2020",

    xaxis_title="Time",

    yaxis_title="Downloads"

)



fig.show()
fig = px.line(dl2, x='time', y='views')

fig.update_layout(

    title="Views Per Month 2013 to Present",

    xaxis_title="Time",

    yaxis_title="Views"

)



fig.show()
fig = px.line(dl3, x='time', y='views')

fig.update_layout(

    title="Views Per Month April 2019 to April 2020",

    xaxis_title="Time",

    yaxis_title="Views"

)



fig.show()