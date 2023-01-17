import pandas as pd

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf



init_notebook_mode(connected=True)

cf.go_offline()
data = pd.read_csv('../input/data.csv', parse_dates=['Date'])

data = data.loc[data['Region'] == 'global']
data.columns = (data.columns.str.lower()

                .str.replace(' ', '_'))



data.describe(include='all')
track_counts = data['track_name'].value_counts()

track = track_counts.index[-700]

(data.loc[data['track_name'] == track, ('date', 'streams')]

 .set_index('date')

 .iplot(kind='bar',

        yTitle='# Streams',

        title=track))
TOP_N = 200



data = data.sort_values(['artist', 'track_name', 'date'])

data['next_date'] = data.groupby(['artist', 'track_name'])['date'].shift(-1)



data['in_next_day'] = (data['next_date'] - data['date']).dt.days == 1

probabilities = data.groupby('date')['in_next_day'].sum().divide(TOP_N)



probabilities.iplot(title='Conditional Probability')