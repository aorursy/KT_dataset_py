!pip install bar_chart_race
import pandas as pd

from IPython.display import HTML

import bar_chart_race as bcr
data_path = "/kaggle/input/us-counties-covid-19-dataset/us-counties.csv"

#local: 'data/us-counties-2020-04-28.csv' 



df = pd.read_csv(data_path, index_col='date')

df.head()
df.index = pd.to_datetime(df.index)
df.info()
df_cases = df.loc[:, ['state', 'cases']]
df_states = df_cases.groupby(['date','state']).sum().reset_index()

df_states
df_states = df_states.set_index('date')

df_states.info()
df_states.head()
df_pivoted = df_states.pivot(values='cases', columns='state')

df_pivoted.tail(2)
df_pivoted.info()
df_pivoted_later = df_pivoted[df_pivoted.index >= "2020-02-20"]

df_pivoted_later.head(2)
bcr.bar_chart_race(

    df=df_pivoted_later,

    filename='covid19_county_state_h_later.mp4',

    orientation='h',

    sort='desc',

    label_bars=True,

    use_index=True,

    steps_per_period=10,

    period_length=300,

    figsize=(8, 6),

    cmap='dark24',

    title='COVID-19 Cases by State',

    bar_label_size=7,

    tick_label_size=7,

    period_label_size=16,

)
bcr_html = bcr.bar_chart_race(df=df_pivoted_later, filename=None, period_length=300, figsize=(8, 6))
HTML(bcr_html)
df_pivoted.to_csv('pivoted_covid19_through_apr_27_wf.csv')
df_pivoted_later.to_csv('pivoted_covid19_through_feb_20_to_apr_27_wf.csv')