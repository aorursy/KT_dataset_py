import pandas as pd
import plotly.express as px
df=pd.read_csv('../input/world-life-expectancy-18002016/indicator-life_expectancy_at_birth.csv')
df.head()
toplot = df.sort_values(by='2016')

px.scatter(toplot, x='2016', y='country', width=800, height=2000)
# YOUR CODE HERE
toplot = toplot.dropna(subset=['2016'])
px.scatter(toplot, x='2016', y='country', width=800, height=2000)
!pip install bar_chart_race
import bar_chart_race as bcr
# YOUR CODE HERE
transposed = df.T
transposed.head()
df_bar = transposed.iloc[1:]
df_bar.columns = transposed.iloc[0]
df_bar.head()
df_bar = df_bar.dropna(axis=1)

for col in df_bar.select_dtypes(include=['object']).columns:
    df_bar[col] = df_bar[col].astype(float)

bcr.bar_chart_race(df=df_bar,n_bars=15,period_length=1000,filter_column_colors=True,fixed_max=True,steps_per_period=20)