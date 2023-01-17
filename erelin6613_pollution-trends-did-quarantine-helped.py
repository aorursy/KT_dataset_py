!pip install -q plotly==4.9.0
!pip install -q "notebook>=5.3" "ipywidgets>=7.2"
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots
import plotly.express as px
import seaborn as sns
init_notebook_mode(connected=True)
root_dir = '../input/silkboard-bangalore-ambient-air-covid19lockdown'
files = list(os.listdir(root_dir))
dfs = {}
for file in files:
    f_name = os.path.join(root_dir, file)
    dfs[file.split('_')[0]] = pd.read_csv(f_name)
for name, df in dfs.items():
    print(name)
    print('Columns:', *list(df.columns))
    print('NaNs:', df.isna().sum().sum())
    print('-'*10)
for name, df in dfs.items():
    print('name')
    print('country:', df.country.unique())
    print('city:', df.city.unique())
    print('locations:', df.location.unique())
    print('coordinates:', df.latitude.unique(), df.longitude.unique())
for name, df in dfs.items():
    print(name, '\n', 
          'maximum:', df.value.max(),
          'minimum:', df.value.min(),
          'mean:', df.value.mean(),
          'median:', df.value.median()
         )
to_drop =['country', 'city', 
          'location', 'latitude', 
          'longitude']

for name, df in dfs.items():
    dfs[name] = df.drop(to_drop, axis=1)
def plot_dfs(x, y, title=None, transform=None, vis_type='scatter'):
    if title is None:
        title=f'{x} {y}'
    fig = make_subplots(rows=2, cols=3, subplot_titles=tuple(dfs.keys()))
    fig.update_layout(showlegend=False, title=title)
    i, j = 1, 1
    for df in dfs.values():
        if vis_type == 'scatter':
            if transform is None:
                fig.add_trace(go.Scatter(x=df[x], y=df[y]), row=i, col=j)
            else:
                fig.add_trace(go.Scatter(x=df[x], y=transform(df[y])), row=i, col=j)
        else:
            fig.add_trace(go.Histogram(x=df[x], histnorm='percent'), row=i, col=j)
            fig.update_layout(barmode='stack')
        i+=1
        if i==3:
            j+=1
            i=1
        if j==4:
            j=1
            
    return fig

fig = plot_dfs('utc', 'value', title='Gases concentration values')
iplot(fig)
fig = plot_dfs('utc', 'value', 
               title='Gases concentration percent change',
               transform=pd.Series.pct_change)
iplot(fig)
for df in dfs.values():
    print(df.unit.unique())
for name, df in dfs.items():
    dfs[name] = df[['value', 'utc']].rename(columns={'value': name})
    

gases_df = pd.merge(dfs['co'], dfs['no2'], 
                    on='utc', how='outer')

gases_df = pd.merge(gases_df, dfs['o3'], 
                    on='utc', how='outer')
gases_df = pd.merge(gases_df, dfs['pm10'], 
                    on='utc', how='outer')
gases_df = pd.merge(gases_df, dfs['pm25'], 
                    on='utc', how='outer')
gases_df = pd.merge(gases_df, dfs['so2'], 
                    on='utc', how='outer')
gases_df
gases_df.isna().sum()
df = gases_df.dropna(how='any')
fig = go.Figure()
for gas in dfs.keys():
    fig.add_trace(go.Scatter(x=df.utc,
                             y=df[gas],
                             mode='lines+markers',
                            name=gas))
iplot(fig)
for gas in dfs.keys():
    gases_df[gas+'_pct'] = gases_df[gas].pct_change()
gases_df
df = gases_df.dropna(how='any')
fig = go.Figure()
for gas in dfs.keys():
    fig.add_trace(go.Scatter(x=df.utc,
                             y=df[gas+'_pct'],
                             mode='lines+markers',
                            name=gas))
iplot(fig)
sns.heatmap(gases_df[list(dfs.keys())].corr());