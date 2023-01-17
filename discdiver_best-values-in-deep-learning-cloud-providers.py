# import the usual frameworks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import warnings

from IPython.core.display import display, HTML
from sklearn.preprocessing import MinMaxScaler
    
# import plotly 
import plotly
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls

# for color scales in plotly
import colorlover as cl 

# configure things
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:,.2f}'.format  
pd.options.display.max_columns = 999

py.init_notebook_mode(connected=True)

%load_ext autoreload
%autoreload 2
%matplotlib inline
# !pwd
# !pip list
df_orig = pd.read_csv(
    '../input/clouds.csv',
    skiprows=0,
    thousands=',',
)
df_orig
df = df_orig.copy()
df
df.info()
df['full'] = df['Cloud Service'] + " " + df['NVIDIA GPU']
df.head(3)
df.set_index('full', inplace = True)
df
df.drop('Kaggle K80', inplace = True)
df
df = df.sort_values(by = ['Cost to Train'])
df.head(5)
cmax=200
cmin=50
color_s = np.linspace(cmin, cmax, 13)

data = [
    go.Bar(
        x=df.index,          
        y=df['Cost to Train'],
        marker=dict(
            colorscale='Jet',
            color=color_s,
        ),
    )
]

layout = {
    'title': 'Cost to Train',
    'xaxis': {'title': 'Cloud Software', 'tickmode': 'linear'},
    'yaxis': {'title': "$ USD"}
}

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df = df.iloc[:-2]
df
labs = [
    'Google Colab K80',
    'vast.ai GTX1070Ti',
    'GCP P100',
    'GCP V100',
    'GCP P4',
    'GCP K80',
    'AWS EC2 V100',
    'Paperspace M4000',
    'GCP V100 x2',
    'AWS EC2 K80',
    'GCP V100 x4',
    'AWS EC2 V100 x4',
]

len(labs)
df['short_names'] = labs
df
cmax=200
cmin=50
color_s = np.linspace(cmin, cmax, 12)

data = [
    go.Bar(
        x=df.short_names,          
        y=df['Cost to Train'],
        marker=dict(
            colorscale='Jet',
            # cauto=True,
            color=color_s,
        ),
    )
]

layout = {
    'title': 'Cost to Train',
    'xaxis': {'title': 'Cloud Software', 'tickmode': 'linear'},
    'yaxis': {'title': "$ USD"}
}

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df = df.sort_values(by = ['Wall Time'])
df
cmax=200
cmin=50
color_s = np.linspace(cmin, cmax, 12)

data = [
    go.Bar(
        x=df.short_names,          
        y=df['Wall Time'],
        marker=dict(
            colorscale='Jet',
            color=color_s,
        ),
    )
]

layout = {
    'title': 'Time to Train',
    'xaxis': {'title': 'Cloud Software', 'tickmode': 'linear'},
    'yaxis': {'title': "Minutes"}
}

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df_best = df.iloc[[0,1,2,4,5,6,11],:]
df_best
df_aws = df[df['Cloud Service']=='AWS EC2']
df_aws
df_gcp = df[df['Cloud Service']=='Google Cloud Compute Engine']
df_gcp
df_vast = df[df['Cloud Service']=='vast.ai']
df_vast
df_colab = df[df['Cloud Service']=='Google Colab']
df_colab
df_paper = df[df['Cloud Service']=='Paperspace']
df_paper
dot_size = 10


trace1 = go.Scatter(
    x=df_aws['Wall Time'], 
    y=df_aws['Cost Per Hour'],
    marker={'color': 'purple', 'size': dot_size}, 
    mode="markers+text",  
    text=df_aws['short_names'],
    name='AWS',
    textposition="top center",
)

# the line

trace2 = go.Scatter(
    x=df_best['Wall Time'], 
    y=df_best['Cost Per Hour'],
    marker={'color': 'yellow',}, 
    mode="lines",  
    name='Efficient Frontier',
)

trace3 = go.Scatter(
    x=df_colab['Wall Time'], 
    y=df_colab['Cost Per Hour'],
    marker={'color': 'red', 'size': dot_size}, 
    mode="markers+text",  
    text=df_colab['short_names'],
    name="Google Colab",
    textposition="top center"
)

trace4 = go.Scatter(
    x=df_paper['Wall Time'], 
    y=df_paper['Cost Per Hour'],
    marker={'color': 'green', 'size': dot_size}, 
    mode="markers+text",  
    text=df_paper['short_names'],
    name="Paperspace",
    textposition="top center"
)

trace5 = go.Scatter(
    x=df_vast['Wall Time'], 
    y=df_vast['Cost Per Hour'],
    marker={'color': 'blue', 'size': dot_size}, 
    mode="markers+text",  
    text=df_vast['short_names'],
    name="vast.ai",
    textposition="top center"
)

trace6 = go.Scatter(
    x=df_gcp['Wall Time'], 
    y=df_gcp['Cost Per Hour'],
    marker={'color': 'black', 'size': dot_size}, 
    mode="markers+text",  
    text=df_gcp.short_names,
    name="Google Cloud",
    textposition="top center"
)

                   
data=[trace1, trace3, trace4, trace5, trace6, trace2 ]
layout = go.Layout(title='Cost per Hour vs Time to Train')

fig = go.FigureWidget(data,layout)

py.iplot(fig)
fig
sns.scatterplot(data=df,x="Wall Time",y="Cost Per Hour")
fig, ax = plt.subplots()

sns.scatterplot(
    data=df,
    x="Wall Time",
    y="Cost Per Hour",
    hue="Cloud Service", 
    legend=False,
    s=100,
)

for item_num in range(0,df.shape[0]):
     ax.text(
         df['Wall Time'][item_num]+0.2, 
         df['Cost Per Hour'][item_num]-0.12, 
         df['short_names'][item_num], 
     )

sns.despine()
fig.set_size_inches(11.7, 8.27)

ax.set_title('Cost per Hour vs Time to Train', fontsize=24)
# ax.legend(loc='upper right', fontsize=16,)
ax.set_xlabel('Time to Train (Minutes)',fontsize=16);
ax.set_ylabel('Cost per Hour ($ USD)',fontsize=16);

import matplotlib.ticker as mtick
fmt = '${x:,.2f}'
tick = mtick.StrMethodFormatter(fmt)
ax.yaxis.set_major_formatter(tick) 


# https://stackoverflow.com/questions/38152356/matplotlib-dollar-sign-with-thousands-comma-tick-labels

fig.savefig('cost_v_time.png')
