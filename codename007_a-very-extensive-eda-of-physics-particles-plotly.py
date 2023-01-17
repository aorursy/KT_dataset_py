import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import seaborn
import seaborn as sns

# import plotly and offline mode
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import plotly.figure_factory as ff

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

% matplotlib inline

# # for 3D plotting
# import ipyvolume as ipv
# import ipywidgets as widgets

from trackml.dataset import load_event, load_dataset
from trackml.randomize import shuffle_hits
from trackml.score import score_event
hits, cells, particles, truth = load_event('../input/train_1/event000001005')
detectors = pd.read_csv("../input/detectors.csv")
hits.head()
hits_samples = hits.sample(8000)

fig = ff.create_facet_grid(
    hits_samples,
    x='x',
    y='y',
    color_name='volume_id',
    show_boxes=False,
    marker={'size': 10, 'opacity': 1.0},
    )
iplot(fig, filename='facet - custom colormap')
hits_samples = hits.sample(8000)

fig = ff.create_facet_grid(
    hits_samples,
    x='x',
    y='y',
    facet_col='volume_id',
    color_name='volume_id',
    color_is_cat=True,
#     color_name='volume_id',
#     show_boxes=False,
#     marker={'size': 10, 'opacity': 1.0},
    )
iplot(fig, filename='facet - custom colormap')
hits_samples = hits.sample(8000)
hits_samples.iplot(kind='scatter',x='x',y='y',mode='markers',size=10)
data = [
    go.Histogram2d(
        x=hits.x,
        y=hits.y
    )
]
iplot(data)
hits_sample = hits.sample(8000)
cols_to_use = ['x','y','z']
hits_sample[cols_to_use].iplot(kind='surface',colorscale='rdylbu')
hits_sample = hits.sample(8000)
trace1 = go.Scatter3d(
    x=hits_sample.x,
    y=hits_sample.y,
    z=hits_sample.z,
    mode='markers',
    marker=dict(
        size=6,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=1
    ),
    name = hits_sample.volume_id
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
hits_samples = hits.sample(8000)

fig = ff.create_facet_grid(
    hits_samples,
    x='z',
    y='y',
    color_name='volume_id',
    show_boxes=False,
    marker={'size': 10, 'opacity': 1.0},
    )
iplot(fig, filename='facet - custom colormap')
df = pd.DataFrame({'volume id': hits.volume_id,
                   'Layer id': hits.layer_id,
                   'Module id': hits.module_id})
df.iplot(kind='histogram', subplots=True, shape=(3, 1), filename='cufflinks/histogram-subplots')
hits_samples = hits.sample(8000)

fig = ff.create_scatterplotmatrix(hits_samples[['x','y','z','volume_id']], index='volume_id', size=10, height=800, width=800)
iplot(fig, filename = 'Index a Column')
cells.head()
cells_samples = cells.sample(8000)
fig = ff.create_facet_grid(
    cells_samples,
    x='ch0',
    y='ch1',
#     facet_col='value',
#     color_name='value',
#     color_is_cat=True,
    color_name='value',
    show_boxes=False,
    marker={'size': 10, 'opacity': 1.0},
    )
iplot(fig, filename='facet - custom colormap')
df = pd.DataFrame({'channel identifier(ch0) unique within one module': cells.ch0,
                   'channel identifier(ch1) unique within one module': cells.ch1,
                   'Signal value information': cells.value})
df.iplot(kind='histogram', subplots=True, shape=(3, 1), filename='cufflinks/histogram-subplots')
cells_samples = cells.sample(8000)

fig = ff.create_scatterplotmatrix(cells_samples, index='value', size=10, height=800, width=800)
iplot(fig, filename = 'Index a Column')
particles.head()
particles_samples = particles.sample(10000)

fig = ff.create_facet_grid(
    particles_samples,
    x='vx',
    y='vy',
#     facet_col='nhits',
#     color_name='nhits',
#     color_is_cat=True,
    color_name='nhits',
    show_boxes=False,
    marker={'size': 10, 'opacity': 1.0},
    )
iplot(fig, filename='facet - custom colormap')
particles_samples = particles.sample(10000)

fig = ff.create_facet_grid(
    particles_samples,
    x='vx',
    y='vy',
    facet_col='nhits',
    color_name='nhits',
    color_is_cat=True,
#     color_name='nhits',
#     show_boxes=False,
#     marker={'size': 10, 'opacity': 1.0},
    )
iplot(fig, filename='facet - custom colormap')
temp = particles['q'].value_counts()
labels = temp.index
sizes = (temp / temp.sum())*100
trace = go.Pie(labels=labels, values=sizes, hoverinfo='label+percent')
layout = go.Layout(title='Distribution of particle charges in event000001005')
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# color set

_11_lg = '#95C061'
_12_lb = '#75ABDE'
_13_lo = '#FE9C43'
_14_lbr = '#A6886D'
_15_lr = '#E7675D'
_16_lp = '#7C5674'

color_set = [_11_lg, _12_lb, _13_lo, _14_lbr, _15_lr, _16_lp]
data = []
i = 0
for col in particles_samples['q'].unique():
    data.append(go.Scatter(x=particles_samples[particles_samples['q'] == col]['vx'], y=particles_samples[particles_samples['q'] == col]['vy'], 
                           mode='lines+markers', line=dict(color=color_set[i], width=1, dash='dash'), 
                           marker=dict(color=color_set[i], size=10), name=col))
    i += 1

layout = go.Layout(
    xaxis = dict(
        title = 'Vx',
    ),
    yaxis = dict(
        title = 'vy',
    ),
)
    
fig = go.Figure(data=data, layout=layout)

iplot(fig) 
particles_samples = particles.sample(10000)
data = []
i = 0
for col in particles_samples['q'].unique():
    data.append(go.Scatter(x=particles_samples[particles_samples['q'] == col]['px'], y=particles_samples[particles_samples['q'] == col]['py'], 
                           mode='lines+markers', line=dict(color=color_set[i], width=1, dash='dash'), 
                           marker=dict(color=color_set[i], size=10), name=col))
    i += 1

layout = go.Layout(
    xaxis = dict(
        title = 'Vx',
    ),
    yaxis = dict(
        title = 'vy',
    ),
)
    
fig = go.Figure(data=data, layout=layout)

iplot(fig) 
particles_samples = particles.sample(10000)

trace1 = go.Scatter3d(
    x=particles_samples.px,
    y=particles_samples.py,
    z=particles_samples.pz,
    mode='markers',
    marker=dict(
        size=6,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=1
    ),
    name = 'initial momentum (in GeV/c) along each global axis'
)
trace2 = go.Scatter3d(
    x=particles_samples.vx,
    y=particles_samples.vy,
    z=particles_samples.vz,
    mode='markers',
    marker=dict(
        size=6,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=1
    ),
    name = 'initial position or vertex (in millimeters) in global coordinates.'
)
data = [trace1, trace2]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
particles_samples = particles.sample(10000)

fig = ff.create_facet_grid(
    particles_samples,
    x='vz',
    y='vy',
#     facet_col='nhits',
#     color_name='nhits',
#     color_is_cat=True,
    color_name='nhits',
    show_boxes=False,
    marker={'size': 10, 'opacity': 1.0},
    )
iplot(fig, filename='facet - custom colormap')
particles_samples = particles.sample(10000)

fig = ff.create_facet_grid(
    particles_samples,
    x='vx',
    y='vy',
    facet_row = 'q',
    facet_col = 'nhits',
#     color_name='nhits',
    color_is_cat=True,
#     color_name='nhits',
    show_boxes=False,
    marker={'size': 6, 'opacity': 1.0},
    )
iplot(fig, filename='facet - custom colormap')
particles['nhits'].iplot(kind='histogram',filename='cufflinks/histogram-subplots')
truth.head()
truth_samples = truth.sample(8000)

fig = ff.create_facet_grid(
    truth_samples,
    x='tx',
    y='ty',
    color_name='weight',
    show_boxes=False,
    marker={'size': 10, 'opacity': 1.0},
    )
iplot(fig, filename='facet - custom colormap')
truth_samples = truth.sample(8000)

fig = ff.create_facet_grid(
    truth_samples,
    x='tz',
    y='ty',
    color_name='weight',
    show_boxes=False,
    marker={'size': 10, 'opacity': 1.0, 'color': 'rgb(86, 7, 100)'},
    )
iplot(fig, filename='facet - custom colormap')
truth_samples = truth.sample(8000)

trace2 = go.Scatter3d(
    x=truth_samples.tpx,
    y=truth_samples.tpy,
    z=truth_samples.tpz,
    mode='markers',
    marker=dict(
        size=6,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=1
    ),
)
data = [trace2]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
df = pd.DataFrame({'Hit ID': truth.hit_id,
                   'Particle ID': truth.particle_id,
                   ' per-hit weight used for the scoring metric': truth.weight})
df.iplot(kind='histogram', subplots=True, shape=(3, 1), filename='cufflinks/histogram-subplots')
detectors.head()
detectors['module_t'].iplot(kind='hist', bins=10)
detectors['module_hv'].iplot(kind='hist', bins=10)
data = [
    go.Heatmap(
        z= detectors.corr().values,
        x=detectors.columns.values,
        y=detectors.columns.values,
        colorscale='Viridis',
        reversescale = False,
        text = True ,
        opacity = 1.0 )
]

layout = go.Layout(
    title='Correlation of features',
    xaxis = dict(ticks='', nticks=36),
    yaxis = dict(ticks='' ),
    width = 900, height = 700)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='labelled-heatmap')
df = pd.DataFrame({'Volume ID(of the detector group)': detectors.volume_id,
                   'Layer ID (detector layer inside the group)': detectors.layer_id,
                   'Module ID (detector module inside the layer)': detectors.module_id})
df.iplot(kind='histogram', subplots=True, shape=(3, 1), filename='cufflinks/histogram-subplots')
df = pd.DataFrame({'the size of detector cells along the local u direction(in millimeter)': detectors.pitch_u,
                   'the size of detector cells along the local v direction(in millimeter)': detectors.pitch_v})
df.iplot(kind='histogram', subplots=True, shape=(3, 1), filename='cufflinks/histogram-subplots')