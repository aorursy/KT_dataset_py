import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



### Visualization ###

# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



### Utils ###

import warnings

warnings.filterwarnings("ignore")
movement_cols = ['orientation_X', 'orientation_Y', 'orientation_Z',

                 'orientation_W', 'angular_velocity_X', 'angular_velocity_Y',

                 'angular_velocity_Z', 'linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z']

label = ['surface']



train = pd.read_csv('../input/carrercon-join-of-x-and-y-train/x_y_train/x_y_train.csv')
surface_distribution = (train

                       .groupby('surface')

                       .size()

                       .to_frame('count')

                       .reset_index()

                       .sort_values('count', ascending=True))



data = [go.Bar(

            y=surface_distribution['surface'],

            x=surface_distribution['count'],

            orientation = 'h',

            text=surface_distribution['count']

    )]



layout = go.Layout(

    title='Distribution of surfaces',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
metrics = (train[movement_cols+label]

                   .groupby('surface')

                   .agg(['min','max','mean', 'std'])

              )
surfaces = list(surface_distribution.sort_values('surface').surface)

surfaces
metrics
metrics.columns
def bars(df, col_x, col_y):

    return go.Bar(

            y=df[col_y],

            x=df[col_x],

            orientation = 'h',

            text=df[col_x],

            name=col_x

    )
metrics_name = ['std', 'mean', 'max', 'min']
orientation_x = metrics.orientation_X

orientation_x['surface'] = surfaces



data = []



for m in metrics_name:

    data.append(bars(orientation_x, m, 'surface'))



layout = go.Layout(

    title='Orientation X',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
orientation_y = metrics.orientation_Y

orientation_y['surface'] = surfaces



data = []



for m in metrics_name:

    data.append(bars(orientation_y, m, 'surface'))



layout = go.Layout(

    title='Orientation Y',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
orientation_Z = metrics.orientation_Z

orientation_Z['surface'] = surfaces



data = []



for m in metrics_name:

    data.append(bars(orientation_Z, m, 'surface'))



layout = go.Layout(

    title='Orientation Z',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
orientation_W = metrics.orientation_W

orientation_W['surface'] = surfaces



data = []



for m in metrics_name:

    data.append(bars(orientation_W, m, 'surface'))



layout = go.Layout(

    title='Orientation W',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
angular_velocity_X = metrics.angular_velocity_X

angular_velocity_X['surface'] = surfaces



data = []



for m in metrics_name:

    data.append(bars(angular_velocity_X, m, 'surface'))



layout = go.Layout(

    title='Angular velocity X',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
angular_velocity_Y = metrics.angular_velocity_Y

angular_velocity_Y['surface'] = surfaces



data = []



for m in metrics_name:

    data.append(bars(angular_velocity_Y, m, 'surface'))



layout = go.Layout(

    title='Angular velocity Y',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
angular_velocity_Z = metrics.angular_velocity_Z

angular_velocity_Z['surface'] = surfaces



data = []



for m in metrics_name:

    data.append(bars(angular_velocity_Z, m, 'surface'))



layout = go.Layout(

    title='Angular velocity Z',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
linear_acceleration_X = metrics.linear_acceleration_X

linear_acceleration_X['surface'] = surfaces



data = []



for m in metrics_name:

    data.append(bars(linear_acceleration_X, m, 'surface'))



layout = go.Layout(

    title='Linear Acceleration X',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
linear_acceleration_Y = metrics.linear_acceleration_Y

linear_acceleration_Y['surface'] = surfaces



data = []



for m in metrics_name:

    data.append(bars(linear_acceleration_Y, m, 'surface'))



layout = go.Layout(

    title='Linear Acceleration Y',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
linear_acceleration_Z = metrics.linear_acceleration_Z

linear_acceleration_Z['surface'] = surfaces



data = []



for m in metrics_name:

    data.append(bars(linear_acceleration_Z, m, 'surface'))



layout = go.Layout(

    title='Linear Acceleration Z',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
new_cat = {'carpet': 'bad', 'concrete': 'good', 'fine_concrete': 'good',

           'hard_tiles': 'good', 'hard_tiles_large_space': 'bad','soft_pvc': 'medium',

           'soft_tiles': 'medium', 'tiled': 'bad', 'wood': 'good'}
train['type_surface'] = train['surface'].map(new_cat)
train.head()
metrics_new_cat = (train[movement_cols+['type_surface']]

                   .groupby('type_surface')

                   .agg(['min','max','mean', 'std'])

              )
type_surface = ['bad', 'good', 'medium']
metrics_new_cat
orientation_x = metrics_new_cat.orientation_X

orientation_x['type_surface'] = type_surface



data = []



for m in metrics_name:

    data.append(bars(orientation_x, m, 'type_surface'))



layout = go.Layout(

    title='Orientation X',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
orientation_Y = metrics_new_cat.orientation_Y

orientation_Y['type_surface'] = type_surface



data = []



for m in metrics_name:

    data.append(bars(orientation_Y, m, 'type_surface'))



layout = go.Layout(

    title='Orientation Y',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
orientation_Z = metrics_new_cat.orientation_Z

orientation_Z['type_surface'] = type_surface



data = []



for m in metrics_name:

    data.append(bars(orientation_Z, m, 'type_surface'))



layout = go.Layout(

    title='Orientation Z',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
orientation_W = metrics_new_cat.orientation_W

orientation_W['type_surface'] = type_surface



data = []



for m in metrics_name:

    data.append(bars(orientation_W, m, 'type_surface'))



layout = go.Layout(

    title='Orientation W',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
angular_velocity_X = metrics_new_cat.angular_velocity_X

angular_velocity_X['type_surface'] = type_surface



data = []



for m in metrics_name:

    data.append(bars(angular_velocity_X, m, 'type_surface'))



layout = go.Layout(

    title='Angular velocity X',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
angular_velocity_Y = metrics_new_cat.angular_velocity_Y

angular_velocity_Y['type_surface'] = type_surface



data = []



for m in metrics_name:

    data.append(bars(angular_velocity_Y, m, 'type_surface'))



layout = go.Layout(

    title='Angular velocity Y',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
angular_velocity_Z = metrics_new_cat.angular_velocity_Z

angular_velocity_Z['type_surface'] = type_surface



data = []



for m in metrics_name:

    data.append(bars(angular_velocity_Z, m, 'type_surface'))



layout = go.Layout(

    title='Angular velocity Z',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
linear_acceleration_X = metrics_new_cat.linear_acceleration_X

linear_acceleration_X['type_surface'] = type_surface



data = []



for m in metrics_name:

    data.append(bars(linear_acceleration_X, m, 'type_surface'))



layout = go.Layout(

    title='Linear Acceleration X',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
linear_acceleration_Y = metrics_new_cat.linear_acceleration_Y

linear_acceleration_Y['type_surface'] = type_surface



data = []



for m in metrics_name:

    data.append(bars(linear_acceleration_Y, m, 'type_surface'))



layout = go.Layout(

    title='Linear Acceleration Y',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)
linear_acceleration_Z = metrics_new_cat.linear_acceleration_Y

linear_acceleration_Z['type_surface'] = type_surface



data = []



for m in metrics_name:

    data.append(bars(linear_acceleration_Z, m, 'type_surface'))



layout = go.Layout(

    title='Linear Acceleration Z',

    titlefont=dict(size=20),

    width=800,

    height=600

)



fig = go.Figure(data=data, layout=layout)



iplot(fig)