import numpy  as np

import pandas as pd



import plotly

import plotly.express    as px

import plotly.graph_objs as go



from plotly.subplots     import make_subplots
x = np.arange(0, 5, 0.1)

def f(x):

    return x**2



px.scatter(x=x, y=f(x)).show()





# More readable way:



# fig = px.scatter(x=x, y=f(x))

# fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=f(x)))

fig.add_trace(go.Scatter(x=x, y=x))

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=f(x),  name='f(x)=x<sup>2</sup>'))

fig.add_trace(go.Scatter(x=x, y=x, name='$$g(x)=x$$'))

fig.update_layout(legend_orientation="h",

                  legend=dict(x=.5, xanchor="center"),

                  title="Line Plot",

                  xaxis_title="x Axis",

                  yaxis_title="y Axis",

                  margin=dict(l=0, r=0, t=30, b=0))

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=f(x), mode='lines+markers',  name='f(x)=x<sup>2</sup>'))

fig.add_trace(go.Scatter(x=x, y=x, mode='markers', name='$$g(x)=x$$'))

fig.update_layout(legend_orientation="h",

                  legend=dict(x=.5, xanchor="center"),

                  margin=dict(l=0, r=0, t=0, b=0))

fig.update_traces(hoverinfo="x+y")

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=f(x), mode='lines+markers',  name='f(x)=x<sup>2</sup>'))

fig.add_trace(go.Scatter(x=x, y=x, mode='markers',name='g(x)=x',

                         marker=dict(color='LightPink', size=7, line=dict(color='MediumPurple', width=2))))

fig.update_layout(legend_orientation="h",

                  legend=dict(x=.5, xanchor="center"),

                  hovermode="x",

                  margin=dict(l=0, r=0, t=0, b=0))

fig.update_traces(hoverinfo="all", hovertemplate="Аргумент: %{x}<br>Функция: %{y}")

fig.show()
fig = go.Figure()

fig.update_yaxes(range=[-0.5, 4], zeroline=True, zerolinewidth=2, zerolinecolor='Yellow')

fig.update_xaxes(range=[-0.5, 4], zeroline=True, zerolinewidth=2, zerolinecolor='#008000')

fig.add_trace(go.Scatter(visible='legendonly', x=x, y=f(x), mode='lines+markers',  name='f(x)=x<sup>2</sup>'))

fig.add_trace(go.Scatter(x=x, y=x, mode='markers',name='g(x)=x',

                         marker=dict(color='LightPink', size=7, line=dict(color='MediumPurple', width=2))))

fig.update_layout(legend_orientation="h",

                  legend=dict(x=.5, xanchor="center"),

                  hovermode="x",

                  margin=dict(l=0, r=0, t=0, b=0))

fig.update_traces(hoverinfo="all", hovertemplate="Arg: %{x}<br>Func: %{y}")

fig.show()
fig = make_subplots(rows=1, cols=2, column_widths=[2, 1])



fig.update_yaxes(range=[-0.5, 4], zeroline=True, zerolinewidth=2, zerolinecolor='Yellow', col=1)

fig.update_xaxes(range=[-0.5, 4], zeroline=True, zerolinewidth=2, zerolinecolor='#008000', col=2)



fig.add_trace(go.Scatter(x=x, y=f(x), mode='lines+markers',  name='f(x)=x<sup>2</sup>'), 1, 1)

fig.add_trace(go.Scatter(x=x, y=x, mode='markers',name='g(x)=x',

                         marker=dict(color='LightPink', size=7, line=dict(color='MediumPurple', width=2))), 1, 2)

fig.update_layout(legend_orientation="h",

                  legend=dict(x=.5, xanchor="center"),

                  hovermode="x",

                  margin=dict(l=0, r=0, t=0, b=0))

fig.update_traces(hoverinfo="all", hovertemplate="Arg: %{x}<br>Func: %{y}")

fig.show()
fig = make_subplots(rows=2, cols=2,

                    specs=[[{"rowspan": 2}, {}], [None, {}]])





fig.update_yaxes(range=[-0.5, 10], zeroline=True, zerolinewidth=2, zerolinecolor='Yellow', col=1)

fig.update_xaxes(range=[-0.5, 5], zeroline=True, zerolinewidth=2, zerolinecolor='#008000', col=2)



fig.add_trace(go.Scatter(x=x, y=x**3,  name='h(x)=x<sup>3</sup>'), 1, 1)

fig.add_trace(go.Scatter(x=x, y=f(x), mode='lines+markers',  name='f(x)=x<sup>2</sup>'), 1, 2)

fig.add_trace(go.Scatter(x=x, y=x, mode='markers',name='g(x)=x',

                         marker=dict(color='LightPink', size=7, line=dict(color='MediumPurple', width=2))), 2, 2)

fig.update_layout(legend_orientation="h",

                  legend=dict(x=.5, xanchor="center"),

                  hovermode="x",

                  margin=dict(l=0, r=0, t=0, b=0))

fig.update_traces(hoverinfo="all", hovertemplate="Arg: %{x}<br>Func: %{y}")

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=f(x), mode='lines+markers',  name='f(x)=x<sup>2</sup>', 

                         marker=dict(color=np.sin(x), 

                                     colorbar=dict(title="h(x)=sin(x)"), 

                                     colorscale='Inferno',

                                     size=50*abs(np.sin(x)))

                        ))



fig.update_layout(legend_orientation="h",

                  legend=dict(x=.5, xanchor="center"),

                  margin=dict(l=0, r=0, t=0, b=0))



fig.update_traces(hoverinfo="all", hovertemplate="Arg: %{x}<br>Func: %{y}")

fig.show()
config = dict({'scrollZoom': True,

               'displayModeBar': True,

               'modeBarButtonsToAdd':['drawline',

                                      'drawopenpath',

                                      'drawclosedpath',

                                      'drawcircle',

                                      'drawrect',

                                      'eraseshape'

                                      ]})



fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=f(x)))

fig.show(config=config)