import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode()
trace1 = go.Scatter3d(
    x=[0,0,2,2],
    y=[0,1,0,1],
    z=[0,0,0,0],
    mode='markers',
    marker=dict(
        size=8,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

u, v = np.mgrid[0:2:5j, 0:1:5j]
w = 0*u*v
trace2 = go.Surface(
    x=u,
    y=v,
    z=w,
        opacity=0.8
    )

X, Y = np.mgrid[0:2:20j, 0:1:20j]
Z = np.sqrt((36 * X**2 * (4 - X**2)))
trace3 = go.Surface(
    x=X,
    y=Y,
    z=Z,
        opacity=0.8
    )


data = [trace1, trace2, trace3]
layout = go.Layout(
                    scene = dict(
                    xaxis = dict(
                        nticks=4, range = [-3,3],),
                    yaxis = dict(
                        nticks=4, range = [-3,3],),
                    aspectratio=dict(x=1, y=1, z=1)
                    ),
                  )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
from sympy import *
from sympy import init_printing
init_printing()
x, y = symbols('x y')
f = sqrt((36 * x**2 * (4-x**2)))
f
integral = integrate(f, (x, 0, 2), (y, 0, 1))
integral
N(integral)
trace1 = go.Scatter3d(
    x=[0,1,2],
    y=[2,1,2],
    z=[0,0,0],
    surfaceaxis=2,
    mode='markers',
    marker=dict(
        size=8,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

X, Y = np.mgrid[0:2:20j, 1:2:20j]
Z = X*Y
trace3 = go.Surface(
    x=X,
    y=Y,
    z=Z,
        opacity=0.8
    )


data = [trace1, trace3]
layout = go.Layout(
                    scene = dict(
                    xaxis = dict(
                        nticks=4, range = [-1,3],),
                    yaxis = dict(
                        nticks=4, range = [0,3],),
                    aspectratio=dict(x=1, y=1, z=1)
                    ),
                  )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
trace1 = go.Scatter3d(
    x=[-1,1],
    y=[-1,1],
    z=[0, 0],
    surfaceaxis=2,
    mode='markers',
    marker=dict(
        size=2,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

u = np.linspace(-2,2,20)
v = u**2
w = 0*u*v
trace2 = go.Scatter3d(
    x=v,
    y=u,
    z=w,
    mode='lines',
    line=dict(color='blue', width=5)
    )

u = np.linspace(-2,2,20)
v = -u
w = 0*u*v
trace3 = go.Scatter3d(
    x=v,
    y=u,
    z=w,
    mode='lines',
    line=dict(color='blue', width=5)
    )

u = np.linspace(-2,2,20)
v = np.ones(len(u))
w = 0*u*v
trace4 = go.Scatter3d(
    x=u,
    y=v,
    z=w,
    mode='lines',
    line=dict(color='blue', width=5)
    )

X, Y = np.mgrid[-1:1:20j, 0:1:20j]
Z = Y*(X+2)
surface = go.Surface(
    x=X,
    y=Y,
    z=Z,
        opacity=0.8
    )


data = [trace1, trace2, trace3, trace4, surface]
layout = go.Layout(
                    scene = dict(
                    xaxis = dict(
                        nticks=4, range = [-2,2],),
                    yaxis = dict(
                        nticks=4, range = [0,2],),
                    aspectratio=dict(x=1, y=1, z=1)
                    ),
                  )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
from sympy import *
from sympy.integrals import Integral
from sympy import init_printing
init_printing()

x, y = symbols('x y')
f = y*(x+2)

integral_f = Integral(f,(x,-y,y**2), (y, 0, 1))
display(Eq(integral_f, integral_f.doit()))
f = y*(x+2)

integral_f = Integral(f,(x,-y,y**2))
display(Eq(integral_f, integral_f.doit()))
from sympy import *
from sympy.integrals import Integral
from sympy import init_printing
init_printing()

x, y = symbols('x y')

f = E**(-y**2 / 2)

integral_f = Integral(f, (y, x, 2),(x,0,2))
display(Eq(integral_f, integral_f.doit()))
f = E**(-y**2 / 2)

integral_f = Integral(f, (x, 0, y),(y,0,2))
display(Eq(integral_f, integral_f.doit()))
f = E**(-y**2 / 2)

integral_f = Integral(f, (y, x, 2),(x,0,2))
display(Eq(integral_f, integral_f.doit()))
from sympy import *
from sympy.integrals import Integral
from sympy import init_printing
init_printing()

x, y = symbols('x y')
f = 1 / log(y)

integral_f = Integral(f,(x,0,2*log(y)), (y, 1, 4))
display(Eq(integral_f, integral_f.doit()))
# 5.2.42 Visualization

import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode()

import numpy as np

trace1 = go.Scatter3d(
    x=[0,0,1,1],
    y=[0,1,0,1],
    z=[0,0,0,0],
    mode='markers',
    marker=dict(
        size=8,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

u, v = np.mgrid[0:1:5j, 0:1:5j]
w = 0*u*v
trace2 = go.Surface(
    x=u,
    y=v,
    z=w,
        opacity=0.8
    )

X, Y = np.mgrid[0:1:20j, 0:1:20j]
Z = 2*X + 3*Y + 6
surface1 = go.Surface(
    x=X,
    y=Y,
    z=Z,
        opacity=0.8
    )

X, Y = np.mgrid[0:1:20j, 0:1:20j]
Z = 2*X + 7*Y + 8
surface2 = go.Surface(
    x=X,
    y=Y,
    z=Z,
    colorscale='Viridis',
        opacity=0.8
    )


data = [trace1, trace2, surface1, surface2]
layout = go.Layout(
                    scene = dict(
                    xaxis = dict(
                        nticks=4, range = [-3,3],),
                    yaxis = dict(
                        nticks=4, range = [-3,3],),
                    aspectratio=dict(x=1, y=1, z=1)
                    ),
                  )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode()

import numpy as np

X, Y = np.mgrid[0:2:20j, 0:2:20j]
Z = 1-X-Y
surface = go.Surface(
    x=X,
    y=Y,
    z=Z,
        opacity=0.8
    )


data = [surface]
layout = go.Layout(
                    scene = dict(
                    xaxis = dict(
                        nticks=4, range = [0,1],),
                    yaxis = dict(
                        nticks=4, range = [0,1],),
                    zaxis = dict(
                        nticks=4, range = [0,1],),
                    aspectratio=dict(x=1, y=1, z=1)
                    ),
                  )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
from sympy import *
from sympy.integrals import Integral
from sympy import init_printing
import numpy as np
init_printing()

x, y = symbols('x y')
f = 1-x-y

# Find I_y:
integral_f = Integral(y*f,(y,0,1-x), (x, 0, 1))
display(Eq(integral_f, integral_f.doit()))

# Find I_y:
integral_f = Integral(x*f,(y,0,1-x), (x, 0, 1))
display(Eq(integral_f, integral_f.doit()))

f = ((1/2)*(1-x-y)**2).expand()
display(f)

# find M if density = z
integral_f = Integral(f,(y,0,1-x), (x, 0, 1))
display(Eq(integral_f, integral_f.doit()))

# find z-bar if desnity = z
f = ((1/3)*(1-x-y)**3).expand()
display(f)
integral_f = Integral(f,(y,0,1-x), (x, 0, 1))
display(Eq(integral_f, integral_f.doit()))
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode()

import numpy as np

X, Y = np.mgrid[-3:3:60j, -3:3:60j]
Z = X**2 + Y**2
surface = go.Surface(
    x=X,
    y=Y,
    z=Z,
    colorscale='Viridis',
        opacity=0.8
    )

u = np.linspace(-2,2,20)
v = u**2
w = np.zeros(len(u))
trace1 = go.Scatter3d(
    x=u,
    y=w,
    z=v,
    mode='lines',
    line=dict(color='blue', width=5)
    )

u = np.linspace(-2,2,20)
v = u**2
w = np.zeros(len(u))
trace2 = go.Scatter3d(
    x=w,
    y=u,
    z=v,
    mode='lines',
    line=dict(color='red', width=5)
    )


data = [trace1, trace2, surface]
layout = go.Layout(
                    scene = dict(
                    xaxis = dict(
                        nticks=4, range = [-3,3],),
                    yaxis = dict(
                        nticks=4, range = [-3,3],),
                    zaxis = dict(
                        nticks=4, range = [0,2],),
                    aspectratio=dict(x=1, y=1, z=1)
                    ),
                  )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
