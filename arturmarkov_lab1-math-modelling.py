import numpy as np
import plotly.graph_objects as go
from ipywidgets import interact
from plotly.subplots import make_subplots
def lissajous_curve(a1, a2, w1, w2, f):
    t = np.linspace(0,10, 1000)
    x = a1 * np.cos(w1 * t + f)
    y = a2 * np.sin(w2 * t)
    return x,y

fig = go.FigureWidget()
fig.add_scatter()

@interact(a1=(1.0, 10.0, 0.01),
        a2=(1.0, 10.0, 0.01),
        w1=(0, 10.0, 0.01),
        w2=(0, 10.0, 0.01),
        f=(0, 10.0, 0.01)
        )
def update(a1=1,a2=1,w1=1,w2=1,f=1):
    with fig.batch_update():
        fig.data[0].x,fig.data[0].y = lissajous_curve(a1=a1, a2=a2, w1=w1, w2=w2, f=f)
        
fig.update_layout(title_text='Lissajous curve',
                  xaxis_title_text="x", 
                  yaxis_title_text="y",
                  )
fig
def lissajous_curve_3d(a, b, c, n, m, phi, psi):
    t = np.linspace(0,10, 1000)
    x = a * np.sin(t)
    y = b * np.sin(n*t + phi)
    z = c* np.sin(m*t+psi)
    return x,y,z
fig = go.FigureWidget()
fig.add_scatter3d()

@interact(a=(1.0, 10.0, 0.01),
        b=(1.0, 10.0, 0.01),
        c=(0, 10.0, 0.01),
        n=(0, 10.0, 0.01),
        m=(0, 10.0, 0.01),
        phi=(0, 10.0, 0.01),
        psi=(0, 10.0, 0.01)
        )
def update(a=1, b=1, c=1, n=1, m=1, phi=1, psi=1):
    with fig.batch_update():
        fig.data[0].x,fig.data[0].y,fig.data[0].z = lissajous_curve_3d(a, b, c, n, m, phi, psi)
        
fig.update_layout(title_text='Lissajous curve 3d',height=900)
fig
def chebyshev(n,x):
    
    return np.cos(n*np.arccos(x))
fig = go.FigureWidget()
fig.add_scatter()

@interact(n=(1.0, 10.0, 1))
def update(n=1):
    with fig.batch_update():
        fig.data[0].x = np.linspace(-1, 1, 1000)
        fig.data[0].y = chebyshev(n,np.linspace(-1, 1, 1000))        
fig.update_layout(title_text='Chebyshev polynomials',
                  xaxis_title_text="x", 
                  yaxis_title_text="y")
fig
#from variants import d8 as example
def example(p):
    return p**5 + 4*p**3 + p**2 + 2*p + 3
def Michailov_criterion(polynom, w):
    jw = 1j * w
    d = polynom(jw)
    return d.real, d.imag


w = np.linspace(0, 1000, 10000)
x, y = Michailov_criterion(example, w)
# Initialize figure with subplots
fig = make_subplots(
    rows=2, cols=2, subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4")
)

# Add traces
for i in range(1,3):
    for j in range(1,3):
        fig.add_trace(go.Scatter(x=x,y=y), row=i, col=j)

# Update xaxis properties
fig.update_xaxes(range=[0,10], row=1, col=1)
fig.update_xaxes(range=[0,50], row=1, col=2)
#fig.update_xaxes(row=2, col=1)
fig.update_xaxes(type="log", row=2, col=2)

# Update yaxis properties
fig.update_yaxes(range=[-20,10],row=1, col=1)
fig.update_yaxes(range=[-50, 10], row=1, col=2)
#fig.update_yaxes(row=2, col=1)
#fig.update_yaxes(row=2, col=2)

# Update title and height
fig.update_layout(title_text="Hodograph of Mikhailov in different scaling ",width=1000 ,height=1000)

fig
