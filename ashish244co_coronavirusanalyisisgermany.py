l=[16,16,16,16,16,16,16,16,16,16,18,26,48,74,79,130,165,203,262,525,670,800,1040,1224,1565,1966,2714,3621]

import numpy as np

m=np.linspace(1,len(l),len(l))

from scipy.optimize import curve_fit

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.express as px



import numpy as np

def func(t,a,b):

    return a*np.exp(b*t)
popt, pcov = curve_fit(func,  m,  l)
popt
import matplotlib.pyplot as plt
q=np.linspace(1,37,38)
fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x=m,

        y=func(m,*popt),

        mode="markers+lines",

        name="Modeled Curve",

        line=dict(

            color="blue"

        )

    )

)



fig.add_trace(

    go.Scatter(

        x=m,

        y=l,

        mode="markers+lines",

        name="Real",

        line=dict(

            color="red"

        )

    )

)



fig.update_layout(

    title="Germany",

    xaxis_title="Days",

    yaxis_title="Number of Infections",

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="black"

    )

)



fig.show()



#py.iplot([{"x": m, "y": func(m,*popt)}])

#py.iplot([{"x": m, "y": l}])
fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x=q,

        y=func(q,*popt),

        mode="markers+lines",

        name="Curve",

        line=dict(

            color="blue"

        )

    )

)



fig.add_trace(

    go.Scatter(

        x=m,

        y=l,

        mode="markers+lines",

        name="Real",

        line=dict(

            color="black"

        )

    )

)



fig.update_layout(

    title="Germany",

    xaxis_title="Days",

    yaxis_title="Number of Infections",

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="black"

    )

)



fig.show()
res = [(l[i + 1] - l[i]) for i in range(len(m)-1)] 

resnew=[(res[i + 1]/res[i]) for i in range(10,len(res)-1)]
fig = go.Figure()



fig.add_trace(

    go.Scatter(

        x=[i for i in range(10,len(res)-1)],

        y=resnew,

        mode="markers+lines",

        name="Real",

        line=dict(

            color="blue"

        )

    )

)



fig.update_layout(

    title="Germany",

    xaxis_title="Days",

    yaxis_title="Growth factor",

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="black"

    )

)



fig.show()

resnew[-1]
change=[((resnew[i + 1]-resnew[i])/resnew[i]) for i in range(len(resnew)-1)]
fig = go.Figure()



fig.add_trace(

    go.Scatter(

        x=m,

        y=change,

        mode="markers+lines",

        name="Real",

        line=dict(

            color="blue"

        )

    )

)



fig.update_layout(

    title="Germany",

    xaxis_title="Days",

    yaxis_title="Rate of change of growth factor",

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="black"

    )

)



fig.show()



def func2(t):

    return 52000/(1+np.exp(-0.3*(t-37)))
t=np.linspace(1,74,74)

fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x=t,

        y=func2(t),

        mode="markers+lines",

        name="Curve",

        line=dict(

            color="blue"

        )

    )

)



fig.add_trace(

    go.Scatter(

        x=m,

        y=l,

        mode="markers+lines",

        name="Real",

        line=dict(

            color="black"

        )

    )

)



fig.update_layout(

    title="Germany",

    xaxis_title="Days",

    yaxis_title="Number of infections",

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="black"

    )

)





fig.show()



#py.iplot([{"x": m, "y": func(m,*popt)}])

#py.iplot([{"x": m, "y": l}])
import numpy as np

from scipy.integrate import odeint

import matplotlib.pyplot as plt

import pandas as pd

# Total population, N.

def SIRModel(beta):

    N = 82790000



    I0, R0 = 16, 0



    S0 = N - I0 - R0

   

    beta, gamma = beta, 1./14



    t = np.linspace(0, 100, 100)



    def SIRderiv(y, t, N, beta, gamma):

        S, I, R = y

        dSdt = -beta * S * I / N

        dIdt = beta * S * I / N - gamma * I

        dRdt = gamma * I

        return dSdt, dIdt, dRdt



    y0 = S0, I0, R0



    ret = odeint(SIRderiv, y0, t, args=(N, beta, gamma))

    S, I, R = ret.T



    return S,I,R
fig1 = go.Figure()



for step in np.arange(0, 1.1, .1):

    

    fig1.add_trace(

        go.Scatter(

            visible=False,

            line=dict(color="red", width=6),

            name="𝜈 = " + str(step),

            x=np.linspace(1,160,160),

            y=SIRModel(step)[1]))

    

    



# Make 10th trace visible

#fig1.data[10].visible = True



# Create and add slider

steps = []

for i in range(len(fig1.data)):

    step = dict(

        method="restyle",

        args=["visible", [False] * len(fig1.data)],

    )

    step["args"][1][i] = True  # Toggle i'th trace to "visible"

    steps.append(step)



sliders = [dict(

    active=10,

    currentvalue={"prefix": "Beta: "},

    steps=steps

)]



fig1.update_layout(

    sliders=sliders

)



fig1.show()
S,I,R=SIRModel(0.523)

t=np.linspace(1,100,100)

fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x=t,

        y=I/1000,

        mode="markers+lines",

        name="Infections (SIR)",

        line=dict(

            color="red"

        )

    )

)

fig.add_trace(

    go.Scatter(

        x=t,

        y=R/1000,

        mode="markers+lines",

        name="Recovered (SIR)",

        line=dict(

            color="green"

        )

    )

)



fig.add_trace(

    go.Scatter(

        x=t,

        y=S/1000,

        mode="markers+lines",

        name="Susceptible (SIR)",

        line=dict(

            color="black"

        )

    )

)





fig.add_trace(

    go.Scatter(

        x=m,

        y=l,

        mode="markers+lines",

        name="Real Infections",

        line=dict(

            color="blue"

        )

    )

)

fig.update_layout(

    title="Germany",

    xaxis_title="Days",

    yaxis_title="Number of cases",

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="black"

    )

)



fig.show()
i=[3,3,3,3,3,4,21,79,157,229,323,470,655,889,1128,1701,2036,2502,3089,3858,4636,5883,7375,9172,10149,12462,15113,17660]

n=np.linspace(1,len(i),len(i))

popt2, pcov2 = curve_fit(func,  n,  i)

fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x=n,

        y=func(n,*popt2),

        mode="markers+lines",

        name="Curve",

        line=dict(

            color="blue"

        )

    )

)



fig.add_trace(

    go.Scatter(

        x=n,

        y=i,

        mode="markers+lines",

        name="Real",

        line=dict(

            color="red"

        )

    )

)



fig.update_layout(

    title="Italy",

    xaxis_title="Days",

    yaxis_title="Number of Infections",

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="black"

    )

)

fig.show()

fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x=m,

        y=l,

        mode="markers+lines",

        name="Germany",

        line=dict(

            color="Red"

        )

    )

)



fig.add_trace(

    go.Scatter(

        x=n,

        y=i,

        mode="markers+lines",

        name="Italy",

        line=dict(

            color="Green"

        )

    )

)



fig.update_layout(

    title="Italy",

    xaxis_title="Days",

    yaxis_title="Number of Infections",

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="black"

    )

)



fig.show()
from scipy.signal import correlate

fig = go.Figure()

fig.add_trace(

    go.Scatter(

        x=m-8.5,

        y=l,

        mode="markers+lines",

        name="Germany",

        line=dict(

            color="Red"

        )

    )

)



fig.add_trace(

    go.Scatter(

        x=n,

        y=i,

        mode="markers+lines",

        name="Italy",

        line=dict(

            color="Green"

        )

    )

)





fig.update_layout(

    title="Italy vs Germany(time warped)",

    xaxis_title="Days",

    yaxis_title="Number of Infections",

    font=dict(

        family="Courier New, monospace",

        size=14,

        color="black"

    )

)



fig.show()

np.corrcoef(l,i)[0,1]