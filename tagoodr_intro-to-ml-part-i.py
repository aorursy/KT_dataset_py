import numpy



x = numpy.array([1, 2, 3, 4, 5])

y = numpy.array([4, 6, 7, 9, 10])
import matplotlib.pyplot as pyplot

%matplotlib inline



fig1 = pyplot.figure()

axes1 = pyplot.axes(title='Vizualization of the data')

scatter1 = axes1.scatter(x, y)
def Predict(x, w, b):

    return w * x + b 
w = 0

b = 0
yp = Predict(x, w, b)

axes1.plot(x, yp, color='red')

fig1
def Loss(x, y, w, b):

    yp = Predict(x, w, b)

    J = (yp - y)**2      

    loss = numpy.average(J)

    return loss    
ws, bs = numpy.meshgrid(numpy.linspace(0, 5, 20), numpy.linspace(0, 5, 20))

ws, bs = ws.ravel(), bs.ravel()

yp = numpy.outer(ws, x) + bs.reshape(-1, 1)

losses = numpy.average((yp - y)**2, axis=1).ravel()
from plotly.offline import iplot

import plotly.graph_objs as go



trace0 = go.Mesh3d(x=ws, y=bs, z=losses, opacity=0.5)

layout = dict(scene=dict(xaxis=dict(title='w'), yaxis=dict(title='b'), zaxis=dict(title='loss')))

fig2 = go.Figure(data=[trace0], layout=layout)

iplot(fig2)
idx = numpy.where(losses == numpy.amin(losses))

w, b, loss = ws[idx][0], bs[idx][0], losses[idx][0]

print('w:', w, 'b:', b, 'loss:', loss)
trace1 = go.Scatter3d(x=(w,), y=(b,), z=(loss,), marker=dict(size=5, color='cyan'))

fig3 = go.Figure(data=[trace0, trace1], layout=layout)

iplot(fig3)
yp = Predict(x, w, b)

axes1.plot(x, yp, color='cyan')

fig1
def Fit(x, y):

    xavg = numpy.average(x)

    yavg = numpy.average(y)



    xyavg = numpy.average(x * y)

    x2avg = numpy.average(x**2) 



    w = (xyavg - xavg * yavg) / (x2avg - xavg**2)

    b = yavg - w * xavg



    loss = Loss(x, y, w, b)

    

    return w, b, loss
w, b, loss = Fit(x, y)

print('w:', w, 'b:', b, 'loss:', loss)
trace2 = go.Scatter3d(x=(w,), y=(b,), z=(loss,), marker=dict(size=5, color='green'))

fig4 = go.Figure(data=[trace0, trace1, trace2], layout=layout)

iplot(fig4)
yp = Predict(x, w, b)

axes1.plot(x, yp, color='green')

fig1
def Gradient(x, y, w, b):

    yp = Predict(x, w, b)

    dLdw = 2 * numpy.average((yp - y) * x)

    dLdb = 2 * numpy.average(yp - y)    

    return dLdw, dLdb
def GradientDescent(x, y, gradient, alpha, max_steps, goal):

    w = 0

    b = 0

    loss = Loss(x, y, w, b)

    ws=[w]; bs=[b]; losses=[loss]

    for i in range(max_steps):

        dLdw, dLdb = gradient(x, y, w, b)

        w = w - alpha * dLdw

        b = b - alpha * dLdb

        loss = Loss(x, y, w, b)

        ws.append(w); bs.append(b); losses.append(loss)

        if loss < goal:

            break

    return ws, bs, losses
ws, bs, losses = GradientDescent(x, y, Gradient, alpha=0.01, max_steps=10000, goal=0.06)

w, b, loss = ws[-1], bs[-1], losses[-1]

print('w:', w, 'b:', b, 'loss:', loss, 'steps:', len(losses)-1)
trace3 = go.Scatter3d(x=ws, y=bs, z=losses, marker=dict(size=2, color='blue'))

fig5 = go.Figure(data=[trace0, trace1, trace2, trace3], layout=layout)

iplot(fig5)
yp = Predict(x, w, b)

axes1.plot(x, yp, color='blue')

fig1
def NumGradient(x, y, w, b):

    eps = 1E-12         

    loss = Loss(x, y, w, b)

    dLdw = (Loss(x, y, w + eps, b) - loss) / eps

    dLdb = (Loss(x, y, w, b + eps) - loss) / eps

    return dLdw, dLdb
ws, bs, losses = GradientDescent(x, y, NumGradient, alpha=0.01, max_steps=10000, goal=0.06)

w, b, loss = ws[-1], bs[-1], losses[-1]

print('w:', w, 'b:', b, 'loss:', loss, 'steps:', len(losses)-1)
trace4 = go.Scatter3d(x=ws, y=bs, z=losses, marker=dict(size=2, color='purple'))

fig6 = go.Figure(data=[trace0, trace1, trace2, trace3, trace4], layout=layout)

iplot(fig6)
yp = Predict(x, w, b)

axes1.plot(x, yp, color='purple')

fig1