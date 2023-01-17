import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.offline as plt
data = pd.read_csv("../input/Concrete_Data_Yeh.csv")
data.head()
def plotColumn(X, Y, FileName):
    trace = go.Scatter(y=Y, x=X, mode='markers', marker=dict( size=16,  color = Y, colorscale='Viridis', showscale=True ))
    layout= go.Layout(title= FileName, hovermode= 'closest',
    xaxis= dict( title= FileName, ticklen= 5, zeroline= False, gridwidth= 2, ),
    yaxis=dict( title='csMPa', ticklen=5, gridwidth=2,), showlegend=False )
    FileName = FileName + ".html"
    trace = go.Figure(data=[trace], layout=layout)
    plt.plot(trace, filename=FileName)
    
names = data.columns.unique()
for name in names:
    plotColumn(data[name],data["csMPa"], name)
axis = dict(showline=True, zeroline=False, gridcolor='#fff', ticklen=4)
trace1 = go.Splom(dimensions=[dict(label='cement',
                                 values=data['cement']),
                            dict(label='slag',
                                 values=data['slag']),
                            dict(label='flyash',
                                 values=data['flyash']),
                            dict(label='water',
                                 values=data['water']),
                             dict(label='superplasticizer',
                                 values=data['superplasticizer']),
                             dict(label='coarseaggregate',
                                 values=data['coarseaggregate']),
                             dict(label='fineaggregate',
                                 values=data['fineaggregate']),
                             dict(label='age',
                                 values=data['age']),
                             dict(label='csMPa',
                                 values=data['csMPa'])],
                text=data['csMPa'],
                #default axes name assignment :
                #xaxes= ['x1','x2',  'x3'],
                #yaxes=  ['y1', 'y2', 'y3'], 
                marker=dict(color=data['csMPa'],
                            size=7,
                            colorscale=data['csMPa'],
                            showscale=False,
                            line=dict(width=0.5,
                                      color='rgb(230,230,230)'))
                )
layout = go.Layout(
    title='Cement Data Set',
    dragmode='select',
    #width=600,
    #height=600,
    #autosize=False,
    hovermode='closest',
    plot_bgcolor='rgba(240,240,240, 0.95)',
    xaxis1=dict(axis),
    xaxis2=dict(axis),
    xaxis3=dict(axis),
    xaxis4=dict(axis),
    xaxis5=dict(axis),
    xaxis6=dict(axis),
    xaxis7=dict(axis),
    xaxis8=dict(axis),
    xaxis9=dict(axis),
    
    yaxis1=dict(axis),
    yaxis2=dict(axis),
    yaxis3=dict(axis),
    yaxis4=dict(axis),
    yaxis5=dict(axis),
    yaxis6=dict(axis),
    yaxis7=dict(axis),
    yaxis8=dict(axis),
    yaxis9=dict(axis),
)
trace1['showupperhalf']=False
trace1['diagonal'].update(visible=False)
fig1 = dict(data=[trace1], layout=layout)
plt.plot(fig1, filename='CementData.html')
data['cementWaterRatio'] = data.cement / data.water
data['grainAggerate'] = (data.coarseaggregate + data.fineaggregate) / 2
data.head()
outputmean = data.csMPa.mean()
data = data/data.mean()
from sklearn.model_selection import train_test_split
Y = data.csMPa.copy()
X = data.drop('csMPa', axis=1).copy()

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=43)
n_alphas = 200
alphas = np.arange(0.1, 10, 0.1)

from sklearn import linear_model

coff = list()
traces = []
def modelSelectio(alpha):
    clf = linear_model.Ridge(alpha=alpha, fit_intercept=True)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    coff.append(clf.coef_)
    trace = go.Scatter(x=alphas, y=clf.coef_, mode='lines', name=str(alpha))
    traces.append(trace)
    print("alpha ", alpha, " score is ", score)
    
for alpha in alphas:
    modelSelectio(alpha)
    
plt.plot(traces, filename='RidgeCoffes.html')
