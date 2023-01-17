import numpy as np 

import pandas as pd 



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score, ShuffleSplit



from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

from plotly import tools



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



init_notebook_mode(connected=True)
# from https://gist.github.com/satra/aa3d19a12b74e9ab7941

from scipy.spatial.distance import pdist, squareform

#from numbapro import jit, float32



def distcorr(X, Y):

    """ Compute the distance correlation function

    

    >>> a = [1,2,3,4,5]

    >>> b = np.array([1,2,9,4,4])

    >>> distcorr(a, b)

    0.762676242417

    """

    X = np.atleast_1d(X)

    Y = np.atleast_1d(Y)

    if np.prod(X.shape) == len(X):

        X = X[:, None]

    if np.prod(Y.shape) == len(Y):

        Y = Y[:, None]

    X = np.atleast_2d(X)

    Y = np.atleast_2d(Y)

    n = X.shape[0]

    if Y.shape[0] != X.shape[0]:

        raise ValueError('Number of samples must match')

    a = squareform(pdist(X))

    b = squareform(pdist(Y))

    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()

    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    

    dcov2_xy = (A * B).sum()/float(n * n)

    dcov2_xx = (A * A).sum()/float(n * n)

    dcov2_yy = (B * B).sum()/float(n * n)

    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

    return dcor
df = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

assert(all(df.isnull().sum()) == False)

assert(all(test.isnull().sum()) == False)
features = list(set(df.columns.tolist()) - set(['formation_energy_ev_natom', 'bandgap_energy_ev', 'id']))

targets = ['formation_energy_ev_natom', 'bandgap_energy_ev']
corr_df = df[features].corr()



data1 = [go.Heatmap(x=features,

                    y=features,

                    z=corr_df.values,

                   showscale = True,

                   zmin=-1, zmax=1)]



layout = dict(title="Pearsons correlation heatmap- Features", 

                xaxis=dict(

        title='Features',

        titlefont=dict(

            size=18,

        ),

        showticklabels=False,

        ticks="",

        tickangle=90,

        tickfont=dict(

            size=4,

        ),

                

    ),

    yaxis=dict(

        title='Features',

        titlefont=dict(

            size=18,

        ),

        showticklabels=False,

        ticks="",

        tickangle=0,

        tickfont=dict(

            size=0,

        ),

    ),

    width = 750, height = 750,

    autosize = False )



figure = dict(data=data1,layout=layout)

iplot(figure)
corr = df[list(set(features)|set(['formation_energy_ev_natom','bandgap_energy_ev']))].corr()



data2 = [go.Heatmap(x=features,

                   y=targets,

                   z=corr.loc[targets, features].values)]



layout2 = dict(title="Correlation heatmap", 

                xaxis=dict(

        title='Features',

        titlefont=dict(

            size=18,

        ),

        showticklabels=False,

        tickangle=90,

        ticks="",

        tickfont=dict(

            size=4,

        ),

                

    ),

    yaxis=dict(

        title='Targets',

        titlefont=dict(

            size=18,

        ),

        showticklabels=False,

        tickangle=0,

        ticks="",

        tickfont=dict(

            size=4,

        ),

    ),

    width = 750, height = 275,

    autosize = False )



figure2 = dict(data=data2,layout=layout2)

iplot(figure2)
features_corr_df = pd.DataFrame(data=np.zeros((len(features),len(features))), columns=features, index=features)



for column in features:

    for feature in features:

        features_corr_df.loc[feature, column] = distcorr(df[feature].values, df[column].values)

    

data2 = [go.Heatmap(x=features_corr_df.columns.tolist(),

                    y=features_corr_df.columns.tolist(),

                    z=features_corr_df.values,

                   showscale = True,

                   colorscale = 'Viridis')]



layout = dict(title="Distance correlation heatmap- Features", 

                xaxis=dict(

        title='Features',

        titlefont=dict(

            size=18,

        ),

        showticklabels=False,

        ticks="",

        tickangle=90,

        tickfont=dict(

            size=4,

        ),

                

    ),

    yaxis=dict(

        title='Features',

        titlefont=dict(

            size=18,

        ),

        showticklabels=False,

        ticks="",

        tickangle=0,

        tickfont=dict(

            size=4,

        ),

    ),

    width = 750, height = 750,

    autosize = False )



figure = dict(data=data2,layout=layout)

iplot(figure)
target_features_corr_df = pd.DataFrame(data=np.zeros((len(targets),len(features))), 

                                       columns=features, 

                                       index=targets)



for feature in features:

    for target in targets:

        target_features_corr_df.loc[target, feature] = distcorr(df[target].values, df[feature].values)

        

data2 = [go.Heatmap(x=features,

                   y=targets,

                   z=target_features_corr_df.values,

                   colorscale = 'Viridis')]



layout2 = dict(title="Target-Features distance correlation heatmap", 

                xaxis=dict(

        title='Features',

        titlefont=dict(

            size=18,

        ),

        showticklabels=False,

        tickangle=90,

        ticks="",

        tickfont=dict(

            size=4,

        ),

                

    ),

    yaxis=dict(

        title='Targets',

        titlefont=dict(

            size=18,

        ),

        showticklabels=False,

        tickangle=0,

        ticks="",

        tickfont=dict(

            size=4,

        ),

    ),

    width = 750, height = 275,

    autosize = False )



figure2 = dict(data=data2,layout=layout2)

iplot(figure2)
modelbased_corr_df = pd.DataFrame(data=np.zeros((len(targets),len(features))), 

                                  columns=features, 

                                  index=targets)



reg = RandomForestRegressor(n_estimators=20, max_depth=4, n_jobs=-1)

for target in targets:

    y = df[target].values

    for feature in features:

        X = df[feature].values

        score = cross_val_score(reg, 

                                X.reshape(-1, 1), 

                                y.ravel(), 

                                scoring='r2', 

                                cv=ShuffleSplit(n_splits= 5, test_size=.2))

        modelbased_corr_df.loc[target, feature] = round(np.mean(score),3)



data2 = [go.Heatmap(x=modelbased_corr_df.columns.tolist(),

                    y=modelbased_corr_df.index.tolist(),

                    z=modelbased_corr_df.values,

                    colorscale = 'Viridis')]



layout2 = dict(title="Target-Features model-based correlation heatmap", 

                xaxis=dict(

        title='Features',

        titlefont=dict(

            size=18,

        ),

        showticklabels=False,

        tickangle=90,

        ticks="",

        tickfont=dict(

            size=4,

        ),

                

    ),

    yaxis=dict(

        title='Targets',

        titlefont=dict(

            size=18,

        ),

        showticklabels=False,

        tickangle=0,

        ticks="",

        tickfont=dict(

            size=4,

        ),

    ),

    width = 750, height = 250,

    autosize = False )



figure2 = dict(data=data2,layout=layout2)

iplot(figure2)
x = df['lattice_vector_1_ang'].values

y = df['lattice_vector_2_ang'].values

z = df['lattice_vector_3_ang'].values





trace1 = go.Scatter3d(

    x=x,

    y=y,

    z=z,

    mode='markers',

    marker=dict(

        size=12,

        color=df['bandgap_energy_ev'].values,                # set color to an array/list of desired values

        colorscale='Jet',   # choose a colorscale

        opacity=0.5

    )

)



data = [trace1]

layout = go.Layout(

    showlegend=True,

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
x = df['lattice_vector_1_ang'].values

y = df['lattice_vector_2_ang'].values

z = df['lattice_vector_3_ang'].values





trace1 = go.Scatter3d(

    x=x,

    y=y,

    z=z,

    mode='markers',

    marker=dict(

        size=12,

        color=df['formation_energy_ev_natom'].values,                # set color to an array/list of desired values

        colorscale='Jet',   # choose a colorscale

        opacity=0.5

    )

)



data = [trace1]

layout = go.Layout(

    showlegend=True,

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)


def makeAxis(title, tickangle): 

    return {

      'title': title,

      'titlefont': { 'size': 20 },

      'tickangle': tickangle,

      'tickfont': { 'size': 15 },

      'tickcolor': 'rgba(0,0,0,0)',

      'ticklen': 5,

      'showline': True,

      'showgrid': True

    }



data = [{ 

    'type': 'scatterternary',

    'mode': 'markers',

    'a': df['percent_atom_al'].values,

    'b': df['percent_atom_in'].values,

    'c': df['percent_atom_ga'].values,

    'hoverinfo': "a+b+c+name+text",

    'hovertext': "A: AL, B: IN, C: GA",

    'text': df['formation_energy_ev_natom'].values,

    'name': "",

    #'text': "a: AL, b: IN, c: GA",

    'marker': {

        'symbol': 100,

        'size': 12,

        'color': df['formation_energy_ev_natom'].values,                # set color to an array/list of desired values

        'colorscale':'Jet',   # choose a colorscale

        'opacity': 0.8,

        'line': { 'width': 2.5 },

        'showscale': True,

        'label': 'formation_energy_ev_natom'

    },

    }]



layout = {

    'ternary': {

        'sum': 1,

        'aaxis': makeAxis('AL at. %', 0),

        'baxis': makeAxis('<br>IN at. %', 45),

        'caxis': makeAxis('<br>GA at. %', -45),

    },

    'annotations': [{

      'showarrow': False,

      'text': 'Formation Energy at different (AL, IN, GA) at. % ',

        'x': 0.5,

        'y': 1.3,

        'font': { 'size': 15 }

    }]

}



fig = {'data': data, 'layout': layout}

iplot(fig, validate=False)
data = [{ 

    'type': 'scatterternary',

    'mode': 'markers',

    'a': df['percent_atom_al'].values,

    'b': df['percent_atom_in'].values,

    'c': df['percent_atom_ga'].values,

#     'hoverinfo': 'text',

#     'hovertext': '',

    'text': "a: AL, b: IN, c: GA",

    'marker': {

        'symbol': 100,

        'size': 12,

        'color': df['bandgap_energy_ev'].values,                # set color to an array/list of desired values

        'colorscale':'Jet',   # choose a colorscale

        'opacity': 0.8,

        'line': { 'width': 2.5 },

        'showscale': True

    },

    }]



layout = {

    'ternary': {

        'sum': 1,

        'aaxis': makeAxis('AL at. %', 0),

        'baxis': makeAxis('<br>IN at. %', 45),

        'caxis': makeAxis('<br>GA at. %', -45),

    },

    'annotations': [{

      'showarrow': False,

      'text': 'Bandgap Energy at different (AL, IN, GA) at. %',

        'x': 0.5,

        'y': 1.3,

        'font': { 'size': 15 }

    }]

}



fig = {'data': data, 'layout': layout}

iplot(fig, validate=False)
data = []

for number_of_total_atoms in df['number_of_total_atoms'].value_counts().index.tolist():

    y0 = df[df['number_of_total_atoms']==number_of_total_atoms]['formation_energy_ev_natom'].values

    data.append(go.Box(y=y0, name=str(number_of_total_atoms), boxpoints = 'suspectedoutliers',boxmean='sd'))

    

    layout = go.Layout(

        title = "Number of total atoms vs. Formation energy",

        yaxis=dict( title = 'Formation energy'),

        xaxis=dict( title = 'Number of total Atoms'))

    

iplot(go.Figure(data=data,layout=layout))
df[df['number_of_total_atoms']==10]['formation_energy_ev_natom']
data = []

for number_of_total_atoms in df['number_of_total_atoms'].value_counts().index.tolist():

    y0 = df[df['number_of_total_atoms']==number_of_total_atoms]['bandgap_energy_ev'].values

    data.append(go.Box(y=y0, name=str(number_of_total_atoms), boxpoints = 'suspectedoutliers',boxmean='sd'))

    layout = go.Layout(

        title = "Number of total atoms vs. Bandgap Energy",

        yaxis=dict( title = 'Bandgap Energy'),

        xaxis=dict( title = 'Number of total Atoms'))

iplot(go.Figure(data=data,layout=layout))
data = []

for spacegroup in df['spacegroup'].value_counts().index.tolist():

    y0 = df[df['spacegroup']==spacegroup]['formation_energy_ev_natom'].values

    data.append(go.Box(y=y0, name=str(spacegroup), boxpoints = 'suspectedoutliers',boxmean='sd'))

    layout = go.Layout(

        title = "Spacegroup vs. Formation Energy",

        yaxis=dict( title = 'Formation Energy'),

        xaxis=dict( title = 'Spacegroup'))

iplot(go.Figure(data=data,layout=layout))    
data = []

for spacegroup in df['spacegroup'].value_counts().index.tolist():

    y0 = df[df['spacegroup']==spacegroup]['bandgap_energy_ev'].values

    #y0 = np.log(y0+1)

    data.append(go.Box(y=y0, name=str(spacegroup), boxpoints = 'suspectedoutliers',boxmean='sd'))

    layout = go.Layout(

        title = "Spacegroup vs. Bandgap Energy",

        yaxis=dict( title = 'Bandgap Energy'),

        xaxis=dict( title = 'Spacegroup'))

iplot(go.Figure(data=data,layout=layout))