import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE



from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering, DBSCAN, SpectralClustering



from matplotlib import pyplot as plt, rcParams

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
df = pd.read_csv('../input/movies.csv', encoding='latin1')

df['released'] = pd.to_datetime(df.released)

df.head()
name = df.pop('name')
class OHE(BaseEstimator, TransformerMixin):

    """Perform one-hot encoding of categorical data"""

    def __init__(self, col):

        self.col = col

        

    def fit(self, X, y=None):

        self.cat = X[self.col].astype('category').cat.categories

        return self

    

    def transform(self, X, y=None):

        return pd.get_dummies(X[self.col].astype('category', categories=self.cat))
class Take(BaseEstimator, TransformerMixin):

    """Pass through a single column without modification"""

    def __init__(self, col):

        self.col = col

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        return X[self.col].to_frame(self.col)
class TimeToEpoch(BaseEstimator, TransformerMixin):

    """Convert a datetime column into seconds-since-epoch"""

    def __init__(self, col):

        self.col = col

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        return X[self.col].astype(int).to_frame(self.col)
features = [

    ('budget', Take('budget')),

    ('gross', Take('gross')),

    ('votes', Take('votes')),

    ('year', Take('year')),

    ('genre', OHE('genre')),

    ('rating', OHE('rating')),

    ('time', TimeToEpoch('released'))

]

pipe = Pipeline([

    ('feat', FeatureUnion(features)),

    ('scale', StandardScaler())

])



trans = pipe.fit_transform(df)
# Try to cluster using KMeans for colouring out plot

cluster = KMeans(n_clusters=20)

group_pred = cluster.fit_predict(trans)



# Perform t-SNE to reduce the dimensionality down to 2 dimenions, for easier plotting.

tsne = TSNE(n_components=2)

tsne_fit = tsne.fit_transform(trans)
init_notebook_mode(connected=True)



trace = go.Scatter(

    x=tsne_fit.T[0], 

    y=tsne_fit.T[1],

    mode='markers',

    name='Lines, Markers and Text',

    text=name,

    textposition='top',

    marker=dict(

        color = group_pred, #set color equal to a variable

        colorscale='Portland',

        showscale=True

    )

)



data = [trace]

layout = go.Layout(

    showlegend=False

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)