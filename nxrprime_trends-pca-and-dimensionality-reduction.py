import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import seaborn as sns

import plotly.io as pio



pio.templates.default = "plotly_dark"



import sklearn

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import GroupKFold

from typing import List, Tuple

from plotly.subplots import make_subplots

import plotly.graph_objects as go



fig = make_subplots(rows=1, cols=2)
PATH = '../input/trends-assessment-prediction/'

fnc = pd.read_csv(PATH+'fnc.csv')

ts = pd.read_csv(PATH+'train_scores.csv')

loading = pd.read_csv(PATH+'loading.csv')
id1 = fnc["Id"]

fnc = fnc.drop(["Id"], axis=1)

fnc.head(3)
ts.head(3)
loading.head(3)
import plotly.graph_objects as go



x, y, z = np.array(ts["age"]), np.array(ts["domain1_var1"]), np.array(ts["domain1_var2"])



fig = go.Figure(data=[go.Scatter3d(

    x=x,

    y=y,

    z=z,

    mode='markers',

    marker=dict(

        size=12,

        color=z,                # set color to an array/list of desired values

        colorscale='Viridis',   # choose a colorscale

        opacity=0.8

    )

)])



# tight layout

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.show()
x, y, z = np.array(ts["age"]), np.array(ts["domain2_var1"]), np.array(ts["domain2_var2"])



fig = go.Figure(data=[go.Scatter3d(

    x=x,

    y=y,

    z=z,

    mode='markers',

    marker=dict(

        size=12,

        color=z,                # set color to an array/list of desired values

        colorscale='Viridis',   # choose a colorscale

        opacity=0.8

    )

)])



# tight layout

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.show()
from plotly.subplots import make_subplots



x, y, z = np.array(ts["domain1_var1"]), np.array(ts["domain2_var1"]), np.array(ts["domain2_var2"])

x2, y2, z2 = np.array(ts["domain1_var2"]), np.array(ts["domain2_var1"]), np.array(ts["domain2_var2"])



fig = make_subplots(rows=1, cols=1)



fig.add_trace(go.Scatter3d(

    x=x,

    y=y,

    z=z,

    mode='markers',

    marker=dict(

        size=12,

        color=z,                # set color to an array/list of desired values

        colorscale='Viridis',   # choose a colorscale

        opacity=0.8

    )

))



# tight layout

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.show()
fig = make_subplots(rows=1, cols=1)



fig.add_trace(go.Scatter3d(

    x=x2,

    y=y2,

    z=z2,

    mode='markers',

    marker=dict(

        size=12,

        color=z,                # set color to an array/list of desired values

        colorscale='Viridis',   # choose a colorscale

        opacity=0.8

    ),

))





# tight layout

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.show()
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(fnc)

transformed_pca = pca.fit_transform(fnc)
pd.DataFrame(transformed_pca)
pca.explained_variance_ratio_
from sklearn.decomposition import IncrementalPCA

pca = IncrementalPCA(n_components=5)

pca.fit(fnc)

transformed_pca = pca.fit_transform(fnc)
pd.DataFrame(transformed_pca)
pca.explained_variance_ratio_
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)

svd.fit(fnc)

x = pd.DataFrame(svd.fit_transform(fnc))

x
skewValue = x.skew(axis=1)

skewValue
from sklearn.decomposition import IncrementalPCA

pca = IncrementalPCA(n_components=5)

pca.fit(fnc)

transformed_pca = pca.fit_transform(fnc)

pd.DataFrame(transformed_pca).to_csv('PCAData.csv', index=False)