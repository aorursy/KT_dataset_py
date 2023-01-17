# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import umap

from sklearn.cluster import DBSCAN

from sklearn import preprocessing

import bokeh.io

from bokeh.models import ColumnDataSource, Label

from bokeh.plotting import figure, show



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



bokeh.io.output_notebook()

# Any results you write to the current directory are saved as output.
h_ds = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv', delimiter=',')

cols_to_norm = ['age','trestbps','chol','thalach','oldpeak']

h_ds[cols_to_norm] = preprocessing.StandardScaler().fit_transform(h_ds[cols_to_norm])



heart_d = h_ds.loc[:, ['target']]

heart_ds = heart_d.values.tolist()

X = h_ds.drop(['target'], axis=1) # 0 ds 1 NO ds
clusterable_embedding = umap.UMAP(

                n_neighbors=6,  # does seem to work as the same as eps. but color new cluster

                min_dist=0.0,

                n_components=2,

                random_state=42,

            ).fit_transform(X.values)



labels = DBSCAN(

                eps=0.18,

                min_samples=6).fit_predict(clusterable_embedding)
col_ok = []

col_fr = []

xtx = []

ytx = []

xtx_n = []

ytx_n = []

for l in range(len(labels)):

    if heart_ds[l][0] == 1:

        col_ok.append('green')

        xtx.append(clusterable_embedding[l, 0])

        ytx.append(clusterable_embedding[l, 1])

    else:

        col_fr.append('red')

        xtx_n.append(clusterable_embedding[l, 0])

        ytx_n.append(clusterable_embedding[l, 1])



n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# n_noise_ = list(labels).count(-1)



print("Number of clusters {}".format(n_clusters_))
sourcetx = ColumnDataSource(data=dict(xtx=xtx, ytx=ytx, col_ok=col_ok))

source_noise = ColumnDataSource(data=dict(xtx_n=xtx_n, ytx_n=ytx_n, col_fr=col_fr))



ptx = figure(plot_width=850, plot_height=650,

             title="Heart Disease",

             tools="pan,wheel_zoom,box_zoom,reset",

             active_scroll="wheel_zoom",

             toolbar_location="above"

             )
ptx.scatter('xtx', 'ytx', size=3, alpha=0.5, line_dash='solid', color="col_ok", source=sourcetx,

                     legend_label='NO disease')

ptx.scatter('xtx_n', 'ytx_n', size=4, alpha=0.8, line_dash='solid', color='col_fr',

                         source=source_noise, legend_label='Disease') 

ptx.legend.click_policy = "hide"

ptx.legend.background_fill_alpha = 0.4

ptx.legend.location = "bottom_right"



show(ptx)