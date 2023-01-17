!pip install -qq plotly_express
%matplotlib inline

from sklearn.datasets import make_blobs, make_moons, make_swiss_roll

import plotly_express as px

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

blob_data = make_blobs(n_samples=500, 

                                   cluster_std=2.0,

                                   centers=4,

                                   random_state = 2019)[0]

moon_data = make_moons(n_samples=500, 

                                   random_state = 2019)[0]

swiss_data_1 = make_swiss_roll(n_samples=100, noise=0.2, random_state=2019)[0]

swiss_data_2 = -0.5*make_swiss_roll(n_samples=100, noise=0.1, random_state=2020)[0]

swiss_data = np.concatenate([swiss_data_1, swiss_data_2], 0)[:, [0, 2]]

test_pts = pd.DataFrame(swiss_data, columns=['x', 'y'])

plt.plot(test_pts.x, test_pts.y, '.')

test_pts.sample(5)
from sklearn.cluster import KMeans

km = KMeans(n_clusters=4, random_state = 2018)

n_grp = km.fit_predict(test_pts)
plt.scatter(test_pts.x, test_pts.y, c = n_grp)

grp_pts = test_pts.copy()

grp_pts['group'] = n_grp

grp_pts.groupby(['group']).apply(lambda x: x.sample(2))
def calculate_iter(in_pts, iter_count=1):

    c_pts = in_pts.copy()

    c_pts['iter_count'] = iter_count

    init_clusters = np.array([[-10, -10], [-5, -10], [-10, -5], [-5, -10]])

    init_clusters = c_pts[['x', 'y']].values[:4]

    temp_km = KMeans(n_clusters=4, 

                     init=init_clusters,

                     random_state=2019, 

                     n_init=1, 

                     max_iter=iter_count if iter_count>0 else 1)

    if iter_count>0:

        c_pts['group'] = temp_km.fit_predict(c_pts[['x', 'y']])

    else:

        temp_km.fit(init_clusters)

        c_pts['group'] = temp_km.predict(c_pts[['x', 'y']])

        

    c_pts['group'] = c_pts['group'].map(str)

    # calculate the group centers

    grp_center = c_pts.groupby('group').agg('mean').reset_index()

    c_pts['point_type'] = 'points'

    out_df = pd.concat([c_pts.assign(point_type='points'), 

                      grp_center.assign(point_type='centers')], sort=False)

    out_df['index'] = range(out_df.shape[0])

    return out_df
iter_df = pd.concat([calculate_iter(test_pts, i) for i in range(0, 30)]).reset_index(drop=True)

iter_df.head(5)
iter_df['point_size'] = iter_df['point_type'].map(lambda x: 10 if x=='centers' else 1)

px.scatter(iter_df, 

           x='x', y='y', 

           symbol='point_type', 

           animation_frame='iter_count', 

           animation_group='index',

           size='point_size',

           symbol_sequence=['cross', 'circle', 'diamond', 'square', 'x'],

           color='group')