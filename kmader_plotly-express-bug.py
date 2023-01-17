!pip install -qq plotly_express

import plotly_express as px

import plotly

import pandas as pd

print(plotly.__version__, pd.__version__)
%matplotlib inline

from sklearn.datasets import make_blobs, make_moons, make_swiss_roll

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from string import ascii_lowercase

out_pt_list = []

for frame, std in enumerate(np.linspace(0.1, 2, 50)):

    blob_data = make_blobs(n_samples=100, 

                           cluster_std=std,

                           centers=4,

                           random_state=2019)

    test_pts = pd.DataFrame(blob_data[0], columns=['x', 'y'])

    test_pts['group'] = [ascii_lowercase[x] for x in blob_data[1]]

    test_pts['frame'] = frame

    out_pt_list += [test_pts]

out_pt_df = pd.concat(out_pt_list, sort=False).reset_index(drop=True)

out_pt_df.sample(5)
px.scatter(out_pt_df, 

           x='x', y='y',  

           animation_frame='frame', 

           color='group')