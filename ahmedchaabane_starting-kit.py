%matplotlib inline

import os

import glob

import numpy as np

from scipy import io

import pandas as pd

import xarray as xr



import altair as alt

from pylab import *
X_ds = pd.read_csv("../input/altered-acrobot-system/Train.csv")

y_array = pd.read_csv("../input/altered-acrobot-system/Test.csv")

list(X_ds)
print((X_ds.thetaDot2[1:].values - X_ds.thetaDot2[:-1].values)[X_ds.action[:-1]==0.0].mean())

print((X_ds.thetaDot2[1:].values - X_ds.thetaDot2[:-1].values)[X_ds.action[:-1]==1.0].mean())

print((X_ds.thetaDot2[1:].values - X_ds.thetaDot2[:-1].values)[X_ds.action[:-1]==2.0].mean())
X_ds.restart.values.sum()
import altair as alt

alt.renderers.enable('notebook')

figs = []

start_time = 1303

end_time = 1680

for observable_col in ['thetaDot2', 'theta2', 'thetaDot1', 'theta1'] :

    ground_truth = X_ds.reset_index().reset_index()

    line_gt = alt.Chart(ground_truth[start_time:end_time]).mark_line(color='black').encode(

        x=alt.X('index:Q', title='time step'),

        y=alt.Y(observable_col + ':Q', scale=alt.Scale(zero=False)),

    )

    fig = line_gt

    figs.append(fig)

alt.vconcat(alt.hconcat(*figs[:2]), alt.hconcat(*figs[2:]))