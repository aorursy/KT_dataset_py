import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# set up code checking

import os

if not os.path.exists("../input/spotify.csv"):

    os.symlink("../input/data-for-datavis/spotify.csv", "../input/spotify.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex6 import *

print("Setup Complete")
# load data

spotify_filepath = "../input/spotify.csv"

spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)
# change the style of the figure

sns.set_style("dark")



# line chart 

plt.figure(figsize=(12,6))

sns.lineplot(data=spotify_data)



# mark the exercise complete after the code cell is run

step_1.check()