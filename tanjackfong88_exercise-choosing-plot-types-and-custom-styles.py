import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

import os

if not os.path.exists("../input/spotify.csv"):

    os.symlink("../input/data-for-datavis/spotify.csv", "../input/spotify.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex6 import *

print("Setup Complete")
# Path of the file to read

spotify_filepath = "../input/spotify.csv"



# Read the file into a variable spotify_data

spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)
# Change the style of the figure

sns.set_style("dark")



# Line chart 

plt.figure(figsize=(12,6))

sns.lineplot(data=spotify_data)



# Mark the exercise complete after the code cell is run

step_1.check()
# Darkgrid theme

sns.set_style("darkgrid")

plt.figure(figsize=(10, 5))

sns.lineplot(data=spotify_data)
# Whitegrid theme

sns.set_style("whitegrid")

plt.figure(figsize=(10, 5))

sns.lineplot(data=spotify_data)
# White theme

sns.set_style("white")

plt.figure(figsize=(10, 5))

sns.lineplot(data=spotify_data)
# Ticks theme

sns.set_style("ticks")

plt.figure(figsize=(10, 5))

sns.lineplot(data=spotify_data)