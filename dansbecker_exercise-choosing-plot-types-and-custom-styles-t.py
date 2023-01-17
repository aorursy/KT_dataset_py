!pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git@master
import sys

sys.path.append('/kaggle/working')
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_easy.ex6 import *

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