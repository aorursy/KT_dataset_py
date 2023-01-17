import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import os

import warnings



%matplotlib inline

warnings.filterwarnings("ignore")
# Set the size of the plots 

plt.rcParams["figure.figsize"] = (18,8)

sns.set(rc={'figure.figsize':(18,8)})
data = pd.read_csv("../input/pubg-presentation-features-engineering/train.csv")

print("Finished loading the data")
data = data[['walkDistance', 'swimDistance', 'rideDistance', 'matchDuration', 'kills', 'killPlace', 'maxPlace', 'numGroups', 'winPlacePerc']]
data.head(20)