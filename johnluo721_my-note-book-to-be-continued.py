

# Let us load in the relevant Python modules

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

from collections import Counter

from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings('ignore')

# Load data file 



train = pd.read_csv("../input/train.csv")

train.head()



train.shape

# train.tail

plt.scatter(data=train,x="ps_ind_01", marker="o")
