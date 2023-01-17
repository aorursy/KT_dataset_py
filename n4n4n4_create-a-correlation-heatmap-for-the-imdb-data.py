import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



df = pd.read_csv("../input/movie_metadata.csv")

df.head()
sns.heatmap(df.corr(), cmap="coolwarm")