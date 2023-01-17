from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler



import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd



import seaborn as sns
pd.set_option('display.max_columns', None)

sns.set(color_codes=True)



# Inline print matplotlib

%matplotlib inline



# Retina display

%config InlineBackend.figure_format="retina"



# Figure size

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [15, 3]
df = pd.read_csv("/kaggle/input/superheroes_nlp_dataset.csv")

df.head(2)
df.columns
superpowers_cols = df.columns[df.columns.str.startswith("has_")]

superpowers_cols[:10]
df[superpowers_cols].sum().sort_values(ascending=False)[:5]
title = "Distribution of genders:"

df['gender'].value_counts().plot.bar();
hulk = df.query("name == 'Hulk'")

hulk_img = "https://www.superherodb.com" + hulk['img'].values[0]



import requests

import IPython.display as Disp

Disp.Image(requests.get(hulk_img).content)