#22p21c0878_ณัฐภัทร_W2HW3_27092020



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import scipy.cluster.hierarchy as shc
data = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

data = data.sample(n=10000)

data = [[str(row["latitude"]), str(row["longitude"])] for index, row in data.iterrows()]



plt.figure(figsize=(10, 7))

plt.title("Dendograms")

dend = shc.dendrogram(shc.linkage(data, method='complete'))