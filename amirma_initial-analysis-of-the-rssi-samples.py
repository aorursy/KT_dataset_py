%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import pandas as pd

import scipy as sc

sns.set()

import matplotlib

matplotlib.rcParams['figure.figsize'] = [12, 8]
path = '../input/rssi.csv'

data = pd.read_csv(path)
fig = plt.figure(figsize=(8, 20))

from itertools import product



axs = fig.subplots(4,2)

for pair, ax in zip(product((1,2), ("A","B","C","D")), axs.flatten()):

    (floor, ap) = pair

    mask = (data.z == floor) & (data.ap == ap)

    signal = data[mask][["signal", "x", "y"]]

    ax.scatter(signal.x, signal.y, c=signal.signal)

    ax.set_title("Floor: %s AP: %s" %(floor, ap))

    
# Find the Euclidean distance of each sampling location to its respectve AP

ap_coordinates = {"A": (23, 17, 2), "B": (23, 41, 2), "C" : (1, 15, 2), "D": (1, 41, 2)}

g = data.groupby(["x", "y", "z", "ap"])

def dist(df):

    ap_coords = ap_coordinates[df.iloc[0].ap]

    x, y, z = ap_coords

    df["distance"] = np.sqrt((df.x - x) ** 2 + (df.y - y) ** 2 + (df.z - z) ** 2)

    return df

data = g.apply(dist)
fig, axes = plt.subplots(4,2, figsize=(18, 16))

for pair, ax in zip(product((1,2), ("A","B","C","D")), axes.flatten()):

    (floor, ap) = pair

    mask = (data.z == floor) & (data.ap == ap)

    signal = data[mask][["signal", "distance"]]

    ax.plot(signal.distance, signal.signal, '.')

    ax.set_ylabel("RSSI")

    ax.set_title("Floor %s, AP: %s" %(floor, ap))
fig, axes = plt.subplots(2,2, figsize=(18, 16))

estimators = [np.min, np.max, np.mean, np.median]

for ax, estimator in zip(axes.flatten(), estimators):

    mask = (data.z == 2) & (data.ap == "A")

    signal = data[mask][["signal", "distance"]]

    sns.regplot("distance", "signal", data=data, 

                x_estimator=estimator, x_bins=100, ax=ax)

    ax.set_title(estimator.__name__)