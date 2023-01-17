import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
wine_dataset = load_wine()



X = pd.DataFrame(wine_dataset["data"])

X.columns = wine_dataset["feature_names"]

y = pd.DataFrame(wine_dataset["target"])


fig_1, ax = plt.subplots(figsize=(7, 6))

ax.plot([0, 1, 2], [2, 4, 6], c='blue', linestyle='', marker='*',  label='curve1')

ax.plot([0, 1, 2], [3, 6, 9], c='green', linestyle='--', label='curve2')

# position: 90% of the length, 50% of height

ax.legend(loc=(0.9, 0.5))

plt.show()
fig_2, ax = plt.subplots(2, 3, figsize=(7, 6))

fig_2.suptitle("Figure title", fontsize=14, fontweight='bold')

ax[0, 0].set_title("Red")

ax[0, 0].plot([0, 1, 2], [2, 4, 6], marker='o', linestyle='-', c='red')

ax[0, 1].plot([0, 1, 2], [3, 6, 9], marker='*', linestyle='--', c='orange')

ax[0, 2].set_title("grey")

ax[0, 2].plot([0, 1, 2], [3, 6, 9], marker='+', linestyle=':', c='grey')

ax[1, 0].set_title("black")

ax[1, 0].plot([0, 1, 2], [3, 6, 9], marker='^', linestyle='-', c='black')

ax[1, 1].set_title("violet")

ax[1, 1].plot([0, 1, 2], [3, 6, 9], marker='o', linestyle='--', c='violet')

ax[1, 2].set_title("yellow")

ax[1, 2].plot([0, 1, 2], [3, 6, 9], marker='*', linestyle=':', c='yellow')

plt.tight_layout()

plt.show()
fig_3, ax = plt.subplots(figsize=(7, 6))

fig_3.suptitle("Figure title", fontsize=14, fontweight='bold')

ax.scatter(X.alcohol, X.malic_acid, c=wine_dataset["target"], cmap='Set1')

ax.set_xlabel('X name')

ax.set_ylabel('Y name')

ax.set_title("Single visualization title")

plt.show()
colors = np.array(X.alcohol) + np.array(X.malic_acid)

size = np.array(X.alcohol) * np.array(X.malic_acid)

fig_4, ax = plt.subplots(figsize=(7, 6))

ax.scatter(X.alcohol, X.malic_acid,  c=colors, cmap='hsv', s=size)

ax.set_xlabel('Alcohol')

ax.set_ylabel('Malic Acid')

ax.set_title("Wine Dataset")

plt.show()
height = [10, 2, 8]

# x is the position of the bars in x axis

x = [1, 2, 3]

labels = ['Sensor 1', 'Sensor 2', 'Sensor 3']



fig_5, ax = plt.subplots(figsize=(7, 6))

ax.bar(x, height, tick_label=labels)

plt.show()
height_min = [10, 2, 8]

height_max = [8, 6, 5]

x = np.arange(3)

width = 0.4

labels = ['Sensor 1', 'Sensor 2', 'Sensor 3']

fig, ax = plt.subplots(figsize=(7, 6))

ax.bar(x+width/2, height_min, width=width, label='min')

ax.bar(x-width/2, height_max, width=width, label='max')

ax.set_xticks(x) # setup positions of x ticks

ax.set_xticklabels(labels) # set up labels of x ticks

ax.legend(loc=(1.1, 0.5)) # x, y position, in percentage

plt.show()