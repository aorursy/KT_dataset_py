# import necessary libraries

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
# import data file

dataset = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
# data info

dataset.info()
# first 7 lines of data

dataset.head(7)
# features

dataset.columns
# correlation table

dataset.corr()
# correlation table visulation

f, ax = plt.subplots(figsize = (15, 15))

sns.heatmap(dataset.corr(), annot = True, linewidth = 2, fmt = '.1f', ax = ax)

plt.show()
data_frame = pd.DataFrame(dataset)

data_frame['diagnosis'].value_counts().plot(kind='bar', color='red', alpha=0.5, figsize=(12,7))

plt.title("Bar Plot with Diagnosis Values")

plt.xlabel("Diagnosis Values")

plt.ylabel("Number of Values")

plt.show()
dataset.area_worst.plot(kind='line', color='red', alpha=0.5, label="Area Worst", grid=True, linestyle=":", figsize=(12,7))

dataset.area_mean.plot(kind='line', color='blue', alpha=0.7, label= "Area Mean", grid=True ,linestyle="-")

plt.title("Line Plot with Area")

plt.legend()

plt.show()
data1 = dataset.loc[:, ["area_mean", "area_worst", "radius_mean"]]

data1.plot(figsize=(12, 7), grid=True)

plt.show()
data1.plot(subplots=True, figsize=(12,7))

plt.show()
dataset.plot(kind='scatter', x='area_worst', y='area_mean', grid=True, alpha=0.5, color="green", figsize=(12, 7))

plt.title("Scatter Plot with Area")

plt.xlabel("Area Worst")

plt.ylabel("Area Mean")

plt.show()
data1.plot(kind="scatter", x="area_mean", y="radius_mean", figsize=(12,7), fontsize=15, color="gray", alpha=0.7, grid=True)

plt.title("Scatter Plot between Area Mean and Radius Mean")

plt.show()
data_frame['smoothness_mean'].value_counts().plot(kind='hist', color='red', alpha=0.5, figsize=(12,7))

plt.xlabel("Smoothness Mean")

plt.show()
data1.plot(kind="hist", y="area_mean", bins=50, range=(0,250), density=True, color="red", alpha=0.5, figsize=(12,7), grid=True)

plt.xlabel("Area Mean")

plt.show()
fig, axes = plt.subplots(nrows=2, ncols=1)

data1.plot(kind="hist", y="area_mean", bins=50, range=(0,250), density=True, ax=axes[0], figsize=(12,7))

data1.plot(kind="hist", y="area_mean", bins=50, range=(0,250), density=True, ax=axes[1], cumulative=True)

plt.xlabel("Area Mean")

plt.show()
dataset.boxplot(column=["area_mean", "area_worst"], fontsize=15, figsize=(12,7))

plt.show()
dataset[np.logical_and(dataset["area_mean"] > 200, dataset["area_worst"] < 300)]
threshold = 10.00

dataset["m_radius"] = ["value m" if i>threshold else "value b" for i in dataset.radius_mean]

dataset.loc[:10, ["m_radius", "radius_mean", "diagnosis"]]
print(dataset["area_mean"].value_counts(dropna=False))
assert dataset["area_mean"].notnull().all() # turns nothing because we dont have nan values
dataset.dtypes
data1 = data_frame.head(7)

data2 = data_frame.tail(7)

vertical = pd.concat([data1, data2], axis=0, ignore_index=True)

vertical
data1 = dataset["area_mean"].head(7)

data2 = dataset["radius_mean"].head(7)

horizontal = pd.concat([data1, data2], axis=1)

horizontal
new_data = dataset.head(7)

melted = pd.melt(frame=new_data, id_vars="diagnosis", value_vars=["area_mean", "area_worst"])

melted