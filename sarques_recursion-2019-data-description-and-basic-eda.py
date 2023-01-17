import numpy as np 

import pandas as pd 

from tqdm import tqdm

import glob

import os

import matplotlib.pyplot as plt



%matplotlib inline
BASE_DIR = '../input'
os.listdir(BASE_DIR)
df_train = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))

print(df_train.sample(5))

print("*"*40)

print(df_train.info())
df_train.nunique()
df_test = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))

print(df_test.sample(5))

print("*"*40)

print(df_test.info())
df_train_controls = pd.read_csv(os.path.join(BASE_DIR, 'train_controls.csv'))

print(df_train_controls.sample(5))

print("*"*40)

print(df_train_controls.info())

print("*"*80)

df_test_controls = pd.read_csv(os.path.join(BASE_DIR, 'test_controls.csv'))

print(df_test_controls.sample(5))

print("*"*40)

print(df_test_controls.info())
print(df_train_controls["well_type"].unique())

print("*"*40)

print(df_test_controls["well_type"].unique())
df_pixel_stats = pd.read_csv(os.path.join(BASE_DIR, 'pixel_stats.csv'))

df_pixel_stats_updated = df_pixel_stats.set_index(['id_code','site', 'channel'])

print(df_pixel_stats.sample(5))

print("*"*40)

print(df_pixel_stats_updated.sample(5))

print("*"*40)

print(df_pixel_stats.info())
df_train["plate"].value_counts().plot(kind = "bar")
plt.rcParams['figure.figsize'] = [15, 5]

df_train["experiment"].value_counts().plot(kind = "bar")
plt.rcParams['figure.figsize'] = [10, 5]

print(df_train_controls["well_type"].value_counts())

df_train_controls["well_type"].value_counts().plot(kind = "bar")
plt.rcParams['figure.figsize'] = [10, 5]

print(df_test_controls["well_type"].value_counts())

df_test_controls["well_type"].value_counts().plot(kind = "bar")
plt.rcParams['figure.figsize'] = [10, 5]

print(df_pixel_stats["site"].value_counts())

df_pixel_stats["site"].value_counts().plot( kind = "bar")
plt.rcParams['figure.figsize'] = [10, 8]

print(df_pixel_stats["channel"].value_counts())

df_pixel_stats["channel"].value_counts().plot(kind = "pie")
path = []

for index in range(6):

    path.append(os.path.join("../input/train/HUVEC-01/Plate1", os.listdir("../input/train/HUVEC-01/Plate1")[index]))
fig, axes = plt.subplots(2, 3, figsize=(24, 16))

for index, ax in enumerate(axes.flatten()):

    img = plt.imread(path[index])

    ax.axis('off')

    ax.set_title(os.listdir("../input/train/HUVEC-01/Plate1")[index])

    _ = ax.imshow(img, cmap='gray')