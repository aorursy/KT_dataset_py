# importing libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(font_scale=1.5)

%config InlineBackend.figure_format = 'retina' # high dimension pics

%matplotlib inline

df = pd.read_csv('../input/new-egg-gaming-laptops/new_egg_gaming_laptops.csv')

df.head()
# Basic EDA and cheking prices with different columns

f , ax = plt.subplots(figsize=(15,5))

sns.barplot(data = df , x = 'brand_name' , y= 'price')
f , ax = plt.subplots(figsize=(15,5))

sns.barplot(data = df , x = 'processor' , y= 'price')
f , ax = plt.subplots(figsize=(15,5))

sns.barplot(data = df , x = 'rating' , y= 'price')
f , ax = plt.subplots(figsize=(15,5))

sns.barplot(data = df , x = 'touch' , y= 'price')
f , ax = plt.subplots(figsize=(20,5))

sns.barplot(data = df , x = 'gpu' , y= 'price')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.show()