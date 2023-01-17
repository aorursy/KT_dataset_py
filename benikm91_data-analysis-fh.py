import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
data = pd.read_csv('/kaggle/input/machine-learning-lab-cas-data-science-fs-20/houses_train.csv', index_col=0)
test_data = pd.read_csv('/kaggle/input/machine-learning-lab-cas-data-science-fs-20/houses_train.csv', index_col=0)
data.head()
plt.subplots(figsize=(20,15))
plt.scatter(data['long'], data['lat'], marker='.')
plt.show()
fig, ax = plt.subplots(figsize=(20,15))
sns.scatterplot(ax=ax, x='long', y='lat', size='price', hue='price', data=data, marker='o')
fig, ax = plt.subplots(figsize=(20,15))
sns.scatterplot(ax=ax, x='long', y='lat', hue='object_type_name', data=data, marker='o')
fig, ax = plt.subplots(figsize=(20,15))
sns.distplot(data['price'], ax=ax, kde=False)
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(20, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
