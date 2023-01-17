import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.core.display import display
import pandas as pd
train_data = pd.read_csv("/kaggle/input/machine-learning-lab-cas-data-science-hs-20/train-data.csv", index_col=0)
train_data.shape
train_data.count()
train_data.describe()
# Folgende sind auffällig
display(train_data[['famrel', 'freetime', 'failures', 'absences']].describe())

# A few "unknown" for features about family. We can interpolate them later
train_data[['famsize', 'famsup', 'reason']].count()
f, ax = plt.subplots(figsize=(20, 15))
sns.boxplot(data=train_data)

ax.xaxis.grid(True)
sns.despine(trim=True, left=True)
fig, ax = plt.subplots(figsize=(20,15))
plt.title('G1')
_ = sns.distplot(train_data['G1'], ax=ax, kde=False)
fig, ax = plt.subplots(figsize=(20,15))
plt.title('G2')
_ = sns.distplot(train_data['G2'], ax=ax, kde=False)
fig, ax = plt.subplots(figsize=(20,15))
plt.title('G3')
_ = sns.distplot(train_data['G3'], ax=ax, kde=False)
corr = train_data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(20, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
_ = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
