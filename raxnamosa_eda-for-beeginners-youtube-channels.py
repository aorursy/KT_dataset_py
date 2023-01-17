import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
data = pd.read_csv("../input/data.csv")
data.head()
data.info()
data.drop(["Rank", "Grade", "Channel name"],axis=1, inplace = True)
data.columns
data.max()
data.min()
data['Subscribers'] = pd.to_numeric(data['Subscribers'], errors='coerce')

data['Video Uploads'] = pd.to_numeric(data['Video Uploads'], errors='coerce')
data.head()
data.dropna()
data.info()
data.rename(columns={"Video Uploads":"Uploads", "Video views":"Views"}, inplace=True)
data.columns
data.plot(kind='scatter', x='Views', y='Subscribers',alpha = 1,color = 'red')

plt.xlabel('Views')    

plt.ylabel('Subscribers')

plt.title('Video views - Subscribers')
sns.lmplot(x='Subscribers', y='Views', data=data)
rs = np.random.RandomState(33)

d = pd.DataFrame(data=rs.normal(size=(100, 26)))



# Compute the correlation matrix

corr = d.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
#Scatterplot

plt.figure(figsize=(10,6))

sns.lmplot(x='Subscribers', y='Views', data=data, fit_reg=False, hue='Uploads') 
sns.set(style="white")

g = sns.PairGrid(data, diag_sharey=False)

g.map_lower(sns.kdeplot)

g.map_upper(sns.scatterplot)

g.map_diag(sns.kdeplot, lw=3)
sns.jointplot(x=data["Subscribers"], y=data["Views"], kind='scatter')

sns.jointplot(x=data["Subscribers"], y=data["Views"], kind='hex')

sns.jointplot(x=data["Subscribers"], y=data["Views"], kind='kde')
plt.figure(figsize=(15,10))

sns.swarmplot(x='Subscribers', y='Views', data=data, palette='Set2')