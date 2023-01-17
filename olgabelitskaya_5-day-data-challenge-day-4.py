import numpy as np 

import pandas as pd 

import seaborn as sns

import pylab as plt

import scipy.stats 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
cereal = pd.read_csv("../input/cereal.csv")

cereal.head().T
print(list(cereal))
cereal['mfr'].dtype
set(cereal['mfr'].values)
# Plot #1

sns.set(style='whitegrid')

plt.figure(figsize=(15,5))

sns.countplot(x='mfr', data=cereal, 

              facecolor=(0, 0, 0, 0), linewidth=7,

              edgecolor=sns.color_palette("Set1", 7))

plt.title('Cereal by Manufacturers', fontsize=20);
plt.figure(figsize=(15,5))

sns.countplot(x="mfr", hue="fat", data=cereal, palette='Set1')

plt.title('Cereal by Manufacturers Grouped by Fat Content', fontsize=20);
# Extract the color list for the palette 'Set1'

import matplotlib

cmap = plt.get_cmap('Set1', 9)    



for i in range(cmap.N):

    rgb = cmap(i)[:3] 

    print(matplotlib.colors.rgb2hex(rgb))
substances = cereal[['protein', 'fat', 'fiber', 'carbo', 'sugars', 'mfr']]

pal = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

substances[substances['mfr']=='R'].plot.bar(stacked=True, figsize=(15, 7), color=pal)

plt.xlabel('Indexes')

plt.ylabel('Grams')

plt.title('Content of Substances in Cereals of the Manufacturer "Ralston Purina"', fontsize=20);