import pandas as pd 

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline



df = pd.read_csv("../input/wine-dataset/wine.csv")



df.head()
df.describe()
plt.figure(figsize=(6,6))

plt.scatter(df["Color.int"],df["Phenols"],s=15)
df.corr(method='pearson')
df.corr(method='pearson')["Color.int"]
color_corr = sorted(abs(df.corr(method='pearson')["Color.int"]),reverse=True)[0:]

print(color_corr)
fig1,ax1 = plt.subplots()

fig1.set_size_inches(12,7.5)

ax1.set_ylabel('Alcohol')

ax1.set_xlabel('Color Intensity')

ax1.set_title('Relationship Between Color Intensity and Alcohol Content in Wines')

c = df['Color.int']

plt.scatter(df["Color.int"],df["Alcohol"],c=c, 

            cmap = 'Reds', s = df['Alcohol']*5, alpha =0.5)  #Plot of color vs Alcohol

cbar = plt.colorbar()

cbar.set_label('Color.int')
fig2,ax2 = plt.subplots()

fig2.set_size_inches(12,7.5)

ax2.set_ylabel('Hue')

ax2.set_xlabel('Color Intensity')

ax2.set_title('Relationship Between Color Intensity and Hue of Wines')

c = df['Color.int']

plt.scatter(df["Color.int"],df["Hue"],c=c, 

            cmap = 'RdPu', s = df['Hue']*50, alpha =0.5)  #Plot of color vs Hue

cbar = plt.colorbar()

cbar.set_label('Color.int')

fig3,ax2 = plt.subplots()

fig3.set_size_inches(12,7.5)

ax2.set_ylabel('OD')

ax2.set_xlabel('Color Intensity')

ax2.set_title('Relationship Between Color Intensity and OD of Wines')

c = df['Color.int']

plt.scatter(df["Color.int"],df["OD"],c=c, 

            cmap = 'GnBu', s = df['OD']*20, alpha =0.5)  #Plot of color vs OD

cbar = plt.colorbar()

cbar.set_label('Color.int')