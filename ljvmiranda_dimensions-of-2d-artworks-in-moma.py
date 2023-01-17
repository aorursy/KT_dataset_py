# Import modules

import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import seaborn as sns



# Seaborn Settings

sns.set_style(style='white')



# Import data 

df = pd.read_csv("../input/artworks.csv")
df.Classification.unique()
# Get all works in 2D

# Design, Drawing, Painting, Photograph

df_drawing = df[df['Classification'] == 'Drawing']

df_painting = df[df['Classification'] == 'Painting']

df_photo = df[df['Classification'] == 'Photograph']





# Concatenate all of artworks

all_2D = [df_painting, df_drawing, df_photo]

df_2d = pd.concat(all_2D,axis=0,ignore_index=True)
df_2d.info()
# Drop some columns except Classification, Height, and Width

df_2d = df_2d[['Title', 'Classification', 'Height (cm)', 'Width (cm)']]

df_2d = df_2d.rename(columns={'Height (cm)': 'height', 'Width (cm)': 'width'})



# Remove artworks with NaN

df_2d = df_2d.dropna()
# Plot 

ratio =  np.log10(df_2d['height'])/np.log10(df_2d['width'])

width = np.log10(df_2d['width'])



# 4/3

four_thirds = np.log10(4)/np.log10(3)

three_fourths = np.log10(3)/np.log10(4)

h = plt.scatter(width,ratio, alpha=0.02, c='c')

plt.axhline(y=1.0, color='k', linestyle='-',linewidth=0.75,label='Square')

plt.axhline(y=four_thirds, color='r', linestyle='-',linewidth=0.75,label='4x3')

plt.axhline(y=three_fourths, color='r', linestyle='-',linewidth=0.75)

plt.xlim((0.5,3.5))

plt.ylim((0.6,1.4))

plt.title("Dimensions of 2D Artworks in MoMA")

plt.xlabel('Width (cm) [log scale]')

plt.ylabel('Height/Width')

plt.legend()

plt.show()