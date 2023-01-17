import pandas as pd

import numpy as np

import matplotlib.pylab as plt

import seaborn as sns

import matplotlib.style as style

style.use('fivethirtyeight')

from collections import Counter
artists = pd.read_csv("../input/museum-of-modern-art-collection/Artists.csv")

artworks = pd.read_csv("../input/museum-of-modern-art-collection/Artworks.csv")
artists.head()
plt.hist(artists[artists.BeginDate!=0]['BeginDate'])

plt.title('Number of Artists by Date of Birth')

plt.show()
plt.hist(artists[artists.EndDate!=0]['EndDate'])

plt.title('Number of Artists by Date of Death')

plt.show()
Counter(artists['Gender'])
artworks.head()
plt.plot(artworks['Width (cm)'], artworks['Height (cm)'], 'o')

plt.title('Width vs. Height of Artworks \n in MoMA Collection')

plt.show()
paintings = artworks[artworks['Classification'] == 'Painting']
paintings.head()
plt.plot(paintings['Width (cm)'], paintings['Height (cm)'], 'o')

plt.title('Width vs. Height of Paintings \n in MoMA Collection')

plt.show()