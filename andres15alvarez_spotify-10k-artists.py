import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



artists = pd.read_csv('../input/spotify-artist-metadata-top-10k/top10k-spotify-artist-metadata.csv')
artists.head()
artists.gender.value_counts()
countgender = artists.gender.value_counts()

countgender.plot(kind='bar', color='skyblue', figsize=(10,5))
artists.type.value_counts()
counttype = artists.type.value_counts()

sizes = [5982,2266]

labels = counttype.index

colors = ['skyblue', 'salmon']

fig, ax = plt.subplots()

ax.pie(sizes, colors=colors, labels=labels, shadow=False, startangle=90, autopct="% 1.2f %%")

ax.axis('equal')  

    
artists.country.value_counts()
artists[artists.country.isnull()]
artists[artists.country=='VE']