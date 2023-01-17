import pandas as pd

import matplotlib.pyplot as plt
songs = pd.read_csv('../input/top50spotify2019/top50.csv', encoding='latin')
songs.head()
plt.scatter(songs['Beats.Per.Minute'], songs['Danceability'])

plt.title("Correlation between BPM and Danceability")

plt.xlabel("Beats per Minute")

plt.ylabel("Danceability")