import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
url = "https://raw.githubusercontent.com/aniket-spidey/bitgrit-webinar/master/code/datasets/spotify.csv"
spotify_data = pd.read_csv(url)
spotify_data.head(10)
X = spotify_data['Beats.Per.Minute']
y = spotify_data['Danceability']
plt.scatter(X, y)
X = spotify_data['Genre']
y = spotify_data['Energy']
plt.barh(X, y)
