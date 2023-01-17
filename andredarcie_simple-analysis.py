import pandas as pd

df = pd.read_csv('../input/list-of-video-games-considered-the-best/video_games_considered_the_best.csv')
df.columns
df.sample(5)
df.shape
pd.set_option('max_rows', 99999)

df.sort_values('refs', ascending=False)
publishers = df.groupby('publisher').size()



publishers_sorted = sorted(dict(publishers).items(), key=lambda x: x[1], reverse=True)

df_publishers = pd.DataFrame(publishers_sorted)

df_publishers.columns = [ 'publisher', 'games']

df_publishers
publishers = df.groupby('genre').size()



publishers_sorted = sorted(dict(publishers).items(), key=lambda x: x[1], reverse=True)

df_genre = pd.DataFrame(publishers_sorted)

df_genre.columns = [ 'genre', 'games']

df_genre
stealth_games = df[df['genre'] == 'Stealth']

stealth_games
from os import path

from wordcloud import WordCloud

import matplotlib.pyplot as plt



wordcloud = WordCloud(max_font_size=40).generate(" ".join(list(df['title'])))

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
import matplotlib.pyplot as plt



plt.title('Game references by year')

plt.ylabel('References')

plt.xlabel('Year')

plt.plot(df['year'], df['refs'])

plt.grid(True)