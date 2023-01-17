import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud

%matplotlib inline
game_df = pd.read_csv('../input/ign.csv')

game_df.head()
game_df.info()
game_df[game_df['platform']=='Web Games']
# Show unique platforms, and their numbers

# We can see PC have most games, and then the several major platform just as we expected

game_df['platform'].value_counts()
# Divide Platforms (I find I learned a lot about game platform history from this divide procedure ) ^-^

# Personal computer in general mearning

PC = ['PC', 'Macintosh', 'Linux', 'Web Games', 'SteamOS']

mobile = ['iPad', 'iPhone', 'Android', 'Wireless', 'iPod', 'Commodore 64/128', 'Windows Phone', 'Windows Surface']

console = ['Saturn', 'Xbox 360', 'PlayStation 3', 'Wii', 'PlayStation 4', 'PlayStation', 'Nintendo 64', \

           'Dreamcast', 'Arcade', 'Nintendo 64DD', 'PlayStation 2', 'Xbox', 'GameCube', 'DVD / HD Video Game', \

           'NES', 'Genesis', 'TurboGrafx-16', 'Super NES', 'Master System', 'NeoGeo', 'Atari 5200', 'TurboGrafx-CD', \

           'Atari 2600', 'Sega 32X', 'Vectrex', 'Sega CD',  'Xbox One', 'Ouya', ]

hand_console = ['PlayStation Vita', 'Nintendo DS', 'Nintendo 3DS', 'Wii U', 'PlayStation Portable', 'Lynx', \

                'Game Boy', 'Game Boy Color', 'NeoGeo Pocket Color', 'Game.Com', 'Dreamcast VMU', 'WonderSwan', \

                'WonderSwan Color', 'Game Boy Advance', 'Pocket PC', 'N-Gage',  'Nintendo DSi', 'New Nintendo 3DS']
# A map function

def map_platform(platform):

    if platform in PC:

        return 'PC'

    elif platform in mobile:

        return 'mobile'

    elif platform in console:

        return 'console'

    else:

        return 'hand_console'
class_df = game_df.copy()

class_df['platform'] = game_df['platform'].apply(map_platform)
fig1, ax1 = plt.subplots(figsize=(5,5))

class_df['platform'].value_counts().plot.pie(shadow=True, autopct='%1.1f%%', startangle=90, \

                                            explode=[0.1, 0, 0, 0], ax=ax1)

ax1.set_title('Top Platforms')

ax1.set_ylabel('')
# General Trend

fig2, ax2 = plt.subplots(figsize=(10,5))

sns.countplot(class_df['release_year'], palette='Set1').set_title('Release on Years')

ax2.set_xlabel('Years')

ax2.set_ylabel('Release Number')
# Time trend according to each platforms

fig3, ax3 = plt.subplots(figsize=(18,5))

sns.countplot(data=class_df,x='release_year',hue='platform', palette='Set1')

ax3.set_title('Releas on Years on Platforms')

ax3.set_xlabel('Years')

ax3.set_ylabel('Release Number')
score_list = ['Masterpiece', 'Amazing', 'Great', 'Good', 'Okay', 'Mediocre', \

              'Bad', 'Awful', 'Painful', 'Unbearable', 'Disaster']
# Score phrase on each platforms

fig4, ax4 = plt.subplots(figsize=(10,5))

plat_score = class_df.groupby(['platform', 'score_phrase'])['score'].count().reset_index()

plat_score = plat_score.pivot('score_phrase', 'platform','score')

plat_score = plat_score.reindex(score_list)

sns.heatmap(plat_score, annot=True, fmt='2.0f', cmap='viridis', linewidths=0.1)

ax4.set_title('Game Number by Score Phrase')
def normalize(df):

    result = df.copy()

    for platform in plat_score.columns:

        total = df[platform].sum()

        result[platform] = df[platform] / total

    return result
fig5, ax5 = plt.subplots(figsize=(10,5))

norm_plat_score = normalize(plat_score)

sns.heatmap(norm_plat_score, annot=True, fmt='.4f', cmap='plasma', linewidths=0.1)

ax5.set_title('Normalized Game Number by Score Phrase')
genre_df = class_df.copy()

# Drop nan row

nan_ind = pd.isnull(genre_df['genre']).nonzero()[0]

genre_df = genre_df.drop(nan_ind)
def split_genre(genre):

    # Split genre string to list

    split_genre = []

    for i in genre.split(','):

        split_genre.append(i.strip())

    return split_genre
def genre_socre(genre):

    return 1/len(genre)
genre_df['genre_score'] = genre_df['genre'].apply(genre_socre)

genre_df['genre'] = genre_df['genre'].apply(split_genre)
multi_tags = genre_df[genre_df['genre'].apply(len) >= 2]

genre_df = genre_df.drop(multi_tags.index)
genre_df['genre'] = genre_df['genre'].apply(''.join)
for i in range(len(multi_tags)):

    for tag in multi_tags['genre'].iloc[i][:]:

        line = multi_tags.iloc[i].copy()

        line['genre'] = tag

        genre_df = genre_df.append(line, ignore_index=True)
fig6, ax6 = plt.subplots(figsize=(10,5))

max_genres = genre_df.groupby('genre')['genre'].count()

top_10_genres=max_genres.sort_values(ascending=False)[:10]

plat_genre = genre_df[genre_df['genre'].isin(top_10_genres.index)]

plat_genre = plat_genre.groupby(['platform', 'genre'])['genre_score'].sum().reset_index()

plat_genre = plat_genre.pivot('genre', 'platform','genre_score')

sns.heatmap(plat_genre, annot=True, fmt='2.0f', cmap='viridis', linewidths=0.1)

ax6.set_title('Game Number by Genre')
fig7, ax7 = plt.subplots(figsize=(10,5))

norm_plat_genre = normalize(plat_genre)

sns.heatmap(norm_plat_genre, annot=True, fmt='.4f', cmap='plasma', linewidths=0.1)

ax7.set_title('Normalized Game Number by genre')
# Try to make a word cloud of genre

initial = genre_df['genre'].str.split(',')

genre_words= []

for item in initial:

    for word in item:

        genre_words.append(word.strip())

wc_text = ' '.join(genre_words)



wordcloud = WordCloud(background_color='black', width=800, height=400).generate(wc_text)

plt.figure(figsize=(15,8))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
fig8, ax8 = plt.subplots(figsize=(16,8))

editor_score = game_df.groupby(['editors_choice', 'score_phrase'])['score'].count().reset_index()

sns.barplot(data=editor_score, x='editors_choice', y='score', hue='score_phrase', \

            hue_order=score_list, palette='Set1')

ax8.set_ylabel('count')

ax8.set_title('Editor Choice and Score Phrase')
# Release trend by month and day

fig, ax= plt.subplots(1,2, figsize=(15,5))

sns.countplot(class_df['release_month'], ax=ax[0], palette='Set1').set_title('Release on Months')

ax[0].set_xlabel('Months')

sns.countplot(class_df['release_day'], ax=ax[1], palette='Set1').set_title('Release on Days')

ax[1].set_xlabel('Days')
# Make a modified title wordcloud

from wordcloud import STOPWORDS

# From the first time word cloud, I find there are several words are no meaning

# Like edition, II, III, so we can add them to STOPWORDS

STOPWORDS.update(['edition', 'II', 'III', '3D', 'Game'])

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='black', 

                      width=800, 

                      height=400).generate(" ".join(game_df['title']))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()