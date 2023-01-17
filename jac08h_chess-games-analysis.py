import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import re



sns.set(color_codes=True, style='darkgrid')

%matplotlib inline
games = pd.read_csv('../input/games.csv')

games.head(2)
games = games[games.rated]  # only rated games

games['mean_rating'] = (games.white_rating + games.black_rating) / 2

games['rating_diff'] = abs(games.white_rating - games.black_rating)
plt.figure(figsize=(10,5))

sns.distplot(games['mean_rating'])
plt.figure(figsize=(10,5))

sns.distplot(games.turns)
games.victory_status.value_counts()
under_1500 = games[games.mean_rating < 1500]

under_2000 = games[games.mean_rating < 2000]

over_2000 = games[games.mean_rating > 2000]



brackets = [under_1500, under_2000, over_2000]

bracket_titles = ['Under 1500', 'Under 2000', 'Over 2000']
plt.figure(figsize=(15,11))

for i, bracket in enumerate(brackets):

    victory_status = bracket.victory_status.value_counts()

    plt.subplot(1, 4, i+1)

    plt.title(bracket_titles[i])

    plt.pie(victory_status, labels=victory_status.index)
mate_games = games[games.victory_status=='mate']



under_1500 = mate_games[mate_games.mean_rating < 1500]

under_2000 = mate_games[mate_games.mean_rating < 2000]

over_2000 = mate_games[mate_games.mean_rating > 2000]



m_brackets = [under_1500, under_2000, over_2000]
turn_means = [b.turns.mean() for b in m_brackets]



plt.figure(figsize=(10,5))

plt.ylim(0, 100)

plt.title('Number of turns until mate')

plt.plot(bracket_titles, turn_means, 'o-', color='r')
plt.figure(figsize=(10,5))

plt.scatter(mate_games.mean_rating, mate_games.turns)
mate_games.loc[mate_games['turns'].idxmax()]
scholar_mates = mate_games[mate_games.turns==4]

scholar_mates

white_upsets = games[(games.winner == 'white') & (games.white_rating < games.black_rating)]

black_upsets = games[(games.winner == 'black') & (games.black_rating < games.white_rating)]

upsets = pd.concat([white_upsets, black_upsets])
THRESHOLD = 900

STEP = 50



u_percentages = []



print(f'Rating difference : Percentage of wins by weaker player')

for i in range(0+STEP, THRESHOLD, STEP):

    th_upsets = upsets[upsets.rating_diff > i]

    th_games = games[games.rating_diff > i]

    upsets_percentage = (th_upsets.shape[0] / th_games.shape[0]) * 100

    u_percentages.append([i, upsets_percentage])

    print(f'{str(i).ljust(18)}:  {upsets_percentage:.2f}%')
plt.figure(figsize=(10,5))

plt.plot(*zip(*u_percentages))

plt.xlabel('rating difference')

plt.ylabel('upsets percentage')
p = re.compile('([a-h][1-8])')

squares = {}

for moves in games.moves:

    for move in moves.split():

        try:

            square = re.search(p, move).group()

        except AttributeError:  # castling

            square = move.replace('+', '')

        squares[square] = squares.get(square, 0) + 1
squares_df = pd.DataFrame.from_dict(squares, orient='index', columns=['count'])



# add castling



total_shorts = int(squares_df.loc['O-O'])

total_longs = int(squares_df.loc['O-O-O'])



half_shorts = total_shorts//2

half_longs = total_longs//2



# white short castling

squares_df.loc['f1'] = squares_df.loc['f1'] + half_shorts

squares_df.loc['g1'] = squares_df.loc['g1'] + half_shorts

# black short castling

squares_df.loc['f8'] = squares_df.loc['f8'] + half_shorts

squares_df.loc['g8'] = squares_df.loc['g8'] + half_shorts 

# white long castling

squares_df.loc['c1'] = squares_df.loc['c1'] + half_longs

squares_df.loc['d1'] = squares_df.loc['d1'] + half_longs

# black long castling

squares_df.loc['c8'] = squares_df.loc['c8'] + half_longs

squares_df.loc['d8'] = squares_df.loc['d8'] + half_longs



squares_df.drop(['O-O', 'O-O-O'], inplace=True)
total_castles = total_shorts + total_longs

print(f'Short: {(total_shorts/total_castles)*100:.2f}%')

print(f'Long: {(total_longs/total_castles)*100:.2f}%')
plt.figure(figsize=(10,5))

plt.pie([total_shorts, total_longs],

       labels=['Short', 'Long'])
squares_df.reset_index(inplace=True)

squares_df['letter'] = squares_df['index'].str[0]

squares_df['number'] = squares_df['index'].str[1]
squares_df = squares_df.pivot('number', 'letter', 'count')

squares_df.sort_index(level=0, ascending=False, inplace=True)  # to get right chessboard orientation

squares_df
sns.set(rc={'figure.figsize':(20,15)})

hm = sns.heatmap(squares_df,

            cmap='Oranges',

            annot=False, 

            vmin=0,

            fmt='d',

            linewidths=2,

            linecolor='black',

            cbar_kws={'label':'occupation'},

            )

hm.set(xlabel='', ylabel='')
