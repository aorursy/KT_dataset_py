# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
chess_df = pd.read_csv("/kaggle/input/chess/games.csv")
chess_df.info()
chess_df['victory_status'].unique()
import math

skill_difference = chess_df.loc[abs(chess_df['white_rating'] - chess_df['black_rating']) > 0]
skill_difference.count()
outsiders = skill_difference.loc[(
    (skill_difference['white_rating'] > skill_difference['black_rating']) & (skill_difference['winner'] == 'black')
    |
    (skill_difference['white_rating'] < skill_difference['black_rating']) & (skill_difference['winner'] == 'white')
)]
outsiders
import matplotlib.pyplot as plt

plt.scatter(x=outsiders['white_rating'].loc[outsiders['winner'] == 'white'], y=outsiders['black_rating'].loc[outsiders['winner'] == 'white'], c='pink', alpha=0.4)
plt.scatter(x=outsiders['white_rating'].loc[outsiders['winner'] == 'black'], y=outsiders['black_rating'].loc[outsiders['winner'] == 'black'], c='black', alpha=0.4)
plt.xlabel('White\'s rating')
plt.ylabel('Black\'s rating')
plt.show()
plt.scatter(x=chess_df['white_rating'].loc[chess_df['winner'] == 'white'], y=chess_df['black_rating'].loc[chess_df['winner'] == 'white'], c='pink', alpha=0.4)
plt.scatter(x=chess_df['white_rating'].loc[chess_df['winner'] == 'black'], y=chess_df['black_rating'].loc[chess_df['winner'] == 'black'], c='black', alpha=0.4)
plt.xlabel('White\'s rating')
plt.ylabel('Black\'s rating')
plt.show()
quick_mate = chess_df.loc[(chess_df['victory_status'] == 'mate') & (chess_df['turns'] < 15)]
qm_copy = quick_mate.copy()

qm_copy['average_rating'] = abs(qm_copy['white_rating'] + qm_copy['black_rating']) / 2.0
qm_copy.sort_values(by=['average_rating'], ascending=False)
quick_mate_moves = qm_copy['moves'].tolist()
for moveset in quick_mate_moves:
    moveset = moveset.split(' ')

def get_last_move(winner=None):
    mates = chess_df.loc[chess_df['victory_status'] == 'mate']
    if(winner != None):
        mates = mates.loc[mates['winner'] == winner]
    mates_all_moves = mates['moves']
    mates_last_move = []
    for mate in mates_all_moves:
        mate = mate.split(' ')
        mates_last_move.append(mate[len(mate) - 1])
    return mates_last_move

def clean_moves_of_pieces(mates_last_move):
    mates_clean = []
    for i in range(len(mates_last_move)):
        mates_clean.append('')
        mates_clean[i] = mates_last_move[i].replace('Q', '') # Deleting Queen notation
        mates_clean[i] = mates_clean[i].replace('R', '') # Deleting Rook notation
        mates_clean[i] = mates_clean[i].replace('B', '') # Deleting Bishop notation
        mates_clean[i] = mates_clean[i].replace('N', '') # Deleting Knight notation
        mates_clean[i] = mates_clean[i].replace('x', '') # Deleting capture notation
        mates_clean[i] = mates_clean[i].replace('#', '') # Deleting mate notation
        mates_clean[i] = mates_clean[i].replace('=', '') # Deleting promotion notation

    for i in range(len(mates_clean)):
        mate = list(mates_clean[i])
        if len(mate) > 2:
            mate[0] = ''
        mates_clean[i] = "".join(mate)
    
    return mates_clean

def turn_into_number_notation(mates_clean):
    number_notation = []
    for mate in mates_clean:
        lmate = list(mate)
        number_notation.append([ord(lmate[0]) - 96, int(lmate[1])])
    return number_notation
        
def full_mate_transformation(winner=None):
    mates_last_move = get_last_move(winner)
    mates_clean = clean_moves_of_pieces(mates_last_move)
    number_notation = turn_into_number_notation(mates_clean)
    
    return np.array(number_notation)

mates_in_number_notation = full_mate_transformation(winner='black')
def turn_into_board_of_values(mates):
    (unique, counts) = np.unique(mates, axis=0, return_counts=True)
    counts = counts.reshape(8, 8)
    percentages = counts / counts.sum() * 100
    percentages = np.round(percentages, 2)
    
    return counts, percentages

mates_count, mates_percentages = turn_into_board_of_values(mates_in_number_notation)
mates_count, mates_percentages
# Visualization, finally
import matplotlib


horizontal = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
vertical = ['1', '2', '3', '4', '5', '6', '7', '8']

vertical.reverse()

# cmaps['Sequential'] = [
#             'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot()
im = ax.imshow(mates_percentages, cmap='cividis')

ax.set_xticks(np.arange(len(horizontal)))
ax.set_yticks(np.arange(len(vertical)))

ax.set_xticklabels(horizontal)
ax.set_yticklabels(vertical)

for i in range(len(vertical)):
    for j in range(len(horizontal)):
        text = ax.text(j, i, mates_percentages[i,j], ha='center', va='center', color='w')
        
ax.set_title('Percentage of the mates delivered on a given square')
fig.tight_layout()

plt.show()
