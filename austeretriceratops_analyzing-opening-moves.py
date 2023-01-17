# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
games = pd.read_csv("/kaggle/input/chess/games.csv")
games.head(3)
games = games.drop(columns=["id", "created_at", "last_move_at", "white_id", "black_id", "rated", "increment_code"])
games.head(3)
opening_moves = games.loc[:, ["white_rating", "moves"]]
opening_moves['moves'] = opening_moves['moves'].apply(lambda s: s[:s.find(' ')])
opening_moves
move_frequencies = opening_moves['moves'].value_counts()/len(opening_moves)

fig = plt.figure(figsize=(14, 8))
sns.barplot(x = move_frequencies.keys(), y = move_frequencies.array)
sns.set()
sns.set_palette("twilight_shifted_r")


opening_keys = move_frequencies.keys()[:4]
top_4_opens = [opening_moves[opening_moves['moves'] == i]["white_rating"] for i in opening_keys]

cumulative_opens = [top_4_opens[3].copy()]

for i in range(1, 4):
    c = cumulative_opens[i-1].append(top_4_opens[3-i].copy())
    cumulative_opens.append(c)
    
fig, ax = plt.subplots(1,1, figsize=(16,10))
ax.set_title("opening frequency vs. ELO")

for i in range(4):
    ax = sns.distplot(cumulative_opens[3-i], 20,  kde=False, hist_kws={"alpha": 1, "range": (800, 2800)}, label=opening_keys[i])
    
ax.legend()
def string_to_moves(string):
    
    moves = []
    
    while " " in string:        
        ind = string.find(" ")
        moves.append(string[0 : ind])
        string = string[ind+1:]
        
    moves.append(string)
    return moves

def move_of_first_capture(moves):
    i = 1
    
    for move in moves:
        if "x" in move:
            return i
        i += 1
    
    return 0
def normalize(row):
    return move_of_first_capture(string_to_moves(row["moves"]))/row["turns"]

capture_data = games.apply(lambda row: normalize(row),axis=1)

fig, ax = plt.subplots((1), figsize=(16,10))
ax = sns.distplot(capture_data, bins=30, kde=False) 
ax.set_title("First capture of a game")
ax.set_xlabel("Duration")
ax.set_ylabel("Frequency")
