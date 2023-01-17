# Import important Modules

!pip install plotly

import pandas as pd

import numpy as np



import seaborn as sb

import matplotlib

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder 

%matplotlib inline
# Load data, and print first 5 rows displaying all columns

data = pd.read_csv("../input/games.csv")

data.head()
# How many Distinct Players do we have?

# Get totals from both black and white which are unique, combine them, then get unique players from that combined list



unique_white = data["white_id"].unique().tolist()

unique_black = data["black_id"].unique().tolist()

unique_white.extend(unique_black)



all_players = []

for player in unique_white:

    if player not in all_players:

        all_players.append(player)



print("Number of unique Players:", len(all_players))

# Remember, rating is changing every game, and since there are plenty of games with the same players,

# a player in this dataset cannot be linked to one single rating



# lets get a measure of the distribution of the ratings of the players for each game

ratings = data["white_rating"].tolist()

b_ratings = data["black_rating"].tolist()

ratings.extend(b_ratings)



# Basic Insights into what ratings we are working with

mean_rating, max_rating, min_rating, std_rating = round(np.mean(ratings), 2), max(ratings), min(ratings), round(np.std(ratings), 2)

print("Mean Rating:", mean_rating)

print("Max Rating:", max_rating)

print("Min Rating:", min_rating)

print("Std Rating:", std_rating)



# Set up a histogram, bins start from 700 to 2800+ is a good range in general for online chess analysis)



# For the bins, bin "length" of 300 seemed reasonable as it distinguishes the different skill levels well 

bins = [i for i in range(700, 3100, 300)]



plt.hist(ratings, bins, histtype="bar", rwidth=0.9, color="orange")

plt.title("Distribution of Player Rating")

plt.xlabel("Player ratings")

plt.ylabel("Number of players")

plt.xticks(range(700, 3100, 300))

plt.show()

# We can get a count of how many wins there were for each color

white_wins = data["winner"].tolist().count("white")

black_wins = data["winner"].tolist().count("black")





plt.bar(["White wins", "Black wins"], [white_wins, black_wins], color="darkblue", width=0.3)

plt.show()
# we can see how many unique openings are played

openings_unique = data["opening_name"].unique().tolist()

print("Number of unique openings:", len(openings_unique))
# Q1 - white opening move is always first move, so just first two characters in each string in each moves column 

list_of_first_moves = [moves[:2] for moves in data["moves"]]

# not many opening moves possible, so we can easily put all possible into a list

poss_opening_moves = ["a3", "a4", "b3", "b4", "c3", "c4", "d3", "d4", "e3", "e4", "f3", "f4", "g3", "g4", "h3", "h4", "Nf", "Nh", "Na", "Nc"]



dict_moves_count = {}



for op_move in poss_opening_moves:

    dict_moves_count[op_move] = list_of_first_moves.count(op_move)

print("Count of moves Dictionary:", dict_moves_count)

# Plot moves frequency via bar chart

plt.bar(range(len(dict_moves_count)), list(dict_moves_count.values()), align="center")

plt.xticks(range(len(dict_moves_count)), list(dict_moves_count.keys()))

plt.show()

# rating difference column creation



def dif(col_1val, col_2val):

    return col_1val - col_2val



data["rating_diff"] = [dif(data["white_rating"][i], data["black_rating"][i]) for i in range(len(data))]

# Encode necessary columns, create new DataFrame

data_copy = data.copy()

l_encode = LabelEncoder()



# Encode winner (target) = 0, 1, 2 (0 black loss, 1 draw, 2 white win)

data_copy["EncWinner"] = l_encode.fit_transform(data["winner"])



data_copy["EncRated"] = l_encode.fit_transform(data["rated"])

data_copy["EncOpeningEco"] = l_encode.fit_transform(data["opening_eco"])

data_copy["EncVictoryStatus"] = l_encode.fit_transform(data["victory_status"])



# Encode Time control (increment_code) column into 4 categories (Bullet, Blitz, Rapid, Classical)

def encode_time(time):

    index_plus = time.index("+")

    minutes = int(time[:index_plus])

    

    if minutes < 3:

        return 0

    elif minutes >= 3 and minutes < 15:

        return 1

    elif minutes >= 15 and minutes < 30:

        return 2

    elif minutes >= 30:

        return 3

    else:

        # invalid

        return -1

    

data_copy["EncIncrementCode"] = data["increment_code"].apply(lambda x: encode_time(x))
# Feature reduction, drop unnessacary colunms

data_copy.drop(["id", "rated", "created_at", "last_move_at", 

                "victory_status", "winner", "increment_code", "white_id",

               "black_id", "moves", "opening_eco", "opening_name", "opening_ply"], axis=1, inplace=True)

# New DataFrame

data_copy.head()
plt.figure(figsize=(12, 10))

cor = data_copy.corr()

sb.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
# Lets get the correlation with the output variable (winner)

cor_target = abs(cor["EncWinner"])

relavent_features = cor_target[cor_target > 0.2]

print(relavent_features)


