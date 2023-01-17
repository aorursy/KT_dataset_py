import numpy as np

import pandas as pd

# Not every data format will be in csv there are other file formats also.

# This exercise will help you deal with other file formats and how toa read it.

path = "../input/ipl_matches_small.csv"

data_ipl = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=str)



print('Data of dataset ',data_ipl[:,3].dtype)
# How many matches were held in total we need to know so that we can analyze further statistics keeping that in mind.

arr = np.array(data_ipl[:,0])

u = np.unique(arr)

print(u)

print('Unique no. of matches in the provided dataset : ',u.size)

len(set(data_ipl[:,0]))
# this exercise deals with you getting to know that which are all those six teams that played in the tournament.

arr1 = np.array(data_ipl[:,12])

print(np.unique(arr1))

t1 = set(data_ipl[:, 3])

t2 = set(data_ipl[:, 4])

unique_teams = t1.union(t2)

unique_teams

# An exercise to make you familiar with indexing and slicing up within data.

arr2 = np.array(data_ipl[:,17])

arr3=arr2.astype('int64')

print('sum of all extras in all deliveries in all matches : ',np.sum(arr3))
extras = data_ipl[:, 17]

extras_int = extras.astype(np.int16)

extras_int.sum()
wicket_filter = (data_ipl[:, 20] == 'SR Tendulkar')

wickets_arr = data_ipl[wicket_filter]

wickets_arr[:, 11]

wickets_arr[:, 21]
wicket_details = data_ipl[np.ix_(data_ipl[:, 21] != '', (11, 20, 21))]

print('Delivery numbers when a player got out, player name and wicket type :\n ',wicket_details)

# this exercise will help you get the statistics on one particular team

team_records = data_ipl[data_ipl[:, 5] == 'Mumbai Indians']

unique_matches = set(team_records[:, 0])

len(unique_matches)
# this exercise will help you get the statistics on one particular team

print(len(np.unique(data_ipl[np.ix_(data_ipl[:, 5] == 'Mumbai Indians', (0, 3, 4))][:, 0])), "times 'Mumbai Indian' has won the toss")
len(set(data_ipl[:, 5] == 'Mumbai Indians'))
# An exercise to know who is the most aggresive player or maybe the scoring player 

sixes = data_ipl[data_ipl[:, 16].astype(np.int16) == 6]
from collections import Counter

most_sixes_scored = Counter(sixes[:,13],)
most_sixes_scored.most_common()
# An exercise to know who is the most aggresive player or maybe the scoring player 

most_sixes = data_ipl[np.ix_(data_ipl[:, 16] == '6', (13, 16))]

print('Batsman scored 6 runs:',most_sixes)
unique, counts = np.unique(most_sixes[:, 0], return_counts=True)

player = sorted(dict(zip(unique, counts)).items(), reverse=True, key=lambda x: x[1])

print(player[0][0],'has scored most number of sixes')