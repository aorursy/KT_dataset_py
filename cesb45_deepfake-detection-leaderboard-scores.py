import os

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt



f = open('/kaggle/input/leaderboard-scores/private_scores.csv')

lines = f.readlines()

lines = [v.rstrip('\n') for v in lines]



teams = lines[2::8]

scores = lines[5::8]





teams = []

scores = []



for i in range(len(lines)):

    line = lines[i]

    if line.isnumeric():

        if i >= 8:

            prev_score = float(lines[i - 3][1:])

            scores.append(prev_score)

            

        team = lines[i + 2][1:]

        teams.append(team)

last_score = float(lines[-3][1:])

scores.append(last_score)





priv_scores = {}

for team, score in zip(teams, scores):

    priv_scores[team] = score

    

#priv_scores



pub_df = pd.read_csv('/kaggle/input/leaderboard-scores/public_scores.csv')

#pub_df



pub_scores = {}

for idx, row in pub_df.iterrows():

    team = row.TeamName

    score = row.Score

    pub_scores[team] = score



priv_sc = []

pub_sc = []



for team, priv_score in priv_scores.items():

    if team in pub_scores:

        pub_score = pub_scores[team]

        priv_sc.append(priv_score)

        pub_sc.append(pub_score)

        #print(team, pub_score, priv_score)



priv_sc = np.array(priv_sc)

pub_sc = np.array(pub_sc)



threshold = 1



x = pub_sc[np.logical_and(pub_sc < threshold, priv_sc < threshold)]

y = priv_sc[np.logical_and(pub_sc < threshold, priv_sc < threshold)]



plt.figure(figsize=(16,12))

plt.scatter(x, y, marker='.')

plt.plot(x, x)

plt.xlabel('Public Score')

_=plt.ylabel('Private Score')
