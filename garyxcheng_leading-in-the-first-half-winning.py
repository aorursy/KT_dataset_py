import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline



pd.options.mode.chained_assignment = None

DOWN_AT_HALF = -1

TIE_AT_HALF = 0

UP_AT_HALF = 1



df16 = pd.read_csv("../input/2016-17_teamBoxScore.csv")

df17 = pd.read_csv("../input/2017-18_teamBoxScore.csv")
#Choose on of the following options to pick which dataset

#df = df16

#df = df17

df = pd.concat((df16, df17))
df2 = df[["teamAbbr", "teamPTS", "teamPTS1", "teamPTS2", "opptPTS", "opptPTS1", "opptPTS2"]]



#record half time points and point differentials

df2.loc[:, "teamPTSH1"] = df2["teamPTS1"] + df["teamPTS2"]

df2.loc[:, "opptPTSH1"] = df2["opptPTS1"] + df["opptPTS2"]



df2.loc[:, "ptdiffH1"] = df2["teamPTSH1"] - df2["opptPTSH1"]

df2.loc[:, "ptdiff"] = df2["teamPTS"] - df2["opptPTS"]
def make_point_diff_mat(df):

    point_diff_df = df[["ptdiffH1", "ptdiff"]]

    point_diff = point_diff_df.as_matrix()

    return point_diff



def make_bool_point_diff_mat(df):

    point_diff = make_point_diff_mat(df)

    bool_point_diff = np.copy(point_diff)

    bool_point_diff[bool_point_diff > 0] = 1

    bool_point_diff[bool_point_diff < 0] = -1

    return bool_point_diff



def prob_of_winning_given(bool_point_diff, event):

    return np.mean((bool_point_diff[bool_point_diff[:,0] == event][:, 1] + 1 ) / 2)
point_diff = make_point_diff_mat(df2)

np.corrcoef(point_diff.T)
plt.scatter(point_diff[:, 0], point_diff[:, 1])

plt.ylabel("point differential: end of game")

plt.xlabel("point differential: end of first half")
bool_point_diff = make_bool_point_diff_mat(df2)

np.corrcoef(bool_point_diff.T)
# Probability of winning given that you are TRAILING in the first half

prob_of_winning_given(bool_point_diff, DOWN_AT_HALF)
# Probability of winning given that you are LEADING in the first half

prob_of_winning_given(bool_point_diff, UP_AT_HALF)
max_prob_winning_DOWN_at_half = 0

max_team = None

for abbr in df2.teamAbbr.unique():

    df_team = df2[df.teamAbbr == abbr]

    bool_point_diff_team = make_bool_point_diff_mat(df_team)

    prob = prob_of_winning_given(bool_point_diff_team, DOWN_AT_HALF)    

    if prob > max_prob_winning_DOWN_at_half:

        max_prob_winning_DOWN_at_half = prob

        max_team = abbr

print(max_team)

print(max_prob_winning_DOWN_at_half)
max_prob_winning_UP_at_half = 0

max_team = None

for abbr in df2.teamAbbr.unique():

    df_team = df2[df.teamAbbr == abbr]

    bool_point_diff_team = make_bool_point_diff_mat(df_team)

    prob = prob_of_winning_given(bool_point_diff_team, UP_AT_HALF)    

    if prob > max_prob_winning_UP_at_half:

        max_prob_winning_UP_at_half = prob

        max_team = abbr

print(max_team)

print(max_prob_winning_UP_at_half)
df_cavs = df2[df2["teamAbbr"] == "CLE"]
point_diff_cavs = make_point_diff_mat(df_cavs)

np.corrcoef(point_diff_cavs.T)
plt.scatter(point_diff_cavs[:, 0], point_diff_cavs[:, 1])

plt.ylabel("point differential: end of game")

plt.xlabel("point differential: end of first half")
bool_point_diff_cavs = make_bool_point_diff_mat(df_cavs)

np.corrcoef(bool_point_diff_cavs.T)
prob_of_winning_given(bool_point_diff_cavs, DOWN_AT_HALF)
prob_of_winning_given(bool_point_diff_cavs, UP_AT_HALF)
df_warr = df2[df2.teamAbbr == "GS"]
point_diff_warr = make_point_diff_mat(df_warr)

np.corrcoef(point_diff_warr.T)
plt.scatter(point_diff_warr[:, 0], point_diff_warr[:, 1])

plt.ylabel("point differential: end of game")

plt.xlabel("point differential: end of first half")
bool_point_diff_warr = make_bool_point_diff_mat(df_warr)

np.corrcoef(bool_point_diff_warr.T)
prob_of_winning_given(bool_point_diff_warr, DOWN_AT_HALF)
prob_of_winning_given(bool_point_diff_warr, UP_AT_HALF)