import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



# coordinates for the goal

GOAL_CENTER_X_COORD = 89

GOAL_CENTER_Y_COORD = 0



# read csv

game_plays_df = pd.read_csv('../input/nhl-game-data/game_plays.csv')

game_plays_df.head()
# get plays that are either a shot or a goal

shots_df = game_plays_df.loc[(game_plays_df['event'] == 'Shot') | (game_plays_df['event'] == 'Goal')]



# remove N/A's

shots_df = shots_df.dropna()



# only include shots taken in offensive zone

shots_df = shots_df[shots_df['st_x'] >= 25]



shots_df.head()
# categorize shots

shots_df.secondaryType = pd.Categorical(shots_df.secondaryType)

shots_df['shot_type'] = shots_df.secondaryType.cat.codes



shots_df[['secondaryType','shot_type']].head()
# compute shot angle

shots_df['shot_angle'] = shots_df[['st_x', 'st_y']].apply(

    lambda row: np.arctan(row['st_y']/(89 - row['st_x'])) * (180/np.pi),

    axis=1

)



# compute shot distance

shots_df['shot_dist'] = shots_df[['st_x', 'st_y']].apply(

    lambda row: np.sqrt((row['st_x'] - GOAL_CENTER_X_COORD)**2 + (row['st_y'] - GOAL_CENTER_Y_COORD)**2),

    axis=1

)
# sample shots w/o replacement

shots_sample_df = shots_df.sample(n=1000, replace=False)
# generate train and test sets

X = shots_sample_df[['shot_angle','shot_dist','shot_type']]

y = shots_sample_df['event']

indicies = range(len(shots_sample_df))



# train with 75% of sample

X_train, X_test, y_train, y_test, train_indicies, test_indicies = train_test_split(X, y, indicies, test_size=0.25)
# perform logistic regression

logisticRegr = LogisticRegression()

logisticRegr.fit(X_train, y_train)



# use the model on the test set and save the prob of goal

prob = logisticRegr.predict_proba(X_test)[:,[0]]
# generate heat map of predicted probabilities

x_coords = shots_sample_df['st_x'].iloc[test_indicies]

y_coords = shots_sample_df['st_y'].iloc[test_indicies]



plt.scatter(x_coords, y_coords, c=prob.T[0], cmap=plt.cm.coolwarm)

plt.plot(GOAL_CENTER_X_COORD, GOAL_CENTER_Y_COORD, 'kx')

plt.axis([25, 100, -42.5, 42.5])

plt.show()
# print the model's score

score = logisticRegr.score(X_test, y_test)

print(score)