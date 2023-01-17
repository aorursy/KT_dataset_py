import pandas as pd

import numpy as np

%matplotlib inline

%pylab inline



pd.set_option('display.max_columns', 200)
chess_games = pd.read_csv('../input/chess/games.csv', delimiter=',')
chess_games.head()
games_delay_in_sec = (chess_games['last_move_at'] - chess_games['created_at']) / 1000

chess_games['duration_in_seconds'] = games_delay_in_sec.copy()
from category_encoders import OneHotEncoder
ohe_victory_status = OneHotEncoder(cols=['victory_status'], use_cat_names=True, drop_invariant=True)

chess_games = ohe_victory_status.fit_transform(chess_games)
minutes = chess_games['increment_code'].str.split('+').map(lambda time_control: time_control[0], na_action=None).astype(int)

incr_seconds = chess_games['increment_code'].str.split('+').map(lambda time_control: time_control[1], na_action=None).astype(int)



chess_games['minutes'] = minutes.copy()

chess_games['incr_seconds'] = incr_seconds.copy()
chess_games = chess_games.drop(columns=['increment_code'], axis=1)
chess_games['created_at'] = pd.to_datetime(chess_games['created_at'], unit='ms')

chess_games['last_move_at'] = pd.to_datetime(chess_games['last_move_at'], unit='ms')
chess_games['rating_difference'] = chess_games['white_rating'] - chess_games['black_rating']
chess_games['rating_difference'].mean()
def get_white_moves(moves):

    return moves[::2]



def get_black_moves(moves):

    return moves[1::2]



def castled(moves):

    return ('O-O' in moves) | ('O-O-O' in moves)



all_moves = chess_games['moves'].str.split()

white_moves = all_moves.apply(get_white_moves)

black_moves = all_moves.apply(get_black_moves)

chess_games['white_castled'] = white_moves.apply(castled).astype(int)

chess_games['black_castled'] = black_moves.apply(castled).astype(int)
def count_takes(moves):

    moves = pd.Series(moves)

    return moves.map(lambda mv: 1 if 'x' in mv else 0).sum()



chess_games['white_takes_count'] = white_moves.apply(count_takes)

chess_games['black_takes_count'] = black_moves.apply(count_takes)
chess_games.head()
max_minutes = chess_games['minutes'].max()

max_incr_seconds = chess_games['incr_seconds'].max()

mean_moves = chess_games['turns'].mean()



duration_threshold_in_hours = (max_minutes + max_incr_seconds / 60 * mean_moves) / 60

duration_threshold_in_hours
chess_games = chess_games[chess_games['turns'] > 3]

chess_games = chess_games[chess_games['duration_in_seconds'] < duration_threshold_in_hours * 3600]
chess_games.shape
duration0 = chess_games[chess_games['duration_in_seconds'] == 0]

duration0['winner'].value_counts()
pyplot.hist(x=chess_games['duration_in_seconds'], bins=100)

pyplot.xlabel('Duration (in seconds)')

pyplot.ylabel('Frequency')

pyplot.title('Matches\' Durations')

pyplot.show()
chess_games = chess_games[chess_games['duration_in_seconds'] > 0]
chess_games.shape
pyplot.hist(x=chess_games['duration_in_seconds'], bins=100)

pyplot.xlabel('Duration (in seconds)')

pyplot.ylabel('Frequency')

pyplot.title('Matches\' Durations')

pyplot.show()
baseline = pd.DataFrame(index=chess_games.index)

baseline['rating_difference'] = chess_games['rating_difference']

baseline.shape
baseline['rating_difference'].mean()
def get_base_winner(rating_diff):

    average = baseline['rating_difference'].mean()

    if rating_diff < average and rating_diff > -average:

        return 'draw'

    elif rating_diff < 0:

        return 'black'

    else:

        return 'white'



baseline['winner'] = baseline['rating_difference'].apply(get_base_winner)

baseline['winner'].value_counts()
from sklearn.metrics import precision_recall_fscore_support, classification_report
print(classification_report(chess_games['winner'], baseline['winner'], digits=4))
results = precision_recall_fscore_support(chess_games['winner'], baseline['winner'])

np.average(results[0], weights=results[3])
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
variables = ['victory_status_outoftime', 'victory_status_resign', 'victory_status_mate', 'victory_status_draw', 

             'white_rating', 'black_rating', 'minutes', 'incr_seconds', 'rating_difference',

             'white_castled', 'black_castled', 'white_takes_count', 'black_takes_count']
X = chess_games[variables]

y = chess_games['winner']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)

model.fit(X_train, y_train)



predicted = model.predict(X_valid)



results = precision_recall_fscore_support(y_valid, predicted)

support = results[3]

prec = np.average(results[0], weights=support)

recall = np.average(results[1], weights=support)

print("Precision: {}".format(prec))

print("Recall: {}".format(recall))

print()
from sklearn.model_selection import RepeatedKFold
avg_weighted_precisions = []

avg_weighted_recalls = []

kf = RepeatedKFold(n_splits=2, n_repeats=10, random_state=0)



X = chess_games[variables]

y = chess_games['winner']

i = 0



for lines_train, lines_valid in kf.split(chess_games):

    X_train, y_train = X.iloc[lines_train], y.iloc[lines_train]

    X_valid, y_valid = X.iloc[lines_valid], y.iloc[lines_valid]

    i = i + 1

    

    model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)

    model.fit(X_train, y_train)

    

    predicted = model.predict(X_valid)



    results = precision_recall_fscore_support(y_valid, predicted)

    support = results[3]

    prec = np.average(results[0], weights=support)

    recall = np.average(results[1], weights=support)

    print("Iteration  #{}".format(i))

    print("=====================")

    print("Precision: {}".format(prec))

    print("Recall: {}".format(recall))

    print()

    

    avg_weighted_precisions.append(prec)

    avg_weighted_recalls.append(recall)



print("Average Precision: {}".format(np.mean(avg_weighted_precisions)))

print("Average Recall: {}".format(np.mean(avg_weighted_recalls)))
pylab.subplot(121)

pylab.xlabel('Precision')

pylab.ylabel('Value')

pylab.hist(avg_weighted_precisions, bins=20)



pylab.subplot(122)

pylab.xlabel('Recall')

pylab.hist(avg_weighted_recalls, bins=20)