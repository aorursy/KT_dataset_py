import pandas as pd

import numpy as np
# If you simply read the songs.csv file, you'll find 2296320 songs.

# If you loop through the file and extract ids, you'll find 513 more.

song_ids_pandas = pd.read_csv('../input/songs.csv', usecols=['song_id'])

print('Song ids from reading with pandas = %d' % len(song_ids_pandas))

print('Unique song ids from reading with pandas = %d' % len(set(song_ids_pandas.song_id)))



f = open('../input/songs.csv'); next(f)

song_ids_manual = []

for l in f:

    song_ids_manual.append(l.split(',')[0])

    assert song_ids_manual[-1].endswith('=')

print('Song ids from reading manually = %d' % len(song_ids_manual))

print('Unique song ids from reading manually = %d' % len(set(song_ids_manual)))



print('Missing song ids = %d' % (len(song_ids_manual) - len(song_ids_pandas)))
# How often do the missing song ids occur in the training and test sets?

# It's not a huge number of records, but still something.

song_ids_missing = set(song_ids_manual) - set(song_ids_pandas.song_id.values)

song_ids_trn = pd.read_csv('../input/train.csv', usecols=['song_id'])

song_ids_tst = pd.read_csv('../input/test.csv', usecols=['song_id'])

is_missing = lambda id_: id_ in song_ids_missing

nb_missing_trn = song_ids_trn.song_id.apply(is_missing).sum()

nb_missing_tst = song_ids_tst.song_id.apply(is_missing).sum()

print('Missing in train set = %d' % nb_missing_trn)

print('Missing in test set = %d' % nb_missing_tst)
# So there are missing ids which is likely indicates formatting errors in the CSV.

# How do you fix this? 



# 1. Do pair-wise comparisons of ids to find the first erroneous line.

# 2. Fix the erroneous line manually.

# 3. Repeat.



# Luckily this only has to be done four times. I'm not sure how to do this in the notebook,

# so I did it offline and uploaded the fixed file to dropbox.



# Below is the first error:

for i, (a,b) in enumerate(zip(song_ids_manual, song_ids_pandas.song_id)):

    if a != b:

        print('Error on line %d' % (i + 1))

        print('%s != %s' % (a, b))

        break