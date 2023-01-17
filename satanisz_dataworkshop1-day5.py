import pandas as pd

import numpy as np



from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score





%matplotlib inline
#df = pd.read_csv('https://raw.githubusercontent.com/aczepielik/KRKtram/master/reports/report_07-23.csv')

df = pd.read_csv('../input/report_07-23.csv')

df.head()
df[df.tripId == 6351558574044883205]
#df.delay.value_counts()

df.delay.value_counts(normalize=True)
df.delay.hist(bins=15);
df.delay.describe()
df.columns
X = df[['number']].values

y = df['delay'].values



model = DecisionTreeRegressor(max_depth=10)

scores = cross_val_score(model, X, y, cv=4, scoring = 'neg_mean_absolute_error')

scores
np.std(scores)
# faktoryzacja dwóch zmiennych:

pd.factorize( [ 

    '{} {}'.format(7, 'A'),

    '{} {}'.format(7, 'B'),

    '{} {}'.format(10, 'C'),

    '{} {}'.format(10, 'D')

])[0]
# przejdzmy jednak na sekundy i dodajmy stop, direction, vehicleID, 'seq_num'

df['delay_secs'] = df['delay'].map(lambda x: x*60)

df['direction_cat'] = df['direction'].factorize()[0]

df['vehicleId'].fillna(-1, inplace=True)

df['seq_num'].fillna(-1, inplace=True)

# UWAGA: proces tworzenia to feature engineering



#df.apply(lambda x : x['number'] x['direction'], axis=1)

df['number_direction_id'] = df.apply(lambda x : '{} {}'.format(x['number'], x['direction']), axis=1).factorize()[0]

df['stop_direction_id'] = df.apply(lambda x : '{} {}'.format(x['stop'], x['direction']), axis=1).factorize()[0]



feats1=['number', 'stop', 'direction_cat', 'seq_num']

feats2=['number', 

        'stop', 

        'direction_cat', 

        'seq_num',

        'number_direction_id',

        'stop_direction_id'

       ]



X = df[feats2].values

y = df['delay_secs'].values

model = RandomForestRegressor(max_depth=10, n_estimators=40, n_jobs=3) #n_jobs - ile rdzeni

model_tree = DecisionTreeRegressor(max_depth=10, random_state=0) # random_state urzywa liczb pseudolosowych

scores = cross_val_score(model, X, y, cv=6, scoring = 'neg_mean_absolute_error')

np.mean(scores), np.std(scores)