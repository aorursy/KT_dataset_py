# Setup



import numpy as np

import pandas as pd



metadata = pd.read_csv('../input/ted-talks/ted_main.csv')
# Transform views

metadata['log_views'] = np.log2(metadata['views'])
# Get timedeltas from the end point

from datetime import datetime



end_date = datetime(year=2017, month=9, day=23)

metadata['days_online'] = end_date - metadata['published_date'].map(datetime.fromtimestamp)

metadata['days_online'] = metadata['days_online'].map(lambda x: x.days)
# Turns tag strings into lists

metadata['tags'] = metadata['tags'].map(eval)
# Get full tag list

tag_set = set()

for tags in metadata['tags']:

    tag_set |= set(tags)
# Add feature columns

for tag in tag_set:

    metadata[tag] = metadata['tags'].map(lambda x: int(tag in x))



metadata[['tags', 'children', 'cars']].head()
# Get number of related talks

metadata['related_number'] = metadata['related_talks'].map(lambda x: len(eval(x)))



metadata['related_number'].hist()
# Full related title list

related_titles = []

for related in metadata['related_talks']:

    related_titles += [x['title'] for x in eval(related)]

metadata['referrals'] = metadata['title'].map(lambda x: related_titles.count(x))

metadata['referrals'].hist()
# Transform and handle zeroes in the referrals column

metadata['log_referrals'] = np.log2(metadata['referrals'])

metadata.loc[metadata['log_referrals'] == -1 * np.inf, 'log_referrals'] = -1



metadata['log_referrals'].hist()
# Make sure all talks have at least 1 language ;)

metadata.loc[metadata['languages'] == 0, 'languages'] = 1
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



for column in ['languages', 'days_online', 'referrals']:

    metadata[f's_{column}'] = scaler.fit_transform(metadata[[column]])
scaler.inverse_transform([0.5])
# Create the training/testing dataframes

from sklearn.model_selection import train_test_split



X = metadata[['s_languages', 's_days_online', 's_referrals'] + list(tag_set)]

y = metadata['log_views']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)



print(X_train.shape)

print(y_train.shape)
# Set up architecture

from keras import models

from keras import layers



def build_model(X_train, y_train, *, epochs, batch_size, verbose=0, **kwargs):

    start_width = 20 #X_train.shape[1]

    nn = models.Sequential()

    nn.add(layers.Dense(start_width, kernel_initializer='normal', activation='relu', input_shape=(X_train.shape[1],)))

    #nn.add(layers.Dense(start_width//2, kernel_initializer='normal', activation='relu'))

    #nn.add(layers.Dense(start_width//4, kernel_initializer='normal', activation='relu'))

    nn.add(layers.Dense(1, kernel_initializer='normal'))

    nn.compile(optimizer='adam', loss='mean_squared_error')

    nn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    return nn



#model = build_model(epochs=100, batch_size=40, verbose=1)

#model.evaluate(X_test, y_test, verbose=1)
# Do a simple grid search for training length and batch size



def evaluate_model(cv=False, **kwargs):

    if cv:

        _X_train, _X_test, _y_train, _y_test = train_test_split(X, y, test_size=0.25)

    else:

        _X_train, _X_test, _y_train, _y_test = (X_train, X_test, y_train, y_test)

    model = build_model(_X_train, _y_train, **kwargs)

    result = model.evaluate(_X_test, _y_test, verbose=1)

    print(result, kwargs)

    return (result, kwargs)



tested_models = []

for epochs in range(80, 151, 10):

    for batch_size in range(20, 101, 20):

        tested_models.append(evaluate_model(epochs=epochs, batch_size=batch_size))



best = sorted(tested_models)[0]

print(f'Lowest error: {best[0]}')

print('Parameters: epochs = {epochs}, batch = {batch_size}'.format(**best[1]))
all_scores = np.array([x[0] for x in tested_models])

all_epochs = np.array([x[1]['epochs'] for x in tested_models])

all_sizes = np.array([x[1]['batch_size'] for x in tested_models])



import matplotlib.pyplot as plt

plt.scatter(all_sizes/all_epochs, all_scores)
tested_models2 = []

for epochs in range(80, 501, 20):

    batch_size = int(epochs * 0.35)

    tested_models2.append(evaluate_model(epochs=epochs, batch_size=batch_size, cv=True))
all_scores2 = np.array([x[0] for x in tested_models2])

all_epochs2 = np.array([x[1]['epochs'] for x in tested_models2])

plt.scatter(all_epochs2, all_scores2)
long_training = []

short_training = []

longer_training = []



for x in range(10):

    short_training.append(evaluate_model(epochs=80, batch_size=28, cv=True)[0])

    long_training.append(evaluate_model(epochs=400, batch_size=140, cv=True)[0])

    longer_training.append(evaluate_model(epochs=480, batch_size=168, cv=True)[0])

plt.boxplot([short_training, long_training, longer_training], labels=['80ep', '400ep', '480ep'])