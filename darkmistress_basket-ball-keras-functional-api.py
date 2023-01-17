import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from keras.models import Model

from keras.layers import Input, Dense
games_tourney = pd.read_csv('../input/basketball-data/games_tourney.csv')

games_tourney.head()
input_tensor = Input(shape=(1,))

output_tensor = Dense(1)(input_tensor)

model = Model(input_tensor, output_tensor)

model.compile(optimizer='adam', loss='mae')
# Now fit the model

model.fit(games_tourney['seed_diff'], games_tourney['score_diff'],

          epochs=1,

          batch_size=128,

          validation_split=0.1,

          verbose=True)
# Load the X variable from the test data

X_test = games_tourney['seed_diff']



# Load the y variable from the test data

y_test = games_tourney['score_diff']



# Evaluate the model on the test data

print(model.evaluate(X_test, y_test, verbose=False))