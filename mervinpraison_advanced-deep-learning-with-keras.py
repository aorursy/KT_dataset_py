# Import Input from keras.layers

from keras.layers import Input



# Create an input layer of shape 1

input_tensor = Input(shape=(1,))
# Load layers

from keras.layers import Input, Dense



# Input layer

input_tensor = Input(shape=(1,))



# Dense layer

output_layer = Dense(1)



# Connect the dense layer to the input_tensor

output_tensor = output_layer(input_tensor)
# Load layers

from keras.layers import Input, Dense



# Input layer

input_tensor = Input(shape=(1,))



# Create a dense layer and connect the dense layer to the input_tensor in one step

# Note that we did this in 2 steps in the previous exercise, but are doing it in one step now

output_tensor = Dense(1)(input_tensor)
# Input/dense/output layers

from keras.layers import Input, Dense

input_tensor = Input(shape=(1,))

output_tensor = Dense(1)(input_tensor)



# Build the model

from keras.models import Model

model = Model(input_tensor, output_tensor)
# Compile the model

model.compile(optimizer='adam', loss='mean_absolute_error')
# Import the plotting function

from keras.utils import plot_model

import matplotlib.pyplot as plt



# Summarize the model

model.summary()



# # Plot the model

plot_model(model, to_file='model.png')



# # Display the image

data = plt.imread('model.png')

plt.imshow(data)

plt.show()
import pandas as pd



games_tourney_train = pd.read_csv('../input/games-tourney/games_tourney.csv')

games_tourney_train.head()
games_tourney_train.shape
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(games_tourney_train['seed_diff'], games_tourney_train['score_diff'])
# Now fit the model

model.fit(X_train, y_train,

          epochs=1,

          batch_size=128,

          validation_split=.10,

          verbose=True)
# Evaluate the model on the test data

model.evaluate(X_test, y_test)
games_season = pd.read_csv('../input/games-season/games_season.csv')

games_season.head()
# Imports

from keras.layers import Embedding

from numpy import unique



# Count the unique number of teams

n_teams = unique(games_season['team_1']).shape[0]



# Create an embedding layer

team_lookup = Embedding(input_dim=n_teams,

                        output_dim=1,

                        input_length=1,

                        name='Team-Strength')
# Imports

from keras.layers import Input, Embedding, Flatten

from keras.models import Model



# Create an input layer for the team ID

teamid_in = Input(shape=(1,))



# Lookup the input in the team strength embedding layer

strength_lookup = team_lookup(teamid_in)



# Flatten the output

strength_lookup_flat = Flatten()(strength_lookup)



# Combine the operations into a single, re-usable model

team_strength_model = Model(teamid_in, strength_lookup_flat, name='Team-Strength-Model')
# Load the input layer from keras.layers

from keras.layers import Input



# Input layer for team 1

team_in_1 = Input((1,), name='Team-1-In')



# Separate input layer for team 2

team_in_2 = Input((1,), name='Team-2-In')
# Lookup team 1 in the team strength model

team_1_strength = team_strength_model(team_in_1)



# Lookup team 2 in the team strength model

team_2_strength = team_strength_model(team_in_2)
# Import the Subtract layer from keras

from keras.layers import Subtract



# Create a subtract layer using the inputs from the previous exercise

score_diff = Subtract()([team_1_strength, team_2_strength])
# Imports

from keras.layers import Subtract

from keras.models import Model



# Subtraction layer from previous exercise

score_diff = Subtract()([team_1_strength, team_2_strength])



# Create the model

model = Model([team_in_1, team_in_2], score_diff)



# Compile the model

model.compile(optimizer='adam', loss='mean_absolute_error')
model.summary()
# Get the team_1 column from the regular season data

input_1 = games_season['team_1']



# Get the team_2 column from the regular season data

input_2 = games_season['team_2']



# Fit the model to input 1 and 2, using score diff as a target

model.fit([input_1, input_2],

          games_season['score_diff'],

          epochs=1,

          batch_size=2048,

          validation_split=.1,

          verbose=True)
games_tourney = pd.read_csv('../input/games-tourney/games_tourney.csv')

games_tourney.head()
# Get team_1 from the tournament data

input_1 = games_tourney['team_1']



# Get team_2 from the tournament data

input_2 = games_tourney['team_2']



# Evaluate the model using these inputs

model.evaluate([input_1, input_2], games_tourney['score_diff'])
from keras.layers.merge import Concatenate
# Create an Input for each team

team_in_1 = Input(shape=(1,), name='Team-1-In')

team_in_2 = Input(shape=(1,), name='Team-2-In')



# Create an input for home vs away

home_in = Input(shape=(1,), name='Home-In')



# Lookup the team inputs in the team strength model

team_1_strength = team_strength_model(team_in_1)

team_2_strength = team_strength_model(team_in_2)



# Combine the team strengths with the home input using a Concatenate layer, then add a Dense layer

out = Concatenate()([team_1_strength, team_2_strength, home_in])

out = Dense(1)(out)
# Import the model class

from keras.models import Model



# Make a Model

model = Model([team_in_1, team_in_2, home_in], out)



# Compile the model

model.compile(optimizer='adam', loss='mean_absolute_error')
model.summary()
# Fit the model to the games_season dataset

model.fit([games_season['team_1'], games_season['team_2'], games_season['home']],

          games_season['score_diff'],

          epochs=1,

          verbose=True,

          validation_split=.1,

          batch_size=2048)



# Evaluate the model on the games_tourney dataset

model.evaluate([games_tourney['team_1'], games_tourney['team_2'], games_tourney['home']], games_tourney['score_diff'])
# Imports

import matplotlib.pyplot as plt

from keras.utils import plot_model



# Plot the model

plot_model(model, to_file='model.png')



# Display the image

data = plt.imread('model.png')

plt.imshow(data)

plt.show()
# Predict

games_tourney['pred'] = model.predict([games_tourney['team_1'],games_tourney['team_2'],games_tourney['home']])
games_tourney.head()
games_tourney.score_diff.unique()
# Create an input layer with 3 columns

input_tensor = Input((3,))



# Pass it to a Dense layer with 1 unit

output_tensor = Dense(1)(input_tensor)



# Create a model

model = Model(input_tensor, output_tensor)



# Compile the model

model.compile(optimizer='adam', loss='mean_absolute_error')
games_tourney_train = games_tourney.query('season < 2010')
games_tourney_train.shape
# Fit the model

model.fit(games_tourney_train[['home', 'seed_diff', 'pred']],

          games_tourney_train['score_diff'],

          epochs=1,

          verbose=True)
games_tourney_test = games_tourney.query('season >= 2010')
games_tourney_test.shape
# Evaluate the model on the games_tourney_test dataset

model.evaluate(games_tourney_test[['home', 'seed_diff', 'pred']], 

               games_tourney_test['score_diff'])
# Define the input

input_tensor = Input((2,))



# Define the output

output_tensor = Dense(2)(input_tensor)



# Create a model

model = Model(input_tensor, output_tensor)



# Compile the model

model.compile(optimizer='adam', loss='mean_absolute_error')
games_tourney_train.shape
# Fit the model

model.fit(games_tourney_train[['seed_diff', 'pred']],

  		  games_tourney_train[['score_1', 'score_2']],

  		  verbose=False,

  		  epochs=1000,

  		  batch_size=64)
import numpy as np
np.mean(model.history.history['loss'])
# Print the model's weights

print(model.get_weights())



# Print the column means of the training data

print(games_tourney_train.mean())
# Evaluate the model on the tournament test data

model.evaluate(games_tourney_test[['seed_diff', 'pred']],games_tourney_test[['score_1', 'score_2']])
# Create an input layer with 2 columns

input_tensor = Input((2,))



# Create the first output

output_tensor_1 = Dense(1, activation='linear', use_bias=False)(input_tensor)



# Create the second output (use the first output as input here)

output_tensor_2 = Dense(1, activation='sigmoid', use_bias=False)(output_tensor_1)



# Create a model with 2 outputs

model = Model(input_tensor, [output_tensor_1, output_tensor_2])
# Import the Adam optimizer

from keras.optimizers import Adam



# Compile the model with 2 losses and the Adam optimzer with a higher learning rate

model.compile(loss=['mean_absolute_error', 'binary_crossentropy'], optimizer=Adam(lr=0.01))



# Fit the model to the tournament training data, with 2 inputs and 2 outputs

model.fit(games_tourney_train[['seed_diff', 'pred']],

          [games_tourney_train[['score_diff']], games_tourney_train[['won']]],

          epochs=10,

          verbose=True,

          batch_size=16384)
# Print the model weights

print(model.get_weights())



# Print the training data means

print(games_tourney_train.mean())
# Import the sigmoid function from scipy

from scipy.special import expit as sigmoid



# Weight from the model

weight = 0.14



# Print the approximate win probability predicted close game

print(sigmoid(1 * weight))



# Print the approximate win probability predicted blowout game

print(sigmoid(10 * weight))
# Evaluate the model on new data

model.evaluate(games_tourney_test[['seed_diff', 'pred']],

               [games_tourney_test[['score_diff']], games_tourney_test[['won']]])