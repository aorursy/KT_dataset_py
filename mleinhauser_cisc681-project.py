import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.callbacks import EarlyStopping

from keras.utils import to_categorical

from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Read in the data and turn the csv files into DataFrames

test_set = pd.read_csv("/kaggle/input/ai-project-v2/test_set.csv")

training_set = pd.read_csv("/kaggle/input/ai-project-v2/training_set.csv")



# Reset the index of the training set just to be it did not change

training_set.reset_index(inplace = True)



# Make sure the training set loaded correctly

training_set.head()
# Clean up the MADE_PLAYOFFS column from categorical data to numerical data

cleanup = {'MADE_PLAYOFFS': {'Y': 1, 'N': 0}}

training_set.replace(cleanup, inplace = True)

test_set.replace(cleanup, inplace = True)



# Check to see the values were replaced

training_set.head()
# Drop Made Playoffs column from training set and test set because that is what we are trying to predict

train_X = training_set.drop(columns=['MADE_PLAYOFFS'])

test_X = test_set.drop(columns=['MADE_PLAYOFFS'])



# The categorical features in this data do not add (or remove) any value so we only focus on the numerical data.



numeric_features_list = ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 

                 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'OMP', 'OFG', 'OFGA', 'OFG%', 'O3P', 'O3PA', 'O3P%', 

                 'O2P', 'O2PA', 'O2P%', 'OFT', 'OFTA', 'OFT%', 'OORB', 'ODRB', 'OTRB', 'OAST', 'OSTL', 'OBLK', 'OTOV', 'OPTS']

x_data = train_X[numeric_features_list].to_numpy()

# Verify the MADE_PLAYOFFS column was dropped from the training set

train_X.head()
train_Y = training_set[['MADE_PLAYOFFS']]

y_data = train_Y[['MADE_PLAYOFFS']].to_numpy()

y_data = y_data.reshape(y_data.shape[0], )
rf_regressor = RandomForestRegressor(n_estimators=200)

rf_regressor.fit(x_data, y_data)
sorted_data_significance = np.argsort(rf_regressor.feature_importances_)[::-1]

for index in sorted_data_significance:

    print(f"{numeric_features_list[index]}: {rf_regressor.feature_importances_[index]}")
feature_selector = SelectKBest(mutual_info_regression, k = 5)

best_feature = feature_selector.fit_transform(x_data, y_data)

sorted_indices = np.argsort(feature_selector.scores_)[::-1]
for index in sorted_indices:

    print(f"{numeric_features_list[index]}: {feature_selector.scores_[index]:.4f}")
# Create a correlation matrix of the features

data = training_set[numeric_features_list + ["MADE_PLAYOFFS"]]

figure = plt.figure(figsize=(40, 40))

figure.set_facecolor('white')

sns.heatmap(data.corr(), annot=True, cmap="RdYlGn")
# Create model using keras

model = Sequential()



# Get the number of columns in the training data

n_cols = x_data.shape[1]



# Add layers to the model

model.add(Dense(20, activation='relu', input_shape=(n_cols,)))

model.add(Dense(10, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
# Compile the model using binary cross entropy and the adam optimizer

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model using the early stopping monitor so we won't waste any effort trying to improve the model 

# when it won't improve anymore.

# If the model does not improve, we will stop training.

early_stopping_monitor = EarlyStopping(patience = 3)



# Train the model

model.fit(x_data, y_data, validation_split = 0.2, epochs = 100, callbacks=[early_stopping_monitor])



# Training the model on x_data (which only contains numerical data) and comparing it to test_X which has 

# every column except for MADE_PLAYOFFS.

test_X = test_X[numeric_features_list].to_numpy()
# Evaluate the model's accuracy on the training set

_, train_accuracy = model.evaluate(x_data, y_data)

print("Accuracy: %.3f" % (train_accuracy * 100) + "%")
# Make predictions

test_y_predictions = model.predict(test_X)

rounded_predictions = [round(index[0]) for index in test_y_predictions]
# Print our predictions

for index in range(len(test_y_predictions)):

    print(f"{test_set['Season'][index]} {test_set['Team'][index]}: {int(rounded_predictions[index])} Expected: {test_set['MADE_PLAYOFFS'][index]}")