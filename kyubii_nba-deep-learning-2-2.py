import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from pandas import plotting

from scipy import stats

plt.style.use("ggplot")

import warnings

warnings.filterwarnings("ignore")

from scipy import stats



sns.set(style="darkgrid", color_codes = True)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer, StandardScaler

import keras

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout
players = pd.read_csv('../input/nba-players-stats/Players.csv')

players.drop(columns = ['Unnamed: 0'], axis = 'column', inplace = True)

players.head()
players_data = pd.read_csv('../input/nba-players-stats/player_data.csv')

players_data.rename(columns={'name': 'Player'}, inplace=True)

players_data.fillna('No College')

players_data.head()
final_df = pd.read_csv('../input/nba-players-stats/Seasons_Stats.csv')

final_df.drop(columns = ['Unnamed: 0'], axis = 'column', inplace = True)

final_df.fillna(0, inplace=True)

final_df.head()
final_df['ppg'] = final_df.PTS/final_df.G

final_df['apg'] = final_df.AST/final_df.G

final_df['rpg'] = final_df.TRB/final_df.G

final_df['tpg'] = final_df.TOV/final_df.G
train = final_df.drop(['Player','Pos','blanl', 'blank2', 'Tm'], axis="columns")

train.head()
X = train.drop(['ppg'], axis=1)

y = train['ppg']



X.fillna(0, inplace=True)

y.fillna(0, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



# Choose the type of regressor. 

RFR = RandomForestRegressor()



# Choose some parameter combinations to try

#YOU CAN TRY DIFFERENTS PARAMETERS TO FIND THE BEST MODEL

parameters = {'n_estimators': [5, 10, 100],

              #'criterion': ['mse'],

              #'max_depth': [5, 10, 15], 

              #'min_samples_split': [2, 5, 10],

              'min_samples_leaf': [1,3]

             }



grid_obj = GridSearchCV(RFR, parameters,

                        cv=5,

                        n_jobs=-1,

                        verbose=1)



grid_obj = grid_obj.fit(X_train, y_train)
RFR = grid_obj.best_estimator_

RFR.fit(X_train, y_train)
from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error



predictions = RFR.predict(X_val)

y_validation_RF = y_val.reset_index()['ppg']



print('R2 score = ',r2_score(y_validation_RF, predictions), '/ 1.0')

print('MSE score = ',mean_squared_error(y_validation_RF, predictions), '/ 0.0')
# test data predictions

predictions = RFR.predict(X_test)



y_test_RF = y_test.reset_index()['ppg']



print('R2 score = ',r2_score(y_test_RF, predictions), '/ 1.0')

print('MSE score = ',mean_squared_error(y_test_RF, predictions), '/ 0.0')
plt.plot(y_validation_RF[0:50], '+', color ='blue', alpha=0.7)

plt.plot(predictions[0:50], 'ro', color ='red', alpha=0.5)

plt.title('Prediction vs Real values')

plt.show()
plt.plot(y_test_RF[0:50], '+', color ='blue', alpha=0.7)

plt.plot(predictions[0:50], 'ro', color ='red', alpha=0.5)

plt.title('Prediction vs Real values')

plt.show()
# Convert data as np.array

features = np.array(X_train)

targets = np.array(y_train)



features_validation= np.array(X_val)

targets_validation = np.array(y_val)



features_test= np.array(X_test)

targets_test = np.array(y_test)
print(features.shape)

print(targets.shape)
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation



# Building the model

model = Sequential()

model.add(Dense(200, input_dim=50, kernel_initializer='normal', activation='relu'))

model.add(Dense(100, kernel_initializer='normal', activation='relu'))

model.add(Dense(50, kernel_initializer='normal', activation='relu'))

model.add(Dense(25, kernel_initializer='normal', activation='relu'))

model.add(Dense(1, kernel_initializer='normal'))



# Compiling the model

model.compile(loss = 'mse', optimizer='adam', metrics=['mse']) #mse: mean_square_error

model.summary()
# Training the model

epochs_tot = 1000

epochs_step = 100

epochs_ratio = int(epochs_tot / epochs_step)

hist =np.array([])



for i in range(epochs_ratio):

    history = model.fit(features, targets, epochs=epochs_step, batch_size=100, verbose=0)

    

    # Evaluating the model on the training and testing set

    print("Step : " , i * epochs_step, "/", epochs_tot)

    score = model.evaluate(features, targets)

    print("Training MSE:", score[1])

    score = model.evaluate(features_validation, targets_validation)

    print("Validation MSE:", score[1], "\n")

    hist = np.concatenate((hist, np.array(history.history['mse'])), axis = 0)

    

# plot metrics

plt.plot(hist)

plt.show()
#prediction and error checking

predictions = model.predict(features_validation, verbose=0)



print('R2 score = ',r2_score(y_val, predictions), '/ 1.0')

print('MSE score = ',mean_squared_error(y_val, predictions), '/ 0.0')
plt.plot(y_val.reset_index()['ppg'][0:50], '+', color ='blue', alpha=0.7)

plt.plot(predictions[0:50], 'ro', color ='red', alpha=0.5)

plt.title('Prediction vs Real values')

plt.show()
predictions = model.predict(features_test, verbose=0)



print('R2 score = ',r2_score(y_test, predictions), '/ 1.0')

print('MSE score = ',mean_squared_error(y_test, predictions), '/ 0.0')
plt.plot(y_test.reset_index()['ppg'][0:50], '+', color ='blue', alpha=0.7)

plt.plot(predictions[0:50], 'ro', color ='red', alpha=0.5)

plt.title('Prediction vs Real values')

plt.show()