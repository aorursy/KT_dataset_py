import numpy as np

import pandas as pd

import tensorflow as tf
#### Download the concrete data

concrete_data = pd.read_csv('../input/us-concrete-data/concrete_data.csv')

print(concrete_data.shape)

print(concrete_data.head()) # By the way, unit is 'cubic meter' and days old of concrete mix, and unit of strength is MPa.

concrete_data.describe()
concrete_data.isnull().sum() # Looks very clean data.
df = concrete_data

cols = df.columns

X = df[cols[cols != 'Strength']]

y = df['Strength']
# To get reproducible results I'm setting random seed

np.random.seed(1)

tf.random.set_seed(1)

import keras

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from keras.models import Sequential

from keras.layers import Dense

test_size = 0.3

def random_data_split(X, y, seed):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    return X_train, X_test, y_train, y_test
# baseline model (One hidden layer 10 nodes, 50 epochs)

mse_list = []

predicted_list = {}

def create_baseline_model():

    baseline_model = Sequential()

    baseline_model.add(Dense(10, activation='relu', input_shape=(X.shape[1],)))

    baseline_model.add(Dense(10, activation='relu'))

    baseline_model.add(Dense(1))

    baseline_model.compile(optimizer='adam', loss='mean_squared_error')

    return baseline_model



# collect 50 mse values.

for i in range(50):

    if (i + 1) % 10 == 0:

        print('Now {} times calculating.'.format(i + 1))

    model = create_baseline_model()

    X_train, X_test, y_train, y_test = random_data_split(X, y, i)

    model.fit(X_train, y_train, epochs=50, verbose=0)

    y_hats = model.predict(X_test)

    mse = mean_squared_error(y_test, y_hats)

    mse_list.append(mse)

    predicted_list[i] = {'y_test': y_test, 'y_hats': y_hats}



# Calculate mean and standard deviation of mse values

mse_mean = np.mean(mse_list)

mse_std = np.std(mse_list)

print('Mean value of MSE:{:.2f}, Standard Deviation value of MSE:{:.2f}.'.format(mse_mean, mse_std), 'First three MSE values:', mse_list[0:3])

print('Acctual Value samples', predicted_list[0]['y_test'].values[0:3], 'Predicted Value samples', np.around(predicted_list[0]['y_hats'].flatten()[0:3], decimals=2))
# In this time, I'm not doing any Normalization. So I'll Normalize continuous values (by subtracting the mean from the individual predictors and dividing by the standard deviation).

X_norm = (X - X.mean()) / X.std()

X_norm.head(3)
# Now I'm getting ready to examin how normalization can improve the baseline model (One hidden layer 10 nodes, 50 epochs)

mse_list_norm = []

predicted_list_norm = {}

# collect 50 mse values.

for i in range(50):

    if (i + 1) % 10 == 0:

        print('Now {} times calculating.'.format(i + 1))

    model = create_baseline_model()

    X_train, X_test, y_train, y_test = random_data_split(X_norm, y, i)

    model.fit(X_train, y_train, epochs=50, verbose=0)

    y_hats = model.predict(X_test)

    mse = mean_squared_error(y_test, y_hats)

    mse_list_norm.append(mse)

    predicted_list_norm[i] = {'y_test': y_test, 'y_hats': y_hats}



# Calculate mean and standard deviation of mse values

mse_mean_norm = np.mean(mse_list_norm)

mse_std_norm = np.std(mse_list_norm)

print('Mean value of MSE:{:.2f} and Standard Deviation value of MSE:{:.2f}.'.format(mse_mean_norm, mse_std_norm), 'First three MSE values:', mse_list_norm[0:3])

print('Acctual Value samples', predicted_list_norm[0]['y_test'].values[0:3], 'Predicted Value samples', np.around(predicted_list_norm[0]['y_hats'].flatten()[0:3], decimals=2))
import matplotlib.pyplot as plt

plt.figure()

fig, ((ax1), (ax2)) = plt.subplots(1, 2, sharex=True, sharey=True)

ax1.hist(mse_list, alpha=0.5, bins=20, color='r', label='baseline model')

ax2.hist(mse_list_norm, alpha=0.5, bins=5, color='b', label='After applied Normalization')

ax1.legend()

ax2.legend()

ax1.set_xlabel('Mean Squared Values of baseline model')

ax2.set_xlabel('Mean Squared Values after the Normalization')

ax1.set_ylabel('Standard Deviation value of MSE')

fig= plt.gcf()

fig.set_size_inches(10, 5.5)

plt.show()
# Then I will increase epoch values to 100 and look at how models are improved by increasing epoch.

# Now I'm getting ready to examin how increasing epochs can improve the normalized model (One hidden layer 10 nodes, 100 epochs)

epochs = 100

mse_list_double_epoch = []

predicted_list_double_epoch = {}

# collect 50 mse values.

for i in range(50):

    if (i + 1) % 10 == 0:

        print('Now {} times calculating.'.format(i + 1))

    model = create_baseline_model()

    X_train, X_test, y_train, y_test = random_data_split(X_norm, y, i)

    model.fit(X_train, y_train, epochs=epochs, verbose=0)

    y_hats = model.predict(X_test)

    mse = mean_squared_error(y_test, y_hats)

    mse_list_double_epoch.append(mse)

    predicted_list_double_epoch[i] = {'y_test': y_test, 'y_hats': y_hats}



# Calculate mean and standard deviation of mse values

mse_mean_double_epoch = np.mean(mse_list_double_epoch)

mse_std_double_epoch = np.std(mse_list_double_epoch)

print('Mean value of MSE:{:.2f} and Standard Deviation value of MSE:{:.2f}.'.format(mse_mean_double_epoch, mse_std_double_epoch), 'First three MSE values:', mse_list_double_epoch[0:3])

print('Acctual Value samples', predicted_list_double_epoch[0]['y_test'].values[0:3], 'Predicted Value samples', np.around(predicted_list_double_epoch[0]['y_hats'].flatten()[0:3], decimals=2))
# Then I will increase hidden layers to three but set epochs back to 50 same as B.

# Now I'm getting ready to examin how increasing hidden layers can improve the normalized model (Three hidden layer 10 nodes, 50 epochs)

epochs = 50

mse_list_three_layers = []

predicted_list_three_layers = {}



def create_three_layer_model():

    baseline_model = Sequential()

    baseline_model.add(Dense(10, activation='relu', input_shape=(X.shape[1],)))

    baseline_model.add(Dense(10, activation='relu'))

    baseline_model.add(Dense(10, activation='relu'))

    baseline_model.add(Dense(10, activation='relu'))

    baseline_model.add(Dense(1))

    baseline_model.compile(optimizer='adam', loss='mean_squared_error')

    return baseline_model



# collect 50 mse values.

for i in range(50):

    if (i + 1) % 10 == 0:

        print('Now {} times calculating.'.format(i + 1))

    model = create_three_layer_model()

    X_train, X_test, y_train, y_test = random_data_split(X_norm, y, i)

    model.fit(X_train, y_train, epochs=epochs, verbose=0)

    y_hats = model.predict(X_test)

    mse = mean_squared_error(y_test, y_hats)

    mse_list_three_layers.append(mse)

    predicted_list_three_layers[i] = {'y_test': y_test, 'y_hats': y_hats}



# Calculate mean and standard deviation of mse values

mse_mean_three_layers = np.mean(mse_list_three_layers)

mse_std_three_layers = np.std(mse_list_three_layers)

print('Mean value of MSE:{:.2f} and Standard Deviation value of MSE:{:.2f}.'.format(mse_mean_three_layers, mse_std_three_layers), 'First three MSE values:', mse_list_three_layers[0:3])

print('Acctual Value samples', predicted_list_three_layers[0]['y_test'].values[0:3], 'Predicted Value samples', np.around(predicted_list_three_layers[0]['y_hats'].flatten()[0:3], decimals=2))