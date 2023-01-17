from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std
model = models.Sequential()
model.add(layers.Dense(64, activation='relu',
                        input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
# track histories across training sessions
histories = []
histories.append(
    model.fit(train_data, train_targets,
              batch_size=2, epochs=20, verbose=0))
def plot_histories(histories):
    plt.clf()
    mae = []
    for history in histories:
        for error in history.history['mean_absolute_error']:
            mae.append(error)
    epochs = range(1, len(mae) + 1)

    plt.plot(epochs, mae, 'b', label='Training mae')
    plt.title('Training mean absolute error')
    plt.legend()
    plt.show()
plot_histories(histories)
# continue training
histories.append(
    model.fit(train_data, train_targets,
              batch_size=2, epochs=20, verbose=0))
len(histories)
plot_histories(histories)