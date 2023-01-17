import numpy as np

import pandas as pd

import sklearn

import matplotlib.pyplot as plt
from sklearn.datasets import load_boston



boston = load_boston()

boston.keys()
print(boston.DESCR)
boston_df = pd.DataFrame(np.append(np.reshape(boston.target, (boston.target.shape[0],1)), boston.data, axis = 1),

                         columns = np.append('PRICE', boston.feature_names))



boston_df.head()
boston_df.describe()
X = boston_df.drop('PRICE', axis = 1)

y = boston_df['PRICE']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
X_train
from sklearn.linear_model import LinearRegression



lm = LinearRegression()

lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)



plt.scatter(y_test, y_pred)

plt.xlabel("Prices: $y_i$")

plt.ylabel("Predicted prices: $\hat{y}_i$")

plt.title("Prices vs Predicted prices: $y_i$ vs $\hat{y}_i$")

plt.show()
sklearn.metrics.r2_score(y_test, y_pred) 
sklearn.metrics.mean_absolute_error(y_test, y_pred)
import statsmodels.api as sm



m = sm.OLS(y_train,sm.add_constant(X_train)).fit()

m.summary()
y_pred = m.predict(sm.add_constant(X_test))
plt.scatter(y_test, y_pred)

plt.xlabel("Prices: $y_i$")

plt.ylabel("Predicted prices: $\hat{y}_i$")

plt.title("Prices vs Predicted prices: $y_i$ vs $\hat{y}_i$")

plt.show()
from keras.datasets import boston_housing



(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()
mean = train_data.mean(axis=0)

train_data -= mean

std = train_data.std(axis=0)

train_data /= std



test_data -= mean

test_data /= std
from keras import models

from keras import layers



def build_model():

    # Because we will need to instantiate

    # the same model multiple times,

    # we use a function to construct it.

    model = models.Sequential()

    model.add(layers.Dense(64, activation='relu',

                           input_shape=(train_data.shape[1],)))

    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model
mymodel = build_model()

mymodel.summary()
mymodel.fit(train_data, train_targets,epochs=10, batch_size=1, verbose=1)
mymodel.evaluate(test_data, test_targets, verbose=1)
from keras import backend as K



# Some memory clean-up

K.clear_session()
import numpy as np



k = 4

num_val_samples = len(train_data) // k

num_epochs = 50



num_epochs = 150

all_mae_histories = []

for i in range(k):

    print('processing fold #', i)

    # Prepare the validation data: data from partition # k

    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]

    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]



    # Prepare the training data: data from all other partitions

    partial_train_data = np.concatenate(

        [train_data[:i * num_val_samples],

         train_data[(i + 1) * num_val_samples:]],

        axis=0)

    partial_train_targets = np.concatenate(

        [train_targets[:i * num_val_samples],

         train_targets[(i + 1) * num_val_samples:]],

        axis=0)



    # Build the Keras model (already compiled)

    model = build_model()

    # Train the model (in silent mode, verbose=0)

    history = model.fit(partial_train_data, partial_train_targets,

                        validation_data=(val_data, val_targets),

                        epochs=num_epochs, batch_size=1, verbose=1)

    mae_history = history.history['val_mae']

    all_mae_histories.append(mae_history)
average_mae_history = [

    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
import matplotlib.pyplot as plt



plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)

plt.xlabel('Epochs')

plt.ylabel('Validation MAE')

plt.show()
def smooth_curve(points, factor=0.9):

  smoothed_points = []

  for point in points:

    if smoothed_points:

      previous = smoothed_points[-1]

      smoothed_points.append(previous * factor + point * (1 - factor))

    else:

      smoothed_points.append(point)

  return smoothed_points



smooth_mae_history = smooth_curve(average_mae_history[10:])



plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)

plt.xlabel('Epochs')

plt.ylabel('Validation MAE')

plt.show()
# Get a fresh, compiled model.

model = build_model()

# Train it on the entirety of the data.

model.fit(train_data, train_targets,

          epochs=80, batch_size=16, verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
test_mae_score