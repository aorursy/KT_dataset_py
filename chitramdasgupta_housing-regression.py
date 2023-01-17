import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_style('dark')
import sklearn
import tensorflow as tf
from tensorflow import keras
data_path = '../input/housing/housing.csv'

df = pd.read_csv(data_path)
df.head()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.head()
df.info()
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].mean())
df.info()
all_labels = df['median_house_value']
all_data = df.drop('median_house_value', axis=1)
all_data['ocean_proximity'].value_counts()
temp = pd.get_dummies(all_data['ocean_proximity'])
all_data = pd.concat([all_data, temp], axis=1, ignore_index=False)
all_data = all_data.drop('ocean_proximity', axis=1)
all_data.head()
all_data = (all_data - all_data.mean())/all_data.std()
all_data.head()
all_data = all_data.values
all_data[0]
all_labels = all_labels.values
all_labels[0]
train_size = int((80/100) * all_data.shape[0])

train_data = all_data[: train_size]
train_labels = all_labels[: train_size]

test_data = all_data[train_size: ]
test_labels = all_labels[train_size: ]

assert(len(train_data) == len(train_labels))
assert(len(test_data) == len(test_labels))
print(test_data[0])
print(test_labels[0])

assert(len(train_data[0]) == len(test_data[0]))
def create_model(optimizer='adam', activation='relu'):

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(128, activation=activation))
    model.add(keras.layers.Dense(64, activation=activation))
    model.add(keras.layers.Dense(1))
    
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    return model
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor


model = KerasRegressor(build_fn=create_model, epochs=60 , verbose=0)

optimizer = ['rmsprop', 'adam']
activation = ['relu', 'elu']
param_grid = dict(optimizer=optimizer, activation=activation)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, verbose=2)

grid_result = grid.fit(train_data, train_labels)
print(grid_result.best_score_)
print(grid_result.best_params_)
best_activation = grid_result.best_params_['activation']
best_optimizer = grid_result.best_params_['optimizer']
num_epochs = 500
all_mae_histories = []
all_scores = []
k = 4
num_val_samples = len(train_data) // k

for i in range(k):
    
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_labels[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]],
        axis=0)
    
    partial_train_targets = np.concatenate(
        [train_labels[:i * num_val_samples],
        train_labels[(i + 1) * num_val_samples:]],
        axis=0)

    model = create_model(optimizer='adam', activation='elu')
    
    my_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    
    history = model.fit(partial_train_data, partial_train_targets,
                            validation_data=(val_data, val_targets),
                            epochs=num_epochs, batch_size=10, verbose=1, callbacks=[my_cb])
    

    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)
    
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=1)
    all_scores.append(val_mae)
print(all_scores)
print(np.mean(all_scores))
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(308)]
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.xticks(np.arange(0, 300, 5))
plt.tight_layout()
# According to this plot, validation MAE stops improving significantly after 50 
# epochs. Past that point, you start overfitting.
# We now train the final model with epochs = 5

model = create_model(optimizer='adam', activation='elu')

my_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(train_data, train_labels,
    epochs=500, validation_split=0.2, batch_size=10, verbose=1, callbacks=[my_cb])
history.history.keys()
num_epochs = len(history.history['loss'])

x = np.arange(1, num_epochs+1)
y1 = history.history['loss']
y2 = history.history['val_loss']

plt.plot(x, y1, y2)
plt.legend(['loss', 'val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tight_layout()
x = np.arange(1, num_epochs+1)
y1 = history.history['mae']
y2 = history.history['val_mae']

plt.plot(x, y1, y2)
plt.legend(['mae', 'val_mae'])
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.tight_layout()
print(test_data[0])
print(test_labels[0])
res = model.predict(test_data)
res[0]
res = res.reshape(-1)
res
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math

mse = mean_squared_error(res, test_labels)
mae = mean_absolute_error(res, test_labels)
print('Mean Sqaured Error:', mse)
print('Root Mean Sqaured Error:', math.sqrt(mse))
print('Mean Absolute Error', mae)
n = 15

x = np.arange(1, n+1)
y_true = test_labels[: n]
y_pred = res[: n]

plt.scatter(x, y_pred)
plt.scatter(x, y_true)
plt.legend(['y_pred', 'y_true'])
plt.title('Housing Price')
plt.ylabel('Median Housing Price')
plt.xticks(np.arange(1, n+1))
plt.tight_layout()