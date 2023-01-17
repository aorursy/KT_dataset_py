# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.datasets import boston_housing

(train_data, train_target), (test_data, test_target) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
train_data /= std
test_data -= mean
test_data /= std
from keras.models import Sequential
from keras.layers import Dense

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
k = 4
num_val_samples = train_data.shape[0] // k
num_epochs = 100
all_scores = []

for i in range(k):
    print('Processing Fold #', i)
    val_data = train_data[i * num_val_samples : (i+1) * num_val_samples]
    val_target = train_target[i * num_val_samples : (i+1) * num_val_samples]
    
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i+1) * num_val_samples:]], axis=0)
    partial_train_target = np.concatenate([train_target[:i * num_val_samples], train_target[(i+1) * num_val_samples:]], axis=0)
    
    model = build_model()
    model.fit(partial_train_data, partial_train_target, epochs=num_epochs, batch_size=1, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_target, verbose=0)
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))
num_epochs = 500
all_mae_histories = []

for i in range(k):
    print('Processing Fold #', i+1)
    val_data = train_data[i * num_val_samples : (i+1) * num_val_samples]
    val_target = train_target[i * num_val_samples : (i+1) * num_val_samples]
    
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i+1) * num_val_samples:]], axis=0)
    partial_train_target = np.concatenate([train_target[:i * num_val_samples], train_target[(i+1) * num_val_samples:]], axis=0)
    
    model = build_model()
    history = model.fit(partial_train_data, partial_train_target, epochs=num_epochs, batch_size=1, validation_data=(val_data, val_target), verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)
    
avg_mae_histories = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
import matplotlib.pyplot as plt

plt.plot(range(1, len(avg_mae_histories)+1), avg_mae_histories)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
            
smoothed_mae_histories = smooth_curve(avg_mae_histories[10:])

plt.plot(range(1, len(smoothed_mae_histories)+1), smoothed_mae_histories)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
model = build_model()
model.fit(train_data, train_target, epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_target)
print(test_mse_score, test_mae_score)