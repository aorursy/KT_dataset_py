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
# Import all necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Read our data
data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
# Take a look on our data
data.head()
# All features scaled but Time and Amount.
data.describe()
# No missing values
data.isnull().sum()
# All features of type float64
data.info()
# Check how many samples we have from each class
num_fraud = len(data.loc[data['Class'] == 1])
num_valid = len(data.loc[data['Class'] == 0])
print(f'{num_fraud} deals are fraud.')
print(f'{num_valid} deals are valid.')
# Let's make it even - downsample
frauds = data.loc[data['Class'] == 1]

valid = data.loc[data['Class']==0].sample(len(frauds))

new_data = frauds.append(valid)
# Sanity check
num_fraud = len(new_data.loc[new_data['Class'] == 1])
num_valid = len(new_data.loc[new_data['Class'] == 0])
print(f'{num_fraud} deals are fraud.')
print(f'{num_valid} deals are valid.')
# Normalize Time and Amount

time_mean, time_std = new_data.Time.mean(), new_data.Time.std()
amount_mean, amount_std = new_data.Amount.mean(), new_data.Amount.std()

norm_Time = (new_data['Time'] - time_mean) / time_std
norm_Amount = (new_data['Amount'] - amount_mean) / amount_std
# Replace the normalized features with the old ones
new_data['norm_Time'] = norm_Time
new_data['norm_Amount'] = norm_Amount

norm_data = new_data.drop(['Time', 'Amount'], axis=1)
norm_data.describe()
# Shuffle before making any splits and turn into numpy tensor
norm_data = norm_data.sample(frac=1)

# Split to independent and dependent features + convert to numpy tensor
X = norm_data.drop(['Class'], axis=1).to_numpy(dtype='float32')
y = norm_data['Class'].to_numpy(dtype='float32')
# Split the data into train & test.
split_idx = round(len(X)*0.2)

x_test = X[:split_idx]
x_train = X[split_idx:]
y_test = y[:split_idx]
y_train = y[split_idx:]

print(f'Train size={len(x_train)}\nTest size={len(x_test)}')
from keras import models
from keras import layers

# Given train data and train labels, performs kfold cross validation and returns the list of accuracies.

def build_model(input_size):
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_shape=(input_size,)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def kfold_cv(X, y, k, epochs):
    k = 4
    num_val_samples = len(X) // k
    num_epochs = epochs
    all_scores = []
    for i in range(k):
        print('processing fold #', i)
        val_data = X[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = y[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate([X[:i * num_val_samples],X[(i + 1) * num_val_samples:]],axis=0)
        partial_train_targets = np.concatenate([y[:i * num_val_samples],y[(i + 1) * num_val_samples:]],axis=0)
        
        model = build_model(X.shape[1])
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        model.fit(partial_train_data, partial_train_targets,
        epochs=num_epochs, batch_size=1, verbose=1)
        val_loss, val_acc = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_acc)
    return all_scores
results = kfold_cv(x_train, y_train, 4, 20)
print(f'Cross validation accuracies = {results}')
print(f'Mean accuracy = {np.mean(results)}')
# Train again, but on the entire training set to use more samples

model = build_model(x_train.shape[1])
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.fit(x_train, y_train,epochs=20, batch_size=1, verbose=1)

# Evaluate on test set
evaluation = model.evaluate(x_test, y_test, verbose=0)
print(f'Evaluation on test set:\nloss = {evaluation[0]},\naccuracy = {evaluation[1]}')