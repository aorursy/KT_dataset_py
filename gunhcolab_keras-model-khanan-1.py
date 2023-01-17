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
# My code begins from here........

df_train = pd.read_csv('/kaggle/input/khanan/Training.csv', skiprows = [2, 1])
df_test = pd.read_csv('/kaggle/input/khanan/Test.csv', skiprows = [2, 1])
print(df_test.shape, df_train.shape)
df = df_train.append(df_test)
df.shape
df
df_act = df.drop('Selected', axis=1)
df_act = df_act.drop('HP', axis=1)
targets = df['HP']
print(df_act, targets)
print(df_act.shape, targets.shape)
df_act.head()
df_act.info()
# basic info
df_act.isnull().sum()
# that means there are no missing values
# distributions......

df_act.hist(figsize=(16, 16), bins=100, xlabelsize=12, ylabelsize=12);
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.distplot(targets, color='b', bins=200, hist_kws={'alpha': 0.5});

# most of the target values are between 12k and 14k
# distribution of the data using seaborn (basically the same thing as panda's hist)
dims = (20, 10)
fig, axes = plt.subplots(ncols=3, nrows=2, figsize=dims)

for i, ax in zip(range(6), axes.flat):
    sns.distplot(df_act[df_act.columns[i]], ax=ax)
plt.show()
for i in range(6):
    for ii in range(6):
        if i != ii:
            x = i
            y = ii
            sns.scatterplot(df_act[df_act.columns[x]], df_act[df_act.columns[y]], markers='.', data=df_act, hue=targets)
        plt.show()
fig, axes = plt.subplots(1, 6, figsize=(15, 5))

for i, ax in enumerate(fig.axes):
    if i < len(df_act.columns):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        sns.countplot(x = df_act.columns[i], alpha = 0.6, data = df_act, ax = ax)

fig.tight_layout()

# for establishing pairwise correlation between any two labels

plt.figure(figsize=(10, 8))
sns.heatmap(df_act.corr(), annot=True)
# preprocessing

from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
mms.fit(df_act)
data_scaled = mms.transform(df_act)

#NOTE ! while you try to use this model on new data you need to down scale the targets with max = 14206.0.
#Else there would be ambiguity in the predictions

max = np.max(targets)
target = targets/max
print(target)
print(data_scaled)
print(max)
# splittng the data in the form it was received
from sklearn.model_selection import train_test_split

X_train_all, X_test, y_train_all, y_test = train_test_split(data_scaled, target, test_size=0.2)

# further splitting into validation set for training purpose
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.1)
hidden_layers = 3 
rate_of_learning = 1e-3 

from tensorflow import keras

def model_for_tuning(layer1, layer2, layer3, hiddenN = hidden_layers, learnR = rate_of_learning, input_shape = [6]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layers in range(hiddenN):
        model.add(keras.layers.Dense(layer1, activation="relu"))
        model.add(keras.layers.Dense(layer2, activation="relu"))
        model.add(keras.layers.Dense(layer3, activation="relu"))
    model.add(keras.layers.Dense(1))

    optim = keras.optimizers.Nadam(lr = learnR)
    model.compile(loss="mse", optimizer = optim, metrics = ["mape"])

    return model
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

# defining our hyperparameter space
params_set = {
    "learnR": reciprocal(1e-5, 1e-1),
    "layer1": np.arange(1, 200),
    "layer2": np.arange(1, 200),
    "layer3": np.arange(1, 200)
}

# wrapping
reg_keras = keras.wrappers.scikit_learn.KerasRegressor(model_for_tuning)

# using the 'best hyperparameters finder' module
rsCV = RandomizedSearchCV(reg_keras, params_set, n_iter=10, cv=3)
rsCV.fit(X_train, y_train, epochs=200, callbacks=[keras.callbacks.EarlyStopping(patience = 100)])


print(rsCV.best_params_,"\n\n\n", rsCV.best_score_)

# model_ = rsCV.best_estimator_.model -> is throwing an unknown error


learnR = 4.11747983602376e-05
#model

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[6]))
for layers in range(3):
    model.add(keras.layers.Dense(155, activation="relu"))
    model.add(keras.layers.Dense(117, activation="relu"))
    model.add(keras.layers.Dense(140, activation="relu"))
model.add(keras.layers.Dense(1))
optim = keras.optimizers.Nadam(lr = learnR)
model.compile(loss="mse", optimizer = optim, metrics = ["mape"])
model.fit(X_train, y_train, epochs=5000, validation_data = (X_val, y_val), callbacks=[keras.callbacks.EarlyStopping(patience = 1000)])
#evaluation
model.evaluate(X_test, y_test)
# predicting
predictions = model.predict(X_test)

print('Predicted\n', predictions[:20], '\n\nActual\n', y_test.values[:20])
# predictions vs actual graphs 

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.scatter(predictions*max, y_test*max, marker='.')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
# upscalling the predictions and actual targets

a = list(predictions*max)
b = list(y_test*max)


data_frame = pd.DataFrame(a,columns=['Predictions'])
data_frame['Actual'] = pd.Series(b, index=data_frame.index)

data_frame.to_csv('model1.csv')

