# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Requirements
import keras
from keras import Sequential

# Turn off complaints
import warnings
warnings.filterwarnings("ignore")
# Import Data
df = pd.read_csv('../input/Iris.csv')
# Check it out
df.head()
# EDA
import seaborn as sns

sns.pairplot(df, hue="Species")
# Data Prep

# Manip
targets = df["Species"]
df.drop(["Species", "Id"], axis = 1, inplace=True)


# Normalize
from sklearn import preprocessing

x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

# Convert Targets to One-Hots
targets = targets.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                [0,1,2])

targets = keras.utils.np_utils.to_categorical(targets, num_classes=3)
# Sanity
len(df)

# Build Neural Network
from keras.layers import Dense

# Create a new Sequential object
model = Sequential()

# Create the input layer, 50 nodes
model.add(Dense(10, input_shape=(4,)))

# Create the hidden layer
model.add(Dense(10, activation="relu"))
          
# Create an output layer, 3 nodes
model.add(Dense(3, activation="sigmoid"))

# Compile model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

# Train the NN model
model.fit(x=df, y=targets, epochs=100, batch_size=2, validation_split=0.10)

# Evaluate model
model.evaluate(x=df, y=targets, batch_size=16)
preds = model.predict_classes(df)

for i in range(len(df)):
    print("Prediction = %s, Actual = %s" % (preds[i], targets[i]))
# Manip
data = df.copy()
data["preds"] = preds

sns.pairplot(data, hue="preds", vars=[0,1,2,3])