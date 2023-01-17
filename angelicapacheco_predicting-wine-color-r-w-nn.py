 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling 

from keras import models

from keras import layers
# Load the dataset

df = pd.read_csv('../input/wines.csv', delimiter =";")

df.head()

#Important: 1 = red, 0 = white
pandas_profiling.ProfileReport(df) 
df.shape
df.isna().sum()
# Display a description of the dataset



df.describe()
#Correlation again

corr=df.corr()

corr
 %matplotlib inline
corr=df.corr()

plt.figure(figsize=(14,6))

sns.heatmap(corr,annot=True)
df.head()
#Only high corr variables to "type" were chosen to build the model

X = df[["fixed acidity", "volatile acidity" , "chlorides", "sulphates", 'free sulfur dioxide', 'total sulfur dioxide']]

y = df['type']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)

x_test  = sc.transform(x_test)
#Mean absolute error taken as metric for modelling. Mean squared error, as loss func

#The RMSprop optimizer restricts the oscillations in the vertical direction in gradient descent: taking larger steps in the horizontal direction converging faster     



model = models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape=(6,)))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(1))



model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])



model.fit(x_train, y_train, batch_size=8, epochs=10)
#Performance with the test data

test_mse, test_mae = model.evaluate(x_test, y_test, verbose=0)

print('Mean Squared Error: ',test_mse, '\n','Mean Absolute Error: ',test_mae)