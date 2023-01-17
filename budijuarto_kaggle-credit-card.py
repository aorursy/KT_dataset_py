import pandas as pd

import numpy as np

import keras
data = pd.read_csv("../input/creditcard.csv")
data.head()
from sklearn.preprocessing import StandardScaler

data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

data = data.drop(['Amount'],axis=1)
data.head()
X = data.iloc[:, data.columns != 'Class']

y = data.iloc[:, data.columns == 'Class']
y.head()
# split data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
X_train.shape
X_test.shape
X_train = np.array(X_train)

X_test = np.array(X_test)

y_train = np.array(y_train)

y_test = np.array(y_test)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout
model = Sequential([

    Dense(units =16, input_dim = 30,activation='relu'),

    Dense(units =24,activation='relu'),

    Dropout(0.5),

    Dense(20,activation='relu'),

    Dense(24,activation='relu'),

    Dense(1,activation='sigmoid'),

])
model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=15, epochs=5)
score = model.evaluate(X_test, y_test)
print(score)