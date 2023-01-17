import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense



np.random.seed(7)
# load dataset

dataset = pd.read_csv("../input/diabetes.csv")

dataset.head()
X = dataset.loc[:,['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

Y = dataset.loc[:, 'Outcome']
# create model

model = Sequential()

model.add(Dense(12, input_dim=8, activation='relu'))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
# compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the model

model.fit(X, Y, epochs=150, batch_size=10)
# evaluate the model

scores = model.evaluate(X, Y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))