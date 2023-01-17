import pandas as pd

import numpy as np



df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

df.head()
print ('Number of Rows :', df.shape[0])

print ('Number of Columns :', df.shape[1])

print ('Number of Patients with outcome 1 :', df.Outcome.sum())

print ('Event Rate :', round(df.Outcome.mean()*100,2) ,'%')
df.describe()
from sklearn.model_selection import train_test_split

X = df.to_numpy()[:,0:8] 

Y = df.to_numpy()[:,8]

seed = 42

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = seed)

print (f'Shape of Train Data : {X_train.shape}')

print (f'Shape of Test Data : {X_test.shape}')
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

model = Sequential([

    Dense(24, input_dim = (8), activation = 'relu'),

    Dense(12, activation = 'relu'),

    Dense(1, activation = 'sigmoid'),

])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()
history = model.fit(X_train, y_train, epochs=150, batch_size=32, verbose = 1)
scores = model.evaluate(X_test, y_test)

print (f'{model.metrics_names[1]} : {round(scores[1]*100, 2)} %')
import matplotlib.pyplot as plt



# Plotting loss

plt.plot(history.history['loss'])

plt.title('Binary Cross Entropy Loss on Train dataset')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.show()



# Plotting accuracy metric

plt.plot(history.history['accuracy'])

plt.title('Accuracy on the train dataset')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.show()