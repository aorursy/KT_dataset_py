import numpy as np

import pandas as pd



data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print('loading complete')
print("--- Train dataset ---")

print(data.info())

print("--- Train Head ---")

print(data.head())

print("--- Test dataset ---")

print(test.info())

print("--- Test Head ---")

print(test.head())
print(data.label.value_counts())
X = data.iloc[:,data.columns != 'label']

print("--- X ---")

print(X.head())

print(X.info())
print("--- target ---")

target = data.iloc[:,data.columns == 'label']



y = []



for i in range(len(target)):

    y.append([])

    for j in range(10):

        if (target.label[i] == j):

            y[i].append(1)

        else:

            y[i].append(0)



y = pd.DataFrame(y)

print(y)
print("--- X_test ---")

X_test = test.iloc[:,test.columns != 'label']

print(X_test.head())

print(X_test.info())
from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import EarlyStopping



model = Sequential()



model.add(Dense(50, activation='relu', input_shape=(784,)))

model.add(Dense(50, activation='relu'))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



early_stopping_monitor = EarlyStopping(patience=10)



model.fit(X, y, validation_split=0.3, epochs=100, callbacks=[early_stopping_monitor])



predictions = model.predict(X_test)



print("done")
import matplotlib.pyplot as plt



img = X_test.values.reshape(-1,28,28,1)



g = plt.imshow(img[0][:,:,0])

print("The model predict :", predictions[0])
import random
n = random.randrange(0, len(img))

g = plt.imshow(img[n][:,:,0])



print("The model predict :", predictions[n])
submission = pd.read_csv('../input/sample_submission.csv')



print(submission)
np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).dot(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
print(predictions)



for i in range(len(predictions)):

    submission.Label[i] = predictions[i].dot(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

    

print(submission)
submission.to_csv('../my_submission.csv')

print('submission saved')