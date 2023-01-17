import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))



mit_test_data = pd.read_csv("../input/mitbih_test.csv", header=None)

mit_train_data = pd.read_csv("../input/mitbih_train.csv", header=None)



print("MIT test dataset")

print(mit_test_data.info())

print("MIT train dataset")

print(mit_train_data.info())
# take a random distribution

sample = mit_test_data.sample(25)



# remove the target column

sampleX = sample.iloc[:,sample.columns != 187]



import matplotlib.pyplot as plt



plt.style.use('classic')



# plt samples

for index, row in sampleX.iterrows():

    plt.plot(np.array(range(0, 187)) ,row)



plt.xlabel("time")

plt.ylabel("magnitude")

plt.title("heartbeat reccording \nrandom sample")



plt.show()



plt.style.use("ggplot")



plt.title("Number of record in each category")



plt.hist(sample.iloc[:,sample.columns == 187].transpose())

plt.show()
print("Train data")

print("Type\tCount")

print((mit_train_data[187]).value_counts())

print("-------------------------")

print("Test data")

print("Type\tCount")

print((mit_test_data[187]).value_counts())
from keras.utils import to_categorical



print("--- X ---")

X = mit_train_data.loc[:, mit_train_data.columns != 187]

print(X.head())

print(X.info())



print("--- Y ---")

y = mit_train_data.loc[:, mit_train_data.columns == 187]

y = to_categorical(y)



print("--- testX ---")

testX = mit_test_data.loc[:, mit_test_data.columns != 187]

print(testX.head())

print(testX.info())



print("--- testy ---")

testy = mit_test_data.loc[:, mit_test_data.columns == 187]

testy = to_categorical(testy)
from keras.models import Sequential

from keras.layers import Dense, Activation



model = Sequential()



model.add(Dense(50, activation='relu', input_shape=(187,)))

model.add(Dense(50, activation='relu'))

model.add(Dense(5, activation='softmax'))



model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])



model.fit(X, y, epochs=100)



print("Evaluation: ")

mse, acc = model.evaluate(testX, testy)

print('mean_squared_error :', mse)

print('accuracy:', acc)