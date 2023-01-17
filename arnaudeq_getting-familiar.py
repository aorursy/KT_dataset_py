import numpy as np
np.arange(12)
2**5
def f(x,y):

    """ test """

    return x**y
z = f(2,5)
try:

    y = x + 1

except Exception as e:

    print(e)
z = f(2,5)



if z is not None:

    print(f"z is not none because z = {z}")

else:

    print("z is none")
[i for i in range(15)]
list(zip(range(1,5), range(1,5)))
list(range(5))
[f(x,y) for (x,y) in zip(range(1,6), range(1,6))]
import pandas as pd
my_data = [(175, 70),

          (168, 59),

          (170, 79),

          (149, 49),

          (185, 69),

          (189, 90)]
df = pd.DataFrame(my_data, columns=["height", "weight"])

df
df.describe()
df.height.hist(bins=10)
my_data = [(175, 70, 0),

          (168, 59, 1),

          (170, 79, 0),

          (149, 49, 1),

          (185, 69, 0),

          (189, 90, 0)]

df = pd.DataFrame(my_data, columns=["height", "weight", "is_woman"])
df
def f(x):

    """ My documentation blablabla """

    return None
df.groupby("is_woman").height.mean()
df.plot(x="height", y="weight", kind="scatter")
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential

from keras.layers import Dense

import matplotlib.pyplot as plt 
X = np.linspace(start=-1,stop=1, num=300)

np.random.shuffle(X)    # randomize the data

#Y = 0.5 * X*X + X + 2 + np.random.normal(0, 0.05, (300, ))

Y = X + 2 + np.random.normal(0, 0.05, (300, ))
plt.scatter(X, Y)

plt.show()
X_train, Y_train = X[:160], Y[:160]

X_val, Y_val = X[160:200], Y[160:200]  

X_test, Y_test = X[200:], Y[200:]   
model = Sequential()

model.add(Dense(input_dim=1, units=1))

#model.add(Dense(input_dim=1, units=1))
model.summary()
model.compile(loss='mse', optimizer='adam')
len(X_train)
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=1, epochs=20)
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=1, epochs=4)
model.get_weights()
print("\nTesting ------------")

cost = model.evaluate(X_test, Y_test, batch_size=1)

print(f"test cost: {cost}")

W, b = model.layers[0].get_weights()

print(f"weight: {W}, bias: {b}")
model.layers[0].get_weights()
import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

import numpy as np
model = Sequential()

model.add(Dense(3, input_shape=(1,), activation='relu'))

#model.add(Dense(3, input_dim=1, activation='relu'))

model.summary()
model = Sequential()

model.add(Dense(128, input_shape=(1,), activation='relu'))
model.summary()
model = Sequential()

model.add(Dense(3, input_shape=(2,), activation='relu'))

model.summary()
model = Sequential()

model.add(Dense(3, input_shape=(1,2), activation='relu'))
model.summary()
model = Sequential()

model.add(Dense(128, input_shape=(1,2), activation='relu'))
model.summary()
model = Sequential()

model.add(Dense(128, input_shape=(1,3), activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Flatten())
model.summary()
model = Sequential()

model.add(Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))

model.summary()
#fill
#fill
#fill
#fill
# fill ?
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(28,28,1)))  # you may notice that padding="valid" by default i.e. no padding

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.summary()