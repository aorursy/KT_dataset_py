import math

import pandas as pd

from keras import models, layers, optimizers, regularizers

import numpy as np

import random

from sklearn import model_selection, preprocessing

import tensorflow as tf

from tqdm import tqdm

import matplotlib.pyplot as plt
file_name = '../input/SAheart.data'

data = pd.read_csv(file_name, sep=',', index_col=0)
data['famhist'] = data['famhist'] == 'Present'

data.head()
n_test = int(math.ceil(len(data) * 0.3))

random.seed(42)

test_ixs = random.sample(list(range(len(data))), n_test)

train_ixs = [ix for ix in range(len(data)) if ix not in test_ixs]

train = data.iloc[train_ixs, :]

test = data.iloc[test_ixs, :]

print(len(train))

print(len(test))
#features = ['sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'typea', 'obesity', 'alcohol', 'age']

features = ['adiposity', 'age']

response = 'chd'

x_train = train[features]

y_train = train[response]

x_test = test[features]

y_test = test[response]
x_train = preprocessing.normalize(x_train)

x_test = preprocessing.normalize(x_test)
hidden_units = 10     # how many neurons in the hidden layer

activation = 'relu'   # activation function for hidden layer

l2 = 0.01             # regularization - how much we penalize large parameter values

learning_rate = 0.01  # how big our steps are in gradient descent

epochs = 5            # how many epochs to train for

batch_size = 16       # how many samples to use for each gradient descent update
# create a sequential model

model = models.Sequential()



# add the hidden layer

model.add(layers.Dense(input_dim=len(features),

                       units=hidden_units, 

                       activation=activation))



# add the output layer

model.add(layers.Dense(input_dim=hidden_units,

                       units=1,

                       activation='sigmoid'))



# define our loss function and optimizer

model.compile(loss='binary_crossentropy',

              # Adam is a kind of gradient descent

              optimizer=optimizers.Adam(lr=learning_rate),

              metrics=['accuracy'])
# train the parameters

history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size)



# evaluate accuracy

train_acc = model.evaluate(x_train, y_train, batch_size=32)[1]

test_acc = model.evaluate(x_test, y_test, batch_size=32)[1]

print('Training accuracy: %s' % train_acc)

print('Testing accuracy: %s' % test_acc)



losses = history.history['loss']

plt.plot(range(len(losses)), losses, 'r')

plt.show()



### RUN IT AGAIN! ###
def train_and_evaluate(model, x_train, y_train, x_test, y_test, n=20):

    train_accs = []

    test_accs = []

    with tqdm(total=n) as progress_bar:

        for _ in range(n):

            model.fit(

                x_train, 

                y_train, 

                epochs=epochs, 

                batch_size=batch_size,

                verbose=False)

            train_accs.append(model.evaluate(x_train, y_train, batch_size=32, verbose=False)[1])

            test_accs.append(model.evaluate(x_test, y_test, batch_size=32, verbose=False)[1])

            progress_bar.update()

    print('Avgerage Training Accuracy: %s' % np.average(train_accs))

    print('Avgerage Testing Accuracy: %s' % np.average(test_accs))

    return train_accs, test_accs
_, test_accs = train_and_evaluate(model, x_train, y_train, x_test, y_test)
plt.hist(test_accs)

plt.show()
print('Min: %s' % np.min(test_accs))

print('Max: %s' % np.max(test_accs))
hidden_units = 10     # how many neurons in the hidden layer

activation = 'relu'   # activation function for hidden layer

l2 = 0.01             # regularization - how much we penalize large parameter values

learning_rate = 0.01  # how big our steps are in gradient descent

epochs = 5            # how many epochs to train for

batch_size = 16       # how many samples to use for each gradient descent update
# create a sequential model

model = models.Sequential()



# add the hidden layer

model.add(layers.Dense(input_dim=len(features),

                       units=hidden_units, 

                       activation=activation))



# add the output layer

model.add(layers.Dense(input_dim=hidden_units,

                       units=1,

                       activation='sigmoid'))



# define our loss function and optimizer

model.compile(loss='binary_crossentropy',

              # Adam is a kind of gradient descent

              optimizer=optimizers.Adam(lr=learning_rate),

              metrics=['accuracy'])
_, __ = train_and_evaluate(model, x_train, y_train, x_test, y_test)