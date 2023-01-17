# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense

import csv 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import pickle

import random

from keras.layers import Concatenate, concatenate

from keras.layers.normalization import BatchNormalization

import numpy as np

import h5py

from sklearn.metrics import accuracy_score

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
import pandas as pd

dataset_haberman = pd.read_csv("../input/haberman/haberman.csv")

dataset_haberman = shuffle(dataset_haberman);
print(dataset_haberman.head(10))


dataset = dataset_haberman.values
X = dataset[:,0:3]

Y = dataset[:,3]
X
X_train, X_test, y_train, y_test = train_test_split(X, Y)
model = Sequential()

model.add(Dense(6, activation='sigmoid', input_shape=(3,)))

model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile( optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'] )
hist = model.fit(X_train, y_train, batch_size=4, epochs=10)
model.evaluate(X_test, y_test)[1]


test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)



print('\nTest accuracy:', test_acc)

print('\n Test loss: ', test_loss)
class GeneticNN(Sequential):

    def __init__(self, weights=None):

        super().__init__()

        if weights != None:

            self.add(

                Dense(

                    6,

                    input_shape=(3,),

                    activation='sigmoid',

                    weights=[weights[0], np.zeros(6)])

                )

            self.add(

                Dense(

                 1,

                 activation='sigmoid',

                 weights=[weights[1], np.zeros(1)])

            )

        else:

            layer1 = Dense(6, input_shape=(3,), activation='sigmoid')

            layer2 = Dense(1, activation='sigmoid')

            

            self.add(layer1)

            self.add(layer2)

    



    def compile_train(self, epochs):

        self.compile(

                      optimizer='rmsprop',

                      loss='binary_crossentropy',

                      metrics=['accuracy']

                      )

        self.fit(X_train, y_train, epochs=epochs)

        

    def forward_propagation(self, X_train, y_train):

        y_hat = self.predict(X_train)

        self.fitness = accuracy_score(y_train, y_hat.round())

def mutation(weights):

    selection = random.randint(0, len(weights)-1)

    mut = random.uniform(0, 1)

    if mut >= .5:

        weights[selection] *= random.randint(2, 5)

    else:

        pass



def dynamic_crossover(nn1, nn2):

    nn1_weights = []

    nn2_weights = []

    weights = []

    for layer in nn1.layers:

        nn1_weights.append(layer.get_weights()[0])



    for layer in nn2.layers:

        nn2_weights.append(layer.get_weights()[0])



    for i in range(0, len(nn1_weights)):

        split = random.randint(0, np.shape(nn1_weights[i])[1]-1)

        for j in range(split, np.shape(nn1_weights[i])[1]-1):

            nn1_weights[i][:, j] = nn2_weights[i][:, j]



        weights.append(nn1_weights[i])



    mutation(weights)



    child = GeneticNN(weights)

    return child





networks = []

pool = []

generation = 0

n = 20



for i in range(0, n):

    networks.append(GeneticNN())



max_fitness = 0



optimal_weights = []





while max_fitness < .9 and generation < 50:

    generation += 1

    print('Generation: ', generation)



    for nn in networks:

      

        nn.forward_propagation(X_train, y_train)

       

        pool.append(nn)





    networks.clear()



    

    pool = sorted(pool, key=lambda x: x.fitness)

    pool.reverse()



    

    for i in range(0, len(pool)):

        if pool[i].fitness > max_fitness:

            max_fitness = pool[i].fitness

            print('Max Fitness: ', max_fitness)

            optimal_weights = []

            for layer in pool[i].layers:

                optimal_weights.append(layer.get_weights()[0])

            print(optimal_weights)



    for i in range(0, 5):

        for j in range(0, 2):

            temp = dynamic_crossover(pool[i], random.choice(pool))

            networks.append(temp)



gnn = GeneticNN(optimal_weights)

gnn.compile_train(10)



y_hat = gnn.predict(X_test)

print('Test Accuracy: %.2f' % accuracy_score(y_test, y_hat.round()))
import pandas as pd

haberman = pd.read_csv("../input/haberman/haberman.csv")