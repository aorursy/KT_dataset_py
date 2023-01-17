# import lybraries

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
# import and null values remove

puls = pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')

puls.head()
# create train and test set

x = puls.drop('target_class', axis=1).values

y = puls.target_class.values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
'''

create class perceptron

@predict used to predict the class, parameter x is a list of values with length = input_dims, trained during training

@train used to train the perceptron: x,y are train_Set and test_set for training, bias the initial bias, learnining_rate the learning_rate

and epochs the number of time the train will iterate throught the data

''' 

class perceptron:

        

    def predict(self,x):

        # insert x0 = 1

        t = x.copy()

        t.insert(0,1)

        # check if input is correct

        if len(t) == self.inputs_dim + 1:

            return sum(self.w * t) > 0 #make prediction

        else:

            return None

        

    def train(self, x, y, bias = 1, learning_rate = 0.001, epochs=10, casual=0):

        x = x.tolist()

        y = y.tolist()

        #check if input is correct

        if len(x) == 0 | len(y) == 0 | (len(x) != len(y)) | (sum(np.array([len(a) for a in x]) != len(x[0])) == 0):

            return None

        # train input_dim

        self.inputs_dim = len(x[0])

        if casual == 1:

            # randomize initial weights

            self.w = np.random.uniform(low=-1, high=1, size=(self.inputs_dim + 1,))

        else:

            # starting all weights to 0

            self.w = np.zeros(self.inputs_dim + 1)

        self.w[0] = bias

        self.learning_rate = learning_rate

        # iterate throught epochs

        for e in range(epochs):

            # iterate throught data

            for i in range(len(x)):

                p = self.predict(x[i])

                # iterate throught single piece of data

                for j in range(len(x[i])):

                    self.w[j] += self.learning_rate * (y[i] - p) * x[i][j]
#create new perceptron and trying with 0-weights initialization

perc = perceptron()

ris = []

# trying more learning_rates

learnings = [0.00001, 0.0001, 0.001, 0.01, 0.1]

for l in learnings:

    ind = learnings.index(l)

    ris.append([])

    # trying more epochs

    for i in range(1,11):

        ris[ind].append([])

        perc.train(x=X_train, y=y_train, bias=1, learning_rate=l, epochs=i, casual=0)

        for e in X_test:

            ris[ind][i-1].append(perc.predict(e.tolist()))

        ris[ind][i-1] = sum(ris[ind][i-1] == y_test)/len(y_test)
# plot results

plt.rcParams["figure.figsize"] = (30, 50)

plt.rcParams.update({'font.size': 22})

fig, (ax1, ax2, ax3,ax4, ax5) = plt.subplots(5, 1)

axis = [ax1, ax2, ax3, ax4, ax5]

for l in range(5):

    axis[l].set_title('learning_rate = {}'.format(learnings[l]))

    axis[l].plot(range(1,11), ris[l])

plt.tight_layout()
#create new perceptron and trying with casual-weights initialization

perc_c = perceptron()

ris_c = []

for l in learnings:

    ind = learnings.index(l)

    ris_c.append([])

    # trying more epochs

    for i in range(1,11):

        ris_c[ind].append([])

        perc_c.train(x=X_train, y=y_train, bias=1, learning_rate=l, epochs=i, casual=1)

        for e in X_test:

            ris_c[ind][i-1].append(perc_c.predict(e.tolist()))

        ris_c[ind][i-1] = sum(ris_c[ind][i-1] == y_test)/len(y_test)
# plot results

fig_c, (ax1c, ax2c, ax3c,ax4c, ax5c) = plt.subplots(5, 1)

axis_c = [ax1c, ax2c, ax3c, ax4c, ax5c]

for l in range(5):

    axis_c[l].set_title('learning_rate = {}'.format(learnings[l]))

    axis_c[l].plot(range(1,11), ris_c[l])

plt.tight_layout()