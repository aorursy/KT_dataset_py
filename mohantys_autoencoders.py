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
import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.parallel

import torch.optim as optim

import torch.utils.data

from torch.autograd import Variable

from sklearn.model_selection import  train_test_split
movies = pd.read_csv('/kaggle/input/moviereviews/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

users = pd.read_csv('/kaggle/input/moviereviews/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

ratings = pd.read_csv('/kaggle/input/moviereviews/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
print(movies.head())

print(users.head())

print(ratings.head())
X = ratings.drop([2,3], axis=1)

Y = ratings[2]

training_set, test_set = train_test_split(ratings, test_size =0.2)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.2)

print(X_test.shape, X_train.shape, Y_test.shape)

X_train = np.array(X_train, dtype = 'int')

print(training_set.head(), test_set.head())

training_set = np.array(training_set, dtype = 'int')

test_set = np.array(test_set, dtype = 'int')
nb_users = int(max(max(X_train[:,0]), max(X_train[:,0])))

nb_movies = int(max(max(X_train[:,1]), max(X_train[:,1])))

print(nb_users, nb_movies)
def convert(data):

    new_data = []

    for id_users in range(1, nb_users + 1):

        #print(id_users)

        id_movies = data[:,1][data[:,0] == id_users]

        id_ratings = data[:,2][data[:,0] == id_users]

        ratings = np.zeros(nb_movies)

        ratings[id_movies - 1] = id_ratings

        new_data.append(list(ratings))

    return new_data

training_set = convert(training_set)

test_set = convert(test_set)

#print(training_set.head(), test_set.head())

training_set = torch.FloatTensor(training_set)

test_set = torch.FloatTensor(test_set)
print(training_set)
class SAE(nn.Module):

    def __init__(self, ):

        super(SAE, self).__init__()

        self.fc1 = nn.Linear(nb_movies, 20)

        self.fc2 = nn.Linear(20, 10)

        self.fc3 = nn.Linear(10, 20)

        self.fc4 = nn.Linear(20, nb_movies)

        self.activation = nn.Sigmoid()

    def forward(self, x):

        x = self.activation(self.fc1(x))

        x = self.activation(self.fc2(x))

        x = self.activation(self.fc3(x))

        x = self.fc4(x)

        return x
sae = SAE()

criterion = nn.MSELoss()

optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)
nb_epoch = 200

for epoch in range(1, nb_epoch + 1):

    train_loss = 0

    s = 0.  #RMS error so its float

    for id_user in range(nb_users):

        input = Variable(training_set[id_user]).unsqueeze(0)  #Pytorch doesn't accept single vector as input but batch

        target = input.clone()

        if torch.sum(target.data > 0) > 0:   #To optimize memory

            output = sae(input)

            target.require_grad = False  #Save computation

            output[target == 0] = 0

            loss = criterion(output, target)

            #print(loss)

            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)

            loss.backward()

            #if loss.data > 0:

                #train_loss += np.sqrt(loss.data[0]*mean_corrector)  #invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number

            train_loss += np.sqrt(loss.item()*mean_corrector)

            s += 1.

            optimizer.step()

    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
loss.data
test_loss = 0

s = 0.

for id_user in range(nb_users):

    input = Variable(training_set[id_user]).unsqueeze(0)

    target = Variable(test_set[id_user])

    if torch.sum(target > 0) > 0:

        output = sae(input)

        target.require_grad = False

        output[target[1] == 0] = 0

        loss = criterion(output, target)

        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)

        test_loss += np.sqrt(loss.item()*mean_corrector)

        s += 1.

print('test loss: '+str(test_loss/s))