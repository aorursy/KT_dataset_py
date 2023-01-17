# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
TRAIN_PATH = '/kaggle/input/fashion-mnist_train.csv'

TEST_PATH = '/kaggle/input/fashion-mnist_test.csv'

train_df = pd.read_csv(TRAIN_PATH)

test_df = pd.read_csv(TEST_PATH)
train_df.head()
X_train = train_df.values[:, 1:]

y_train = train_df.values[:, 0]

X_test = test_df.values

print(X_train.shape, y_train.shape)

print(X_test.shape)

plt.imshow(X_train[0].reshape(28,28), cmap = 'gray')

import matplotlib.pyplot as plt

import torch

torch.__version__
X_train_tensor = torch.FloatTensor(X_train)

y_train_tensor = torch.LongTensor(y_train.astype(np.int64))

print(X_train_tensor.shape, y_train.shape)
print("Кол-во классов: {}".format(y_train_tensor.unique().shape[0]))
num_classes = y_train_tensor.unique().shape[0]

length = y_train_tensor.shape[0]



y_onehot = torch.FloatTensor(length, num_classes)

y_onehot.zero_() 

y_onehot.scatter_(1, y_train_tensor.view(-1, 1), 1)
from collections import OrderedDict

in_neurons, hidden = 784, 200

layers = OrderedDict([

    ('input layer', torch.nn.Linear(in_neurons, hidden)),

    ('relu1', torch.nn.ReLU()),

    ('hidden layer1', torch.nn.Linear(hidden, hidden)),

    ('relu2', torch.nn.ReLU()),

    ('hidden layer2', torch.nn.Linear(hidden, num_classes)),

    ('Softmax', torch.nn.Softmax())

])

net = torch.nn.Sequential(layers)

net
def generate_batches(X, y, batch_size = 64):

    for i in range(0, X.shape[0], batch_size):

        X_batch, y_batch = X[i:i+batch_size], y[i:i + batch_size]

        yield X_batch, y_batch
BATCH_SIZE = 64

NUM_EPOCHS = 200



criterion = torch.nn.CrossEntropyLoss(size_average = False)

learning_rate = 1e-4

optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)

losses = []



for epoch in range(NUM_EPOCHS):

    iter_num = 0

    running_loss = 0.0

    for X_batch, y_batch in generate_batches(X_train_tensor, y_train_tensor, BATCH_SIZE):

        y_pred = net(X_batch)

        loss = criterion(y_pred, y_batch)

        running_loss += loss.item()

        if iter_num % 100 == 99:

            print("[{}, {}] current loss: {}".format(epoch, iter_num + 1, running_loss/2000))

            losses.append(running_loss/2000)

            running_loss = 0.0

            

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        iter_num += 1



    



plt.plot(losses)

plt.ylabel("Loss")

plt.xlabel("$Epoch$")

plt.show()
class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))



classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

           'Sandal', 'Shirt', 'Sneaker','Bag', 'Ankle boot']



with torch.no_grad():

    for X_batch, y_batch in generate_batches(X_train_tensor, y_train_tensor, BATCH_SIZE):

        y_pred = net(X_batch)

        _, predicted = torch.max(y_pred, 1)

        c = (predicted == y_batch).squeeze()

        for i in range(len(y_pred)):

            label = y_batch[i]

            class_correct[label] += c[i].item()

            class_total[label] += 1





for i in range(10):

    print('Accuracy of %5s : %2d %%' % (

        classes[i], 100 * class_correct[i] / class_total[i]))
y_test_pred = net(torch.FloatTensor(X_test))

y_test_pred.shape
_, predicted = torch.max(y_test_pred, 1)

predicted
answer_df = pd.DataFrame(data = predicted.numpy(), columns = ['Category'])

answer_df.head()
answer_df['Id'] = answer_df.index

answer_df.head()

answer_df.to_csv('./answer.csv', index = False)