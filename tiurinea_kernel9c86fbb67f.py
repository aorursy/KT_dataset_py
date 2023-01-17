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
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import torch



train_df = pd.read_csv('../input/dlschool-fashionmnist2/fashion-mnist_train.csv')

test_df = pd.read_csv('../input/dlschool-fashionmnist2/fashion-mnist_test.csv')



X_train = train_df.values[:, 1:]

y_train = train_df.values[:, 0]



X_test = test_df.values
X_train_tensor = torch.FloatTensor(X_train)

y_train_tensor = torch.LongTensor(y_train.astype(np.int64))

X_test_tensor  = torch.FloatTensor(X_test)



def generate_batches(X, y, batch_size=64):

    for i in range(0, X.shape[0], batch_size):

        X_batch, y_batch = X[i:i+batch_size], y[i:i+batch_size]

        yield X_batch, y_batch



# D_in - размерность входа (количество признаков у объекта)

# D_out - размерность выходного слоя (суть - количество классов)

D_in, D_out = 784, 10



# определим нейросеть:

myNet = torch.nn.Sequential(

    torch.nn.Linear(D_in, 150),

    torch.nn.ReLU(), 

    torch.nn.Linear(150, 50),

    torch.nn.Sigmoid(), 

    torch.nn.Linear(50, D_out),

    torch.nn.Softmax()

)



BATCH_SIZE = 1024

NUM_EPOCHS = 200



loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

#oss_fn = torch.nn.NLLLoss(reduction='sum')



optimizer = torch.optim.SGD(myNet.parameters(), lr=0.0001)



Losses=list()

for epoch_num  in range(NUM_EPOCHS):

    iter_num = 0

    running_loss = 0.0

    for X_batch, y_batch in generate_batches(X_train_tensor, y_train_tensor, BATCH_SIZE):

        # forward (подсчёт ответа с текущими весами)

        y_pred = myNet(X_batch)



        # вычисляем loss'ы

        loss = loss_fn(y_pred, y_batch)

        

        running_loss += loss.item()

        

        # выводем качество каждые 100 батчей            

        if iter_num % 100 == 0:

            Losses.append(running_loss / 100)

            #print('{} {}'.format(iter_num,running_loss/100))

            running_loss = 0.0

            

        # зануляем градиенты

        optimizer.zero_grad()



        # backward (подсчёт новых градиентов)

        loss.backward()



        # обновляем веса

        optimizer.step()

        

        iter_num += 1

      

plt.plot(Losses)

plt.show()



class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))



classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker','Bag', 'Ankle boot']



with torch.no_grad():

    for X_batch, y_batch in generate_batches(X_train_tensor, y_train_tensor, BATCH_SIZE):

        y_pred = myNet(X_batch)

        _, predicted = torch.max(y_pred, 1)

        c = (predicted == y_batch).squeeze()

        for i in range(len(y_pred)):

            label = y_batch[i]

            class_correct[label] += c[i].item()

            class_total[label] += 1





for i in range(10):

    print('Accuracy of %5s : %2d %%' % (

        classes[i], 100 * class_correct[i] / class_total[i]))

    

y_test_pred = myNet(X_test_tensor)

_, predicted = torch.max(y_test_pred, 1)



answer_df = pd.DataFrame(data=predicted.numpy(), columns=['Category'])

answer_df.to_csv('output', index=False)