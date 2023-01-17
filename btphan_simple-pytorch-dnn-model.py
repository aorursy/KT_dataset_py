#                           linear1,nn.BatchNorm1d(hiddenLayer1Size),relu,

#                           linear2,dropout,relu,

#                           linear3,dropout,relu,

#                           linear4,dropout,relu,

#                           linear5,dropout,relu,

#                           linear6,dropout,relu,

#                           sigmoid
%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch

from torch import nn

from torch.autograd import Variable

import sklearn

import sklearn.model_selection

from sklearn.metrics import f1_score,confusion_matrix,accuracy_score, log_loss

import seaborn as sns # data visualization library 
news = pd.read_csv('../input/OnlineNewsPopularityReduced.csv')

X = news.iloc[:,2:-1]

column_names = list(X.columns.values)

N_FEATURES = len(column_names)

y = news.iloc[:,-1]

y = (y > 1400) # a news article is considered popular if it is shared more than 1400 times.

news.head()
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42, test_size=0.3)
def xNumpyToTensor(array):

    array = np.array(array, dtype=np.float32) 

    return Variable(torch.from_numpy(array)).type(torch.FloatTensor)



def yNumpyToTensor(array):

    array = np.array(array.astype(int))

    return Variable(torch.from_numpy(array)).type(torch.FloatTensor)



x_tensor_train = xNumpyToTensor(x_train)

y_tensor_train = yNumpyToTensor(y_train)

x_tensor_test = xNumpyToTensor(x_test)

y_tensor_test = yNumpyToTensor(y_test)
# Neural Network parameters

DROPOUT_PROB = 0.9



LR = 0.001

MOMENTUM= 0.99

dropout = torch.nn.Dropout(p=1 - (DROPOUT_PROB))



hiddenLayer1Size=512

hiddenLayer2Size=int(hiddenLayer1Size/4)

hiddenLayer3Size=int(hiddenLayer1Size/8)

hiddenLayer4Size=int(hiddenLayer1Size/16)

hiddenLayer5Size=int(hiddenLayer1Size/32)



#Neural Network layers

linear1=torch.nn.Linear(N_FEATURES, hiddenLayer1Size, bias=True) 

linear2=torch.nn.Linear(hiddenLayer1Size, hiddenLayer2Size)

linear3=torch.nn.Linear(hiddenLayer2Size, hiddenLayer3Size)

linear4=torch.nn.Linear(hiddenLayer3Size, hiddenLayer4Size)

linear5=torch.nn.Linear(hiddenLayer4Size, hiddenLayer5Size)

linear6=torch.nn.Linear(hiddenLayer5Size, 1)

sigmoid = torch.nn.Sigmoid()

threshold = nn.Threshold(0.5, 0)

tanh=torch.nn.Tanh()

relu=torch.nn.LeakyReLU()



#Neural network architecture

net = torch.nn.Sequential(linear1,nn.BatchNorm1d(hiddenLayer1Size),relu,

                          linear2,dropout,relu,

                          linear3,dropout,relu,

                          linear4,dropout,relu,

                          linear5,dropout,relu,

                          linear6,dropout,relu,

                          sigmoid

                          )



optimizer = torch.optim.Adam(net.parameters(), lr=LR,weight_decay=5e-3)

loss_func=torch.nn.BCELoss()

epochs = 200

all_losses = []



#Training in batches

for step in range(epochs):    

    out = net(x_tensor_train)                 # input x and predict based on x

    cost = loss_func(out, y_tensor_train) 

    optimizer.zero_grad()   # clear gradients for next train

    cost.backward()         # backpropagation, compute gradients

    optimizer.step()        # apply gradients 

        

    if step % 5 == 0:        

        loss = cost.data

        all_losses.append(loss)

        print(step, cost.data.cpu().numpy())

        # RuntimeError: can't convert CUDA tensor to numpy (it doesn't support GPU arrays). 

        # Use .cpu() to move the tensor to host memory first.        

        prediction = (net(x_tensor_test).data).float() # probabilities         

#         prediction = (net(X_tensor).data > 0.5).float() # zero or one

#         print ("Pred:" + str (prediction)) # Pred:Variable containing: 0 or 1

#         pred_y = prediction.data.numpy().squeeze()            

        pred_y = prediction.cpu().numpy().squeeze()

        target_y = y_tensor_test.cpu().data.numpy()

        print ('LOG_LOSS={} '.format(log_loss(target_y, pred_y))) 



#Evaluating the performance of the model

%matplotlib inline

plt.plot(all_losses)

plt.show()

pred_y = pred_y > 0.5

print('f1 score', f1_score(target_y, pred_y))

print('accuracy',accuracy_score(target_y, pred_y))

cm = confusion_matrix(target_y, pred_y)

sns.heatmap(cm, annot=True)