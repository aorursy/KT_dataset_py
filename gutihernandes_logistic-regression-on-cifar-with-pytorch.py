# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import torch

from torch import nn, optim

import torchvision.datasets as dsets

import torchvision.transforms as transforms

from torch.autograd import Variable

import matplotlib.pyplot as plt 

import itertools

torch.manual_seed(2)

numpy.random.seed(2)

# Any results you write to the current directory are saved as output.
trainset = dsets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())

testset = dsets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())

mnist_train = dsets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
train_x = trainset.data

train_y = trainset.targets # e.g. 6. If it were trainset.classes then it will be 'airplane'



_to_pil = transforms.ToPILImage()



img = trainset.data[0]

img2 = trainset[0][0]

print(img.shape)

print(img2.shape)







plt.imshow(_to_pil(img))

#plt.imshow(_to_pil(img2))



def one_hot_embedding(labels, num_classes):

    """Embedding labels to one-hot form.



    Args:

      labels: (LongTensor) class labels, sized [N,].

      num_classes: (int) number of classes.



    Returns:

      (tensor) encoded labels, sized [N, #classes].

    """

    y = torch.eye(num_classes) 

    return y[labels]


class LogisticRegression(nn.Module):

    def __init__(self, input_dim, output_dim):

        super(LogisticRegression, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128).cuda()

        self.fc2 = nn.Linear(128, output_dim).cuda()

        self.relu = nn.ReLU().cuda()

        self.softmax = nn.Softmax().cuda()



        

    def forward(self, x):

        #input-to-hidden layer

        h1_out = self.fc1(x)

        h1_act_out = self.relu(h1_out)

        h2_out = self.fc2(h1_act_out)

        out = self.softmax(h2_out)

        return out

      

    def train(self, x, y, batchsize, epochs, optimiser, criterion, val_x, val_y):

      x_len = len(x)

      acc_hist =[]

      acc_hist_val =[]

      loss_hist=[]

      loss_hist_val=[]

      

      for epoch in range(epochs):

        print("Epoch {} is started.".format(epoch))        

        for i in range(0, x_len, batchsize):



          batch_x = x[i:i+batchsize]

          batch_y = y[i:i+batchsize]

          

          optimiser.zero_grad()

          outs = self.forward(batch_x)

          loss = criterion(outs, batch_y)

          loss.backward()

          optimiser.step()

          if (i/batchsize)%10 == 0:

            print("Loss for iteration {} on training set: {}".format(i, loss))

            loss_hist.append(loss)

            acc,cm=self.accuracy(batch_x,batch_y)

            acc_hist.append(acc)

            

        acc_val,cm_val = self.accuracy(val_x, val_y)

        acc_hist_val.append(acc_val)

        loss_val=criterion(self.forward(val_x),val_y)

        loss_hist_val.append(loss_val)

        print("Epoch {} is finished.".format(epoch))

        print("Loss and accuracy for this epoch on validation set: loss:{}, acc:{}".format(loss_val,acc_val))

        #print ("Confusion Matrix of validation set: (rows as predictions - columns as true labels)")

        #print(cm_val)

      return loss_hist, acc_hist, loss_hist_val, acc_hist_val

      

      

    def predict(self, test_x, test_y):

        out = self.forward(test_x)

        largest_index = np.argmax(out.cpu().detach().numpy())

        result = np.zeros((np.shape(out)[0]))

        result[largest_index] = 1

        #return result

        return largest_index

     

    def accuracy(self, x_val, y_val):

        correct = 0

        cm=np.zeros((10,10))

        for index in range(0,np.shape(x_val)[0]):



            prediction = int(self.predict(x_val[index],y_val[index]))



            label = np.argmax(y_val[index].cpu().detach().numpy())

            cm[prediction][label] = cm[prediction][label] + 1 

            is_equal = np.array_equal(prediction, label)

            if prediction == label:

                correct += 1

        #print("Accuracy: {}".format(correct/np.shape(x_val)[0]))

        return (correct/np.shape(x_val)[0]), cm

input_dim = 32*32*3

output_dim = 10



train_x = trainset.data

#train_y = torch.tensor(trainset.targets).long().cuda()



test_x = testset.data

test_x = torch.from_numpy(test_x.reshape([test_x.shape[0], -1])/255).float().cuda()

test_y = torch.tensor(testset.targets).long()

train_y = one_hot_embedding(torch.tensor(trainset.targets),10).cuda()

test_y = one_hot_embedding(torch.tensor(testset.targets),10).cuda()



train_x = torch.from_numpy(train_x.reshape([train_x.shape[0], -1])/255).float().cuda()



model = LogisticRegression(input_dim, output_dim)

for name,x in (model.named_parameters()):

  print (name, x)

optimiser = optim.SGD(model.parameters(),lr=0.1,momentum=0.9)

criterion = nn.MSELoss()



loss_hist, acc_hist, loss_hist_val, acc_hist_val = model.train(train_x, train_y, 128, 200, optimiser, criterion,test_x,test_y)

split_x = np.split(train_x.cpu(),2)

loss_hist, acc_hist, loss_hist_val, acc_hist_val

plt.plot(loss_hist, color="r", label="Loss-Training")

plt.plot(acc_hist, color="b", label="Accuracy-Training")

plt.plot(loss_hist_val, color="g", label="Loss-Validation")

plt.plot(acc_hist_val, color="cyan", label="Accuracy-Validation")

plt.legend()
#returns an accuracy value for each hyperparameter setting. Accuracy value is calculated as a mean of each accuracy value of each folding iteration

input_dim = 32*32*3

output_dim = 10

def cross_validation(k, hyperparameters, train_x, train_y,epochs):

  criterion = nn.MSELoss()

  scores_hyperparameter_settings=[]

  

  for setting in hyperparameters:

    

    l_rate = setting[0]

    optimiser=setting[1]

    batchsize=setting[2]

    

    



    scores_k_fold=0

    val_set_size = len(train_y) / k

    val_set_size = int(val_set_size)

    print("Testing for hyperparameter setting: {}".format(setting))

    i = 0

    for x in range(k):

      model_2 = LogisticRegression(input_dim, output_dim)

      if optimiser == "sgd":

        optimiser = optim.SGD(params=model_2.parameters(), lr=l_rate, momentum=0.9)

      elif optimiser =="adam":

        optimiser = optim.Adam(params=model_2.parameters(), lr=l_rate)

      val_fold_x = train_x[i*val_set_size:(i+1)*val_set_size]

      rest_fold_x_first = train_x[0:(i+1)*val_set_size]

      rest_fold_x_last = train_x[(i+1)*val_set_size:len(train_y)]



      val_fold_y = train_y[i*val_set_size:(i+1)*val_set_size]

      rest_fold_y_first = train_y[0:(i+1)*val_set_size]

      rest_fold_y_last = train_y[(i+1)*val_set_size:len(train_y)]





      if (i*val_set_size == 0):

        rest_fold_y = rest_fold_y_last

        rest_fold_x = rest_fold_x_last

      elif (i+1)*val_set_size == len(train_y):

        

        rest_fold_y = rest_fold_y_first

        rest_fold_x = rest_fold_x_first

      else:

        rest_fold_y = torch.cat((rest_fold_y_first, rest_fold_y_last))

        rest_fold_x = torch.cat((rest_fold_x_first, rest_fold_x_last))

      

      

      loss_hist, acc_hist, loss_hist_val, acc_hist_val = model_2.train(rest_fold_x, rest_fold_y, batchsize, epochs,optimiser,criterion,val_fold_x, val_fold_y )

      scores_k_fold += acc_hist_val[len(acc_hist_val)-1]

      i += 1

    scores_k_fold = scores_k_fold / k

    

    scores_hyperparameter_settings.append(scores_k_fold)

  print("Accuracy values on validation set (k-folded) for each hyperparameter setting: ", scores_hyperparameter_settings)

      

    #model.train()

  return scores_hyperparameter_settings



l_rates=[0.1, 0.01, 0.001]

optimiser=["sgd"]

#optimiser=["sgd", "adam"]

batchsize=[128,512,4096]

hyperparameters = [x for x in itertools.product(l_rates, optimiser, batchsize)]



print("hyperparameters: ", hyperparameters)



x = cross_validation(10, hyperparameters, train_x, train_y,50)

print(x)