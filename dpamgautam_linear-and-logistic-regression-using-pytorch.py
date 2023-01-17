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
import numpy as np



array = [[1,2,3],[4,5,6]]

first_array = np.array(array)



print("type of array : {}".format(type(first_array)))

print("shape of array : {}".format(np.shape(first_array)))

print(first_array)
import torch



tensor = torch.Tensor(array)



print("type of array : {}".format(tensor.type))

print("shape of tensor : {}".format(tensor.shape))

print(tensor)
print("numpy ones \n {}\n".format(np.ones((2,3))))



print("tensor ones \n {}".format(torch.ones((2,3))))
# numpy random

print("Numpy random \n {}\n".format(np.random.rand(2,3)))



# pytorch random

print("tensor random \n {}".format(torch.rand(2,3)))
# numpy array to tensor conversion



numpy_array = np.random.rand(2,3)



print("type of numpy array {} \n{}".format(type(numpy_array), numpy_array))



tensor_from_numpy = torch.from_numpy(numpy_array)



print("type of tensor array {} \n{}".format(tensor_from_numpy.type, tensor_from_numpy))
# tensor to numpy array conversion



tensor = torch.Tensor(2,3)



print("type of tensor {} \n{}".format(tensor.type, tensor))



numpy_from_tensor = tensor.numpy()



print("type of numpy array {} \n{}".format(type(numpy_from_tensor), numpy_from_tensor))
# define two tensor



t = torch.ones(2,3)



# resize



print("{}\n{}\n".format(t,t.view(3,2)))



# addition



print("{}\n".format(torch.add(t,t)))



# subtraction



print("{}\n".format(t.sub(t)))



# elementwise multiplication



print("{}\n".format(torch.mul(t,t)))



# elementwise division



print("{}\n".format(torch.div(t,t)))



# mean



tt = torch.Tensor([1,2,3,4,5,6,7,8])



print("mean of tt is {}\n".format(tt.mean()))



# standard deviation



print("std of tt is {}\n".format(tt.std()))
# import variables from pytorch library



from torch.autograd import Variable



var = Variable(torch.ones(3), requires_grad = True)



var
# define car prices



car_prices_array = [3,4,5,6,7,8,9]

car_prices_array_np = np.array(car_prices_array, dtype=np.float32)

car_prices_array_np = car_prices_array_np.reshape(-1,1)

car_prices_tensor = Variable(torch.from_numpy(car_prices_array_np))



# define number of sold cars



car_sells_array = [7.5, 7, 6.5, 6, 5.5, 5, 4.5]

car_sells_array_np = np.array(car_sells_array, dtype=np.float32)

car_sells_array_np = car_sells_array_np.reshape(-1,1)

car_sells_tensor = Variable(torch.from_numpy(car_sells_array_np))



# visualize the data



import matplotlib.pyplot as plt



plt.scatter(car_prices_array, car_sells_array)

plt.xlabel("car prices in $")

plt.ylabel("number of sold cars")

plt.title("car prices vs number of car sold")

plt.show()
# linear regression with pytorch



import torch

from torch.autograd import Variable

import torch.nn as nn



import warnings

warnings.filterwarnings("ignore")



# create class



class LinearRegression(nn.Module):

    

    def __init__(self, input_size, output_size):

        

        # super function

        # it inherits from nn.Module

        # and we can accces everything in nn.Module

        super(LinearRegression, self).__init__()

        

        # linear function

        self.linear = nn.Linear(input_dim, output_dim)

        

    def forward(self, x):

        return self.linear(x)

    

# define model



input_dim = 1

output_dim = 1



model = LinearRegression(input_dim, output_dim)



#  calculating the mean squared error loss

mse = nn.MSELoss()



# optimization

learning_rate = 0.02

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)



# train model

loss_list = []

iteration = 1000



for it in range(iteration):

    

    #optimization step

    optimizer.zero_grad()

    

    # forward pass

    result = model(car_prices_tensor)

    

    # calculate loss

    loss = mse(result, car_sells_tensor)

    

    # backward propagation

    loss.backward()

    

    # updating parameters

    optimizer.step()

    

    # appending the loss in a list

    loss_list.append(loss.data)

    

    # printing the loss every 100th iteration

    if (it%100 == 0):

        print("epoch: {} ==> loss: {}\n".format(it,loss.data))

    

plt.plot(range(iteration), loss_list)

plt.xlabel("Number of Iterations")

plt.ylabel("Loss")

plt.show()
# predict the car prices



predicted = model(car_prices_tensor).data.numpy()



plt.scatter(car_prices_array, car_sells_array, label="original sell for car prices", color="red")

plt.scatter(car_prices_array, predicted, label="predicted sell for car prices", color="yellow")

plt.legend()

plt.xlabel("Car Price $")

plt.ylabel("Number of Car Sell")

plt.title("Original vs Predicted values")

plt.show()

plt.show()
# now predict if the car price is 10$, what is the expected number of sell



pred_10 = model(torch.from_numpy(np.array([10], dtype=np.float32))).data.numpy()

print(pred_10)
# import libraries



import torch

import torch.nn as nn

import torchvision.transforms as transforms

from torch.autograd import Variable



from sklearn.model_selection import train_test_split
# load and prepare dataset



train = pd.read_csv("../input/train.csv", dtype=np.float32)



# split data into features(pixels values) and targets(numbers from 0 to 9)



features_numpy = train.loc[:, train.columns != "label"].values/255

targets_numpy = train.label.values



# split the dataset, training set size 80% and test set 20%



features_train, features_test, targets_train, targets_test = train_test_split(features_numpy, targets_numpy, test_size=0.2, random_state=42)



# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable



featuresTrain = torch.from_numpy(features_train)

targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)



featuresTest = torch.from_numpy(features_test)

targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)



# batch size, epoch and iteration



batch_size = 100

n_iters = 10000

num_epochs = n_iters/(len(features_train)/batch_size)

num_epochs = int(num_epochs)



# pytorch train and test dataset



train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)

test = torch.utils.data.TensorDataset(featuresTest, targetsTest)



# data loader



train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
# visualize one of the images in the dataset



plt.imshow(features_numpy[10].reshape(28,28))

plt.axis("off")

plt.title(str(targets_numpy[10]))

plt.savefig("graph.png")

plt.show()
# create logistic regression model



class LogisticRegressionModel(nn.Module):

    

    def __init__(self, input_dim, output_dim):

        super(LogisticRegressionModel, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)

        

    def forward(self, x):

        return self.linear(x)

    

input_dim = 28*28

output_dim = 10



model = LogisticRegressionModel(input_dim, output_dim)



error = nn.CrossEntropyLoss()



learning_rate = 0.001

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
print(num_epochs)
# training the model



count = 0

loss_list = []

iteration_list = []



for epoch in range(num_epochs):

    for i, (images,labels) in enumerate(train_loader):

        

        # define variable

        train = Variable(images.view(-1, 28*28))

        labels = Variable(labels)

        

        optimizer.zero_grad()

        

        # forward propagation

        outputs = model(train)

        

        loss = error(outputs, labels)

        

        loss.backward()

        

        # update the parameters

        optimizer.step()

        

        count += 1

        

        # prediction

        if count % 50 == 0:

            correct = 0

            total = 0

            

            for images,labels in test_loader:

                test = Variable(images.view(-1, 28*28))

                outputs = model(test)

                predicted = torch.max(outputs.data, 1)[1]

                # total number of labels

                total += len(labels)

                # total correct prediction

                correct += (predicted==labels).sum()

                

            accuracy = (correct/float(total))*100

            

            # store loss and iteration

            loss_list.append(loss.data)

            iteration_list.append(count)

            

        if count%500 == 0:

            print("iteration : {}, loss : {}, accuracy : {}".format(count, loss.data, accuracy))
# visualization

plt.plot(iteration_list,loss_list)

plt.xlabel("Number of iteration")

plt.ylabel("Loss")

plt.title("Logistic Regression: Loss vs Number of iteration")

plt.show()