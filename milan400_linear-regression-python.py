import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
#create dummy data for training

x_values = [i for i  in range(11)]

x_train = np.array(x_values, dtype = np.float32)

x_train = x_train.reshape(-1,1)



y_values = [2*i+1 for i in x_values]

y_train = np.array(y_values,dtype=np.float32)

y_train = y_train.reshape(-1,1)
import torch

from torch.autograd import Variable
class LinearRegression(torch.nn.Module):

    def __init__(self,inputsize,outputsize):

        super(LinearRegression,self).__init__()

        self.linear = torch.nn.Linear(inputsize, outputsize)

        

    def forward(self, x):

        out = self.linear(x)

        return(out)
inputdim = 1

outputdim = 1

learningrate = 0.01

epochs = 100
model = LinearRegression(inputdim, outputdim)

#for GPU

if(torch.cuda.is_available()):

    model.cuda()
criterion = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(),lr = learningrate)
for epoch in range(epochs):

    #Converting inputs and lables to Varibale

    if(torch.cuda.is_available()):

        inputs = Variable(torch.from_numpy(x_train).cuda())

        labels = Variable(torch.from_numpy(y_train).cuda())

    else:

        inputs = Variable(torch.from_numpy(x_train))

        labels = Variable(torch.from_numpy(y_train))

    # Clear gradient buffers because we don't want any gradient from previous

    #epoch to carry forward,dont want to cummulate gradients

    optimizer.zero_grad()

    

    #get output from the model, given the inputs

    outputs = model(inputs)

    

    #get loss for predicted output

    loss = criterion(outputs, labels)

    print(loss)

    

    #get gradients w.r.t to parameters

    loss.backward()

    

    #update parameters

    optimizer.step()

    

    print('epochs {}, loss {}'.format(epoch, loss.item()))
with torch.no_grad():

    if(torch.cuda.is_available()):

        #no need for gradients in testing phase

        predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()

    else:

        predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()

    print(predicted)
plt.clf()

plt.plot(x_train, y_train,'go',label='True data', alpha=0.5)

plt.plot(x_train, predicted,'go',label='predicted', alpha=0.5,color='red')

plt.legend(loc='best')

plt.show()