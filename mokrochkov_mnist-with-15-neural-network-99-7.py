# usual imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import time



# torch and torchvision in order to build the neural network

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torchvision

import torchvision.transforms as transforms

import torchvision.transforms.functional as Func



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# define neural network architecture

# based on VGG style neural network



class Net(nn.Module):



    def __init__(self):

        super(Net, self).__init__()

        # 1 input image channel, 6 output channels, 5x5 square convolution

        # kernel

        

        # N x N x 1 --> N x N x 64

        self.conv_1_64 = nn.Conv2d(1, 64, 3, padding=1)

        

        # N x N x 64 --> N x N x 64

        self.conv_64_64 = nn.Conv2d(64,64,3,padding = 1) 

        

        # N x N x 64 --> N x N x 128

        self.conv_64_128 = nn.Conv2d(64, 128, 3, padding = 1) 

        

        # N x N x 128 --> N x N x 128

        self.conv_128_128 = nn.Conv2d(128,128,3,padding = 1)

        

        # N in case of MNIST is 7, so we go from 7 x 7 x 128 to a 1024, with further layers of 512, 256, 128 and finally 10

        self.fc1 = nn.Linear(128 * 7 * 7, 1024)

        self.fc2 = nn.Linear(1024, 512)

        self.fc3 = nn.Linear(512, 256)

        self.fc4 = nn.Linear(256, 128)

        self.fc5 = nn.Linear(128, 10)

        

        # initiate dropout layer with a given probability

        self.dropout = nn.Dropout(p=0.5);

        

        # initiate batch normalizations

        self.batch_norm_64 = nn.BatchNorm2d(64)

        self.batch_norm_128_2d = nn.BatchNorm2d(128)

        self.batch_norm_128_1d = nn.BatchNorm1d(128)

        self.batch_norm_256 = nn.BatchNorm1d(256)

        self.batch_norm_512 = nn.BatchNorm1d(512)

        self.batch_norm_1024 = nn.BatchNorm1d(1024)



    def forward(self, x):

        # input --> CNN 1, final dimensions 28 x 28 x 64

        x = F.relu(self.conv_1_64(x))

        x = self.batch_norm_64(x)

        

        # CNN 1 --> CNN 2 --> CNN 3, final dimensions 28 x 28 x 64

        x = self.batch_norm_64(F.relu(self.conv_64_64(x)))

        x = self.batch_norm_64(F.relu(self.conv_64_64(x)))

        

        # max pool, final dimensions 14 x 14 x 64

        x = F.max_pool2d(x, (2, 2))

        

        # CNN 3 --> CNN 4 --> CNN 5 --> CNN 6, final dimensions 14 x 14 x 128

        x = self.batch_norm_128_2d(F.relu(self.conv_64_128(x)))

        x = self.batch_norm_128_2d(F.relu(self.conv_128_128(x)))

        x = self.batch_norm_128_2d(F.relu(self.conv_128_128(x)))

        

        # max pool, final dimensions 7 x 7 x 128

        x = F.max_pool2d(x, (2, 2))

        

        # convert to a vector

        x = x.view(-1, self.num_flat_features(x))

        

        # fully connected layers

        x = self.dropout(self.batch_norm_1024(F.relu(self.fc1(x))))

        x = self.dropout(self.batch_norm_512(F.relu(self.fc2(x))))

        x = self.dropout(self.batch_norm_256(F.relu(self.fc3(x))))

        x = self.dropout(self.batch_norm_128_1d(F.relu(self.fc4(x))))

        x = self.fc5(x)

        return x



    def num_flat_features(self, x):

        size = x.size()[1:]  # all dimensions except the batch dimension

        num_features = 1

        for s in size:

            num_features *= s

        return num_features





netCNN = Net()



# select device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(netCNN.parameters(), lr=0.001)

netCNN.to(device)
num_networks = 15

batchsize = 500

num_epochs = 100



img_rows = 28

img_cols = 28

# for convolutional neural network

x_train = pd.read_csv('../input/train.csv')

x_test = pd.read_csv('../input/test.csv')

y_train = x_train['label'].values

x_train = x_train.drop(['label'],1).values

x_test = x_test.values



x_train = x_train.reshape(np.shape(x_train)[0],1,img_rows,img_cols)

x_test = x_test.reshape(np.shape(x_test)[0],1,img_rows,img_cols)



x_train = x_train.astype('float32')/255

x_test = x_test.astype('float32')/255



train = torch.from_numpy(x_train)

test = torch.from_numpy(x_test)

train_label = torch.from_numpy(y_train)

train_label = train_label.long()
affine = transforms.RandomAffine(degrees=10, translate=(.20,.20), scale=(.9,1.1))

output_all = torch.empty(test.size(0),10,num_networks)

num_iterations_train = int(train.size(0)/batchsize)

num_iterations_test = int(test.size(0)/batchsize)



time_start = time.time()



for n_network in range(num_networks):

    train_temp = torch.empty(np.shape(x_train)[0],1,img_rows,img_cols)

    netCNN = Net()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(netCNN.parameters(), lr=0.001)

    netCNN.to(device)

    for epoch in range(num_epochs):

        # it seems that these have to be done one at a time

        for n_input, input_data in enumerate(train):

            train_pil = transforms.ToPILImage(mode=None)(train[n_input])

            train_temp[n_input,:,:,:] = Func.to_tensor(affine(train_pil))



        running_loss = 0.0

        for i in range(num_iterations_train):



            inputs = train_temp[i*batchsize:(i+1)*batchsize]

            labels = train_label[i*batchsize:(i+1)*batchsize]

            inputs = inputs.to(device)

            labels = labels.to(device)



            # zero the parameter gradients

            optimizer.zero_grad()



            # forward + backward + optimize

            outputs = netCNN(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()



            # print statistics

            running_loss += loss.item()

        #print('Epoch %d loss: %.3f' % (epoch + 1, running_loss/batchsize*1000))

        running_loss = 0.0



    print('Finished Training Network %i' % (n_network+1))



    prediction_all = torch.empty(train.size(0),1)

    for i in range(num_iterations_train):

        test_input = train[i*batchsize:(i+1)*batchsize]

        output = netCNN(test_input.to(device))

        _, prediction = torch.max(output,1)

        prediction_all[i*batchsize:(i+1)*batchsize,0] = prediction.cpu()



    match = train_label.int().reshape(train.size(0),1) == prediction_all.int()

    fraction = int(match.sum())/int(prediction_all.shape[0])

    print(fraction)

    

    #prediction_all = torch.empty(test.size(0),1)

    for i in range(num_iterations_test):

        test_input = test[i*batchsize:(i+1)*batchsize]

        output = netCNN(test_input.to(device))

        output_all[i*batchsize:(i+1)*batchsize,:,n_network] = output.data

        del output

        del test_input

        torch.cuda.empty_cache()



a,_ = torch.max(output_all,2)

_,predicted_final = torch.max(a,1)



runtime = time.time() - time_start

print('Elapsed Time: %.3f seconds' % runtime)
output_file = 'submission.csv'

pred_final_numpy = predicted_final.numpy()

with open(output_file, 'w') as f :

    f.write('ImageId,Label\n')

    for i in range(len(predicted_final)) :

        f.write("".join([str(i+1),',',str(pred_final_numpy[i]),'\n']))