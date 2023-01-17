%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import *
from torch.autograd import Variable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import *


# prepare data
def load_data(data_dir, row_no):
    data = pd.read_csv(data_dir+"train.csv",dtype =np.float32)
    print(data.shape)
    X = data.values[:row_no,1:]/255.0
    Y = data.values[:row_no,0]
    test_x = pd.read_csv(data_dir+"test.csv",dtype =np.float32).values[:]/255.0
    return X, Y, test_x

# read data
data_dir = '../input/'
train_x, train_y, test_x = load_data(data_dir,42000)
print(train_x.shape, train_y.shape, test_x.shape)
plt.imshow(train_x[233].reshape(28,28))
plt.show()

# split the train dataset
from sklearn.model_selection import train_test_split
x_train, x_vali, y_train, y_vali = train_test_split(train_x, train_y, test_size = 0.3, random_state = 0)
print(x_train.shape, x_vali.shape, y_train.shape, y_vali.shape)

batch_size = 100
train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train),torch.from_numpy(y_train).type(torch.LongTensor)),
                          batch_size=batch_size,
                          shuffle=False)
test_loader = DataLoader(TensorDataset(torch.from_numpy(x_vali),torch.from_numpy(y_vali).type(torch.LongTensor)),
                         batch_size=batch_size,
                         shuffle=False)

# Define CNN Model (LeNet)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Layer 1
        self.conv1 = nn.Sequential(
                     nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layer 3
        self.fc1 = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU()
        )

        # Layer 4
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

        # Layer 5
        self.fc3 = nn.Sequential(
            nn.Linear(84, 10),
            nn.ReLU()
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x



# Cross Entropy Loss Function
error = nn.CrossEntropyLoss()

# SGD Optimizer
model = CNNModel()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# CNN model training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
import time
time_list = []
stime = time.time()

for epoch in range(50):

    for i, (images, labels) in enumerate(train_loader):

        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)

        # Clear gradients
        optimizer.zero_grad()

        # Forward propagation
        outputs = model(train)

        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)

        # Calculating gradients
        loss.backward()

        # Update parameters
        optimizer.step()
        count += 1
        if count % 50 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:
                test = Variable(images.view(100, 1, 28, 28))

                # Forward propagation
                outputs = model(test)

                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]

                # Total number of labels
                total += len(labels)

                correct += (predicted == labels).sum()

            accuracy = 100 * correct / float(total)

            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

            # calculate the time cost
            etime = time.time() - stime
            time_list.append(etime)
            stime = time.time()

            if count % 500 == 0:

                # Print Loss
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.item(), accuracy))


# visualization loss
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("Loss via Number of iteration")
plt.show()

# visualization accuracy
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy via Number of iteration")
plt.show()

# visualization accuracy
plt.plot(iteration_list,time_list,color = "purple")
plt.xlabel("Number of iteration")
plt.ylabel("Time")
plt.title("Time Cost via Number of iteration")
plt.show()


# visualization loss
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("Loss via Number of iteration")
plt.show()

# visualization accuracy
plt.plot(iteration_list,accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("Accuracy via Number of iteration")
plt.show()

# visualization accuracy
plt.plot(iteration_list,time_list,color = "purple")
plt.xlabel("Number of iteration")
plt.ylabel("Time")
plt.title("Time Cost via Number of iteration")
plt.show()

def prediciton(data_loader):
    model.eval()
    test_pred = torch.LongTensor()

    for i, (data,tmp) in enumerate(data_loader):
        indata = Variable(data.view(100,1,28,28))
        output = model(indata)
        pred = torch.max(output.data, 1)[1]
        test_pred = torch.cat((test_pred, pred), dim=0)

    return test_pred
stime = time.time()
test_pred = prediciton(DataLoader(TensorDataset(torch.from_numpy(test_x),torch.from_numpy(test_x)),
                                  batch_size=100,
                                  shuffle=False))
print('Test Time Cost %.2f'%(time.time()-stime))
out_df = pd.DataFrame(np.c_[np.arange(1, len(test_x)+1)[:,None], test_pred.numpy()],
                      columns=['ImageId', 'Label'])
out_df.to_csv('submission.csv', index=False)
