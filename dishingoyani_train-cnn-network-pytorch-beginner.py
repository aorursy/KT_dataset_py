# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib import pyplot as plt

from matplotlib import image as img



import torch

import torch.nn as nn

import torch.nn.functional as F



from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

import torch.optim as optim



from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# set all dir

data_dir = "../input/aerial-cactus-identification"            # main dir

train_dir = data_dir + "/train/train"                         # train images folder

test_dir = data_dir + "/test/test"                            # test images folder
# read train csv

data = pd.read_csv(data_dir + "/train.csv")

data.head()                                                   # test csv
# check ratio of images with cactus and without cactus

data["has_cactus"].value_counts()
# visualize ratio of images with cactus and without cactus

plt.hist(data.has_cactus,align='mid',bins=4)

plt.xlim(-1, 2)

plt.xlabel("0: has not cactus and 1: has cactus")

plt.ylabel("Frequency")

plt.title("Number of Images has cactus {1} and without cactus")
# inherit Dataset and make changes as we access images through index latter

class cactData(Dataset):

    def __init__(self, split_data, data_root = './', transform = None):

        super().__init__()

        self.df = split_data.values                       # set dataframe

        self.data_root = data_root                        # images path

        self.transform = transform                        # transform



    def __len__(self):

        return len(self.df)                               # return total length of dataframe 

    

    def __getitem__(self, index):

        img_name,label = self.df[index]                   # get image id and label from csv

        img_path = os.path.join(self.data_root, img_name) # set image path

        image = img.imread(img_path)                      # read image from given image path

        if self.transform is not None:                    # transform image if transform available 

            image = self.transform(image)

        return image, label                               # return image and image label
# split data into train and validation set; so training data = 80% and validation data = 20%

train, valid = train_test_split(data, stratify=data.has_cactus, test_size = 0.2)
# define transforms

train_transf = transforms.Compose([transforms.ToPILImage(),

                                   transforms.ToTensor()])



valid_transf = transforms.Compose([transforms.ToPILImage(),

                                  transforms.ToTensor()])
#  batch size 

batch_size = 128
# set cactData cons

train_data = cactData(train, train_dir, train_transf)

valid_data = cactData(valid, train_dir, valid_transf)



# get training and validation data

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size//2, shuffle=False)
# test data

images, labels = next(iter(train_loader))

images[0], labels[0]
# define CNN model

class Net(nn.Module): 

    def __init__(self):

        super(Net, self).__init__()

        

        # Convolutional Neural Networks

        self.conv1 = nn.Conv2d(3, 10, 3, padding=1)

        self.conv2 = nn.Conv2d(10, 20, 3, padding=1)

        

        # fully connected network

        self.fc1 = nn.Linear(1280, 640)

        self.fc2 = nn.Linear(640, 2)

        

        # pooling and dropout layer

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.25)

        

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        

        # reshape to fit into fully connected net

        x = x.view(x.shape[0],-1)

        x = F.relu(self.fc1(x))

        x = F.dropout(x, training=self.training)

        x = self.fc2(x)

        

        return x
# set hyper parameters

num_epochs = 20

learning_rate = 0.01

momentum = 0.9



# check if GPU available

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

device
# model architecture  

model = Net().to(device)

print(model)
# specify loss function (categorical cross-entropy)

criterion = nn.CrossEntropyLoss()



# SGD optimizer

optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum)
# to track validation loss

valid_loss_min = np.Inf



for epoch in range(1, num_epochs + 1):

    # keep track of training and validation loss

    train_loss = 0.0

    valid_loss = 0.0

    

    # training model

    model.train()

    for data, target in train_loader:

        # move tensors to GPU if CUDA is available

        data = data.to(device)

        target = target.to(device)

        

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the batch loss

        loss = criterion(output, target)

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update training loss

        train_loss += loss.item() * data.size(0)

        

    # validate the model

    model.eval()

    for data, target in valid_loader:

        # move tensors to GPU if CUDA is available

        data = data.to(device)

        target = target.to(device)

        

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the batch loss

        loss = criterion(output, target)

        # update average validation loss 

        valid_loss += loss.item() * data.size(0)

    

    # calculate average losses

    train_loss = train_loss/len(train_loader.sampler)

    valid_loss = valid_loss/len(valid_loader.sampler)

        

    # print training/validation statistics 

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

        epoch, train_loss, valid_loss))

    

    # save model if validation loss has decreased

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,

        valid_loss))

        torch.save(model.state_dict(), 'CNN_model.pt')

        valid_loss_min = valid_loss
# set best model params

model.load_state_dict(torch.load('CNN_model.pt'))
# check accuracy on validation dataset

model.eval()

with torch.no_grad():

    correct = 0

    total = 0

    for data, target in valid_loader:

        # move tensors to GPU if CUDA is available

        data = data.to(device)

        target = target.to(device)

        

        # forward pass: compute predicted outputs by passing inputs to the model

        outputs = model(data)

        

        # convert output probabilities to predicted class

        _, predicted = torch.max(outputs.data, 1)

        

        total += target.size(0)

        correct += (predicted == target).sum().item()

    print('Test Accuracy: {} %'.format(100 * correct / total))
# prepare data for submission

df_submission = pd.read_csv(data_dir + "/sample_submission.csv")



test_data = cactData(df_submission, test_dir, transform=valid_transf)

test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False)
model.eval()



pred_label = []

for data, target in test_loader:

    # move tensors to GPU if CUDA is available

    data, target = data.cuda(), target.cuda()

    

    # forward pass

    output = model(data)

    

    prob = output[:,1].detach().cpu().numpy()

    for p in prob:

        pred_label.append(p)



# Set predicted labels to submission

df_submission['has_cactus'] = pred_label

df_submission.to_csv('submission.csv', index=False)