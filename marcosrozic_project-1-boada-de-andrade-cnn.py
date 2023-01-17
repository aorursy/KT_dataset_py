# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import numpy as np # linear algebra

from numpy import random # random choice



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time # for process execution time



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Pytorch Library

import torch

from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SubsetRandomSampler

from torch.autograd import Variable

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



from   torchvision import datasets

import torchvision.transforms as transforms



#Matplolib Library

import matplotlib.pyplot as plt

%matplotlib inline



train_on_gpu = torch.cuda.is_available()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size    = 64    # How many samples per batch to load

epochs        = 15    # How many times will the model train

lr            = 0.001 # Learning Rate

valid_size    = 0.1   # % of the train set that is assigned for the valid_loader

dropout_rate  = 0.25  # Chance for dropping out neural network units
class DatasetMNIST(Dataset):

    def __init__(self, file_path, shape=(1,28,28), test=False):

        self.data  = pd.read_csv(file_path)

        self.shape = shape

        

        if test:

            zeros = np.zeros(self.data.shape[0])

            self.data.insert(0, 'label', pd.Series(zeros))

            

        print("Path file: " + str(file_path))

        print(self.data.shape)



    def __len__(self):

        return len(self.data)



    def __getitem__(self, index):

        image = self.data.iloc[index, 1:].values

        image = (image-0.5)/0.5

        image = image.astype(np.float32)

        image = image.reshape(self.shape)

        label = self.data.iloc[index, 0]

        

        image = torch.from_numpy(image).float()

        



        return image, label
train_data = DatasetMNIST(file_path="../input/digit-recognizer/train.csv")

test_data  = DatasetMNIST(file_path="../input/digit-recognizer/test.csv", test=True)
# Split to get Train and Validation Data

num_train = len(train_data)

indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]



train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)



# Set Data Loaders

train_loader  = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)

valid_loader  = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)

test_loader   = DataLoader(test_data, batch_size=batch_size)



print("Train Data Length: {}".format(int(len(train_data) - len(train_data)*valid_size)))

print("Valid Data Length: {}".format(int(len(train_data) * valid_size)))

print("Test  Data Length: {}".format(len(test_data)))
dataiter       = iter(train_loader)

images, labels = dataiter.next()



fig = plt.figure(figsize=(25, 4))

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    ax.imshow(np.squeeze(images[idx]), cmap='gray')

    ax.set_title(str(labels[idx].item()))
''' 

    Layers

    - hi: hidden layer for the i-th fully connect layer

    - conv_bni: batch normalization for the i-th convolutional layer

    - out: output -> 10 units

'''

class Net(nn.Module):

    

    def __init__(self, h1=950, h2=300, h3=50, out=10, dropout_rate=0.2):

        super(Net, self).__init__()



        #Convolutional Layers

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=15,kernel_size=(5,5), padding=(2,2))

        self.conv2 = nn.Conv2d(in_channels=15,out_channels=60,kernel_size=(5,5),padding=(1,1))

        

        self.maxPool  = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))



        #Fully Connect Layers

        self.fc1 = nn.Linear(60*6*6, h1)

        self.fc2 = nn.Linear(h1, h2)

        self.fc3 = nn.Linear(h2, h3)

        self.fc4 = nn.Linear(h3, out)



        #Batchnorm Convolutional Layers

        self.conv_bn1 = nn.BatchNorm2d(15)

        self.conv_bn2 = nn.BatchNorm2d(60)



        self.dropout = nn.Dropout(dropout_rate)



    def forward(self, x):



        #Convolutional Layers

        x = self.conv1(x)

        x = self.conv_bn1(x)

        x = self.maxPool(F.relu(x))

        x = self.dropout(x)



        x = self.conv2(x)

        x = self.conv_bn2(x)

        x = self.maxPool(F.relu(x))

        x = self.dropout(x)



        #Flatten

        x = x.view(-1, 60*6*6)

        

        #Fully Connected Layers



        x = self.fc1(x)

        x = F.relu(x)

        x = self.dropout(x)



        x = self.fc2(x)

        x = F.relu(x)

        x = self.dropout(x)



        x = self.fc3(x)

        x = F.relu(x)

        x = self.dropout(x)





        x = self.fc4(x)

        x = F.log_softmax(x, dim=1)



        return x



Net()
#Method Xavier

def weights_init_xavier_uniform(m):

    classname = m.__class__.__name__

    #for every Lineal layer in a model

    if classname.find('Linear')!=-1:

        nn.init.xavier_uniform_(m.weight)

        m.bias.data.fill_(0)



#Method He

def weights_init_he_uniform(m):

    classname = m.__class__.__name__

    #for every Lineal layer in a model

    if classname.find('Linear')!=-1:

        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

        m.bias.data.fill_(0)
def model_evaluation(net, epochs=10, lr=0.001, optim_method="Adam"):

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr) if optim_method == "SGD" else optim.Adam(net.parameters(),lr=lr)



    #-- Initialize Lists --#

    train_losses     = []

    valid_losses     = []

    train_accuracies = []

    valid_accuracies = []

  

    time_av = 0 # acumulator for training average time

    for epoch in range(epochs):

        

        train_loss     = 0

        train_accuracy = 0



        ####### TRAINING ########

        start_time = time.time()

        net.train()

        for input, label in train_loader:

            

            input, label = input.to(device), label.to(device) # transfer data to current device (cuda/cpu)



            optimizer.zero_grad()                                               

            output = net(input)

            loss   = criterion(output, label)

            loss.backward()

            optimizer.step()

            

            #-- Accuracy --#

            ps = torch.exp(output)

            top_p, top_class = ps.topk(1, dim=1)

            equals = top_class == label.view(*top_class.shape)



            train_accuracy += torch.mean(equals.type(torch.FloatTensor))

            train_loss     += loss.item()



        ######## VALIDATION #######

        else:

            valid_loss     = 0

            valid_accuracy = 0

            with torch.no_grad():

                net.eval()

                for input, label in valid_loader:

                    

                    input, label = input.to(device), label.to(device) # transfer data to current device (cuda/cpu)

                    

                    output = net(input)

                    loss   = criterion(output, label)

                    

                    #-- Accuracy --#

                    ps = torch.exp(output)                    

                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == label.view(*top_class.shape) 

                    

                    valid_accuracy += torch.mean(equals.type(torch.FloatTensor))

                    valid_loss     += loss.item()

            

            #-- Save Current Accuracies and Losses --#

            train_losses.append(train_loss/len(train_loader))

            valid_losses.append((valid_loss/len(valid_loader)))

            train_accuracies.append( (train_accuracy/len(train_loader)) )

            valid_accuracies.append( (valid_accuracy/len(valid_loader)) )

            

            exe_time = time.time() - start_time # epoch executed time

            time_av += exe_time # save current executed time

            

            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining Accuracy: {:.6f}% \tValid Accuracy: {:.6f}% \tExcecuted Time: {:.3f}s'.format(

              epoch+1,

              train_loss/len(train_loader),

              valid_loss/len(valid_loader),

              train_accuracy*100/len(train_loader),

              valid_accuracy*100/len(valid_loader),

              exe_time

              ))

      

    valid_acc = valid_accuracy / len(valid_loader) # process mean accuracy



    return (train_losses, valid_losses, train_accuracies, valid_accuracies, valid_acc, time_av/epochs)
model = Net(dropout_rate=dropout_rate)

model.apply(weights_init_xavier_uniform)

model.to(device) # transfer model to curren device (cuda/device)

train_losses, valid_losses, train_accuracies, valid_accuracies, valid_acc, time_av = model_evaluation(net=model, epochs=epochs, lr=lr)
def print_performance(epochs, train_loss, valid_acc, time):

    print("After {} epochs:".format(epochs))

    print("Train Loss: {:.6f}".format(train_loss))

    print("Validation Accuracy: {:.6f}%".format(valid_acc*100))

    print("Average Time: {:.2f}s".format(time))



print_performance(epochs, train_losses[-1], valid_acc, time_av)
# plot loss and accuracy

def plot_loss_accuracy(title_1, title_2, top_list=[], bottom_list=[], x1='accuracy', y1='epoch', x2='accuracy', y2='epoch'):

    fig = plt.figure(figsize=(18, 6))

    colors = ['#0027F2','#FF6005','#E42F48 ','#8B44E4 ', '#F8DD00']

    plt.subplot(1,2,1)

    labels_loss = []

    for i, (model_loss, label_loss) in enumerate(top_list, 0):

        plt.plot(model_loss, color=colors[i])

        labels_loss.append(label_loss)      

    plt.title(title_1)

    plt.ylabel(y1)

    plt.xlabel(x1)

    plt.grid(linestyle='--')

    plt.legend(labels_loss, loc='upper right')



    plt.subplot(1,2,2)

    labels_accs = []

    for i, (model_accs, label_accs) in enumerate(bottom_list, 0):

        plt.plot(model_accs, color=colors[i])

        labels_accs.append(label_accs)

    plt.title(title_2)

    plt.ylabel(y2)

    plt.xlabel(x2)

    plt.grid(linestyle='--')

    plt.legend(labels_accs, loc='lower right')

    plt.tight_layout()



# plot prediction

def plot_prediction(ps):

    ps = ps.data.numpy().squeeze()



    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)



    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())

    ax1.axis('off')



    ax2.barh(np.arange(10), ps)

    ax2.set_aspect(0.1)

    ax2.set_yticks(np.arange(10))

    ax2.set_yticklabels(np.arange(10))

    ax2.set_title('Class Probability')

    ax2.set_xlim(0, 1.1)



    plt.tight_layout()
loss_list = [

      (train_losses, 'Training Loss'),

      (valid_losses, 'Validation Loss')

]



accuracy_list = [

      (train_accuracies, 'Training Accuracy'),

      (valid_accuracies, 'Validation Accuracy')

]



plot_loss_accuracy(title_1="Plot 1.1 - Model Loss", title_2='Plot 1.2 - Model Accuracy', top_list=loss_list, bottom_list=accuracy_list, x1='epoch', y1='loss', x2='epoch', y2='accuracy')
# Visualize Test Data

images,_ = next(iter(test_loader))

img = images[random.randint(len(images)-1)]



plt.imshow(np.squeeze(image))
images,_ = next(iter(test_loader))

img = images[random.randint(len(images)-1)].reshape(1,1,28,28)

with torch.no_grad():

    logps = model(img.to(device))



ps = torch.exp(logps).cpu()

plot_prediction(ps)
correct = 0

total = 0

labels = []

with torch.no_grad():

    model.eval()

    for input,_ in test_loader:

        input = input.to(device)

        outputs = model(input)

        _, predicted = torch.max(outputs.data, 1)

        

        for i in range(len(predicted)):        

            labels.append(predicted[i].cpu().numpy())
# Save Predictions in data frame

df = pd.DataFrame(data={'ImageId':range(1,len(test_data)+1), 'Label':labels}) # submission dataframe

print(df)
df.to_csv("submission.csv", index=False)