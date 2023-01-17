# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

train_df = np.load(R"/kaggle/input/winter2020-mais-202/train_images.npy")

test_df = np.load(R"/kaggle/input/winter2020-mais-202/test_images.npy")



train_df = train_df /255

test_df = test_df /255

# train_data = np.array(train_df, dtype = 'float32')

# test_data = np.array(test_df, dtype='float32')



train_labels_CSV = R"/kaggle/input/winter2020-mais-202/train_labels.csv"



train_labels = np.array((pd.read_csv(train_labels_CSV, index_col = 'ID')) ).astype(int)





#labels



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

plt.imshow(train_df[55])

plt.show()
#Creating a dataset

import torch 

from torch.utils.data import Dataset, DataLoader





class fashionDataSet(Dataset):

    def __init__(self, data, target, transform =None):

        self.data = data.reshape(-1,28,28,1).astype('float32')

        self.target = target

        self.transform = transform

        

    def __getitem__ (self, index):

        image = self.data[index]

        label = self.target[index]

        

        if self.transform:

            image = self.transform(image)

            

        return image, label

    

    def __len__(self):

        return len(self.data)

    

    

        
import torchvision.transforms as transforms





val_percent = 0.2

val_size = int ( len (train_df) * val_percent)





train_x = train_df[:-val_size]

train_y = train_labels[:-val_size]



val_x = train_df[-val_size:]

val_y = train_labels[-val_size:]



BATCHSIZE = 100





train_set = fashionDataSet(train_x, train_y, transform = transforms.Compose([transforms.ToTensor()]))



val_set = fashionDataSet(val_x,val_y, transform = transforms.Compose([transforms.ToTensor()]))





train_loader = DataLoader(train_set, batch_size=BATCHSIZE)



val_loader = DataLoader (val_set, batch_size = BATCHSIZE)



import matplotlib.pyplot as plt



image, label = next (iter(train_set))



plt.imshow(image.squeeze(), cmap = "gray")

plt.show()




#view Image. For some reason I get black pixels



def show_image(arr):

    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)

    plt.imshow(two_d, interpolation='nearest')

    plt.show()

 # 0 is the index of the training image you want to display

print (train_df[2].shape)

plt.imshow(train_df[55])

plt.show()

show_image(train_df[55])
import torch.nn as nn 

import torch.nn.functional as F

import torch.optim as optim

from tqdm import tnrange, tqdm_notebook
class CNN (nn.Module):



  def __init__(self):

    super(CNN,self).__init__()



    self.conv_layer1 = nn.Sequential ( 

        nn.Conv2d(1,32, 3, 1,1),  #size

        nn.BatchNorm2d(32),   # BATCH NORM REG BEST REG

        nn.ReLU(inplace = True), 



        nn.MaxPool2d (2, stride=2)#size 14

    )



    self.conv_layer3 = nn.Sequential (

        

        nn.Conv2d(32,64,3,1),

        nn.BatchNorm2d(64),

        nn.ReLU(inplace = True),

        

        nn.MaxPool2d (2, stride=2) #results in size 6

    )



    self.fc1= nn.Linear(6*6*64, 1000 )

  #  self.drop = nn.Dropout2d(0.5)

    self.fc2 = nn.Linear(1000, 400)

    self.fc3 = nn.Linear ( 400, 10)



  def forward(self, x):

    out = self.conv_layer1(x)

  #  out = self.conv_layer2(out)

    out = self.conv_layer3(out)

    out = out.view (-1, 6*6*64)



    out = F.relu (self.fc1(out) )

 #   out = self.drop(out)

#     out = nn.BatchNorm2d(2048)

    

    out = F.relu(self.fc2(out))

    

#     out = nn.BatchNorm2d(1024)

    

    out = self.fc3(out)



    return out











import tqdm



from ipywidgets import FloatProgress





def fwd_pass(cnn, X, y, train = False):

    if train:

        cnn.zero_grad()

    

    output = cnn(X.view(-1,1,28,28))

    

    

    matches = [torch.argmax(i) == torch.argmax(j) for i,j in zip (output,y)]

    num_matches = sum(matches)

    number = len (matches)

    loss = loss_function(output, torch.max(y,1)[1].long())

    if train:

        loss.backward()

        optimizer.step()

        

    return num_matches.item(), loss, number

    

def train (cnn, train_loader, val_loader, EPOCHS):

    

    i=0

    

    iteration = []

    t_acc = []

    t_loss = []

    

    v_acc = []

    v_loss = []

    

    for epoch in tqdm.notebook.trange(EPOCHS):

        for  images, labels in tqdm.notebook.tqdm (train_loader):

            images, labels = images.to(device), labels.to(device)

            

            matches, loss, number = fwd_pass(cnn, images, labels, train=True)

            

            if i % 50 == 0:

                

                print (matches)

                

                t_acc.append ((matches / number) * 100 )

                t_loss.append (loss)

                

                val_accuracy, val_loss, = test(cnn, val_loader)

                cnn.train() #set back to train

                

                v_acc.append(val_accuracy)

                v_loss.append(val_loss)

                

                iteration.append(i)

            



                

#                 if i % 150 == 0:

#                     index = len(t_acc)-1

#                     print ("Iteration: {}, Training accuracy: {}, Training Loss: {}, Val accuracy: {}, Val Loss: {}".format(i, t_acc[index],t_loss[index], v_acc[index], v_loss[index] ))

                

            i = i+1

    return t_acc, t_loss, v_acc, v_loss, iteration

            



        

def test (model, val_loader):

    model.eval() #set to eval mode

    count = 0

    correct = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images, labels = images.to(device), labels.to(device)



            matches, loss, number  = fwd_pass (model, images, labels, train = False)

            count = count + number

            correct = correct  + matches

    

    accuracy = (correct / count) *100

    

    return accuracy, loss

        

        

    

import torch.optim as optim



cnn = CNN()



cnn = cnn.float() #change model parameters to float



#if gpu available



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



cnn.to(device)



EPOCHS = 15



learning_rate = 1e-5

loss_function = nn.CrossEntropyLoss()



optimizer = optim.Adam(cnn.parameters(), lr = learning_rate)



t_accuracy, t_loss, v_accuracy, v_loss, iteration = train (cnn, train_loader, val_loader, EPOCHS)











plt.plot ( t_accuracy, 'bo', label = 'training accuracy')



plt.plot ( v_accuracy, 'b', label = 'validation accuracy')

plt.title ('training and val accuracy')

plt.legend()

plt.figure

plt.show()



plt.plot ( t_loss, 'bo', label = 'training loss')



plt.plot ( v_loss, 'b', label = 'validation loss')

plt.title ('training and val accuracy')

plt.legend()

plt.figure



plt.show()

torch.FloatTensor(10).uniform_(0, 120).long()