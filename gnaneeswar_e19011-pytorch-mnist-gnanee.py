import os

from pathlib import Path

import torch

from torch.utils.data import TensorDataset ,DataLoader

from torch import nn,optim

import torch.nn.functional as F

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

PATH=Path("../input")

print(os.listdir("../input"))
train=pd.read_csv(PATH/'train.csv')

test=pd.read_csv(PATH/'test.csv')

train.shape,test.shape
# Vizuvalizing the data set



import matplotlib.pyplot as plt

import numpy

def imshow(image, ax=None, title=None, normalize=True):



    if ax is None:

        fig, ax = plt.subplots()

        

    if normalize:

        mean = np.array([0.485, 0.456, 0.406])

        std = np.array([0.229, 0.224, 0.225])

        image = std * image + mean

        image = np.clip(image, 0, 1)



    ax.imshow(image)

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.tick_params(axis='both', length=0)

    ax.set_xticklabels('')

    ax.set_yticklabels('')



    return ax



plt.imshow(numpy.array(train.iloc[10,1:]).reshape(28,28))

print(train.iloc[10,0])

x=train.drop("label",axis=1)

y=np.array(train['label'])

x.shape,y.shape
torch.from_numpy(x.values[1]).shape
torch_X_train = torch.from_numpy(x.values).type(torch.FloatTensor)/255

torch_y_train = torch.from_numpy(y).type(torch.LongTensor)

myDataset = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)

valid_no  = int(0.2 * len(myDataset))



# so divide the data into trainset (80 %) and testset (20 %)

trainSet,testSet = torch.utils.data.random_split(myDataset,(len(myDataset)-valid_no,valid_no))

print(f"len of trainSet {len(trainSet)} , len of testSet {len(testSet)}")

batch_size=80

train_loader  = DataLoader(trainSet , batch_size=batch_size ,shuffle=True) 

test_loader  = DataLoader(testSet , batch_size=batch_size ,shuffle=True)



print(f"len of train_loader {len(train_loader)} , len of test_loader {len(test_loader)}")

class Network(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(784, 392)

        self.fc2 = nn.Linear(392, 196)

        self.fc3 = nn.Linear(196, 64)

        self.fc4 = nn.Linear(64, 32)

        self.fc5 = nn.Linear(32,10)





        # Dropout module with 0.2 drop probability

        self.dropout = nn.Dropout(p=0.2)



    def forward(self, x):

        # make sure input tensor is flattened

        x = x.view(x.shape[0], -1)



        # Now with dropout

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        x = self.dropout(F.relu(self.fc3(x)))

        x = self.dropout(F.relu(self.fc4(x)))



        # output so no dropout here

        x = F.log_softmax(self.fc5(x), dim=1)



        return x

        

model=Network()

#optimizer=optim.Adam(model.parameters(),lr=0.01)

#optimizer=optim.SGD(model.parameters(), lr=0.04, weight_decay= 1e-7, momentum = 0.9,nesterov = True)

#optimizer=torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

optimizer=torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)





criterion=nn.CrossEntropyLoss()
epochs=15

train_losses,test_losses=[],[]

for e in range(epochs):

    running_loss=0

    for images,labels in train_loader:

        optimizer.zero_grad()

        log_ps=model(images)

        loss=criterion(log_ps,labels)

        loss.backward()

        optimizer.step()

        running_loss+=loss.item()

        

    else:

        test_loss=0

        accuracy=0

        

        with torch.no_grad():

            model.eval()

            for images,labels in test_loader:

                log_ps=model(images)

                test_loss+=criterion(log_ps,labels)

                ps=torch.exp(log_ps)

                top_p,top_class=ps.topk(1,dim=1)

                equals=top_class==labels.view(*top_class.shape)

                accuracy+=torch.mean(equals.type(torch.FloatTensor))

        model.train()

        train_losses.append(running_loss/len(train_loader))

        test_losses.append(test_loss/len(test_loader))



        print("Epoch: {}/{}.. ".format(e+1, epochs),

              "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),

              "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),

              "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))    
print("Our model: \n\n", model, '\n')

print("The state dict keys: \n\n", model.state_dict().keys())
torch.save(model.state_dict(), 'checkpoint.pth')
state_dict = torch.load('checkpoint.pth')

print(state_dict.keys())
model.load_state_dict(state_dict)
checkpoint = {'input_size': 784,

              'output_size': 10,

              'hidden_layers': [256,128,64],

              'state_dict': model.state_dict()}



torch.save(checkpoint, 'checkpoint.pth')
test_images = pd.read_csv("../input/test.csv")

test_image = test_images.loc[:,test_images.columns != "label"].values

test_dataset = torch.from_numpy(test_image).type(torch.FloatTensor)/255

print(test_dataset.shape)

#test_dataset = torch.utils.data.TensorDataset(test_dataset)

new_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 100, shuffle = False)
results = []

with torch.no_grad():

    model.eval()

    for images in new_test_loader:

        output = model(images)

        ps = torch.exp(output)

        top_p, top_class = ps.topk(1, dim = 1)

        results += top_class.numpy().tolist()
predictions = np.array(results).flatten()

print(predictions[:5])

print(predictions.shape)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("my_submissions.csv", index=False, header=True)