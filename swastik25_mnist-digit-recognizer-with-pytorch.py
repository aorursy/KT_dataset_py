# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
from pathlib import Path
import torch
from torch.utils.data import TensorDataset ,DataLoader
from torch import nn,optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
PATH=Path("../input/")
print(os.listdir("../input/"))
train=pd.read_csv(PATH/'train.csv')
test=pd.read_csv(PATH/'test.csv')
train.shape,test.shape
x=train.drop("label",axis=1)
y=np.array(train['label'])
torch_X_train = torch.from_numpy(x.values).type(torch.FloatTensor)/255
torch_y_train = torch.from_numpy(y).type(torch.LongTensor)
myDataset = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
valid_no  = int(0.2 * len(myDataset))
# so divide the data into trainset and testset
trainSet,testSet = torch.utils.data.random_split(myDataset,(len(myDataset)-valid_no,valid_no))
print(f"len of trainSet {len(trainSet)} , len of testSet {len(testSet)}")
batch_size=64
train_loader  = DataLoader(trainSet , batch_size=batch_size ,shuffle=True) 
test_loader  = DataLoader(testSet , batch_size=batch_size ,shuffle=True)
from torchvision import datasets, transforms
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_1 = nn.Linear(784, 256)
        self.hidden_2 = nn.Linear(256, 128)
        self.hidden_3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Now with dropout
        x = self.dropout(F.relu(self.hidden_1(x)))
        x = self.dropout(F.relu(self.hidden_2(x)))
        x = self.dropout(F.relu(self.hidden_3(x)))

        # output so no dropout here
        x = F.log_softmax(self.output(x), dim=1)

        return x
        
model_N1=Network()
#optimizer=optim.Adam(model_N1.parameters(),lr=0.001)
criterion=nn.NLLLoss()
def nn_train(epochs, optimizer, model):
    
    train_losses,test_losses=[],[]
    print("No. of Parameters =",sum([p.numel() for p in model.parameters()]))
    print("\n")
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

            print("Epoch: {}/{} ||".format(e+1, epochs),
                  "Training Loss: {:.3f} ||".format(running_loss/len(train_loader)),
                  "Test Loss: {:.3f} ||".format(test_loss/len(test_loader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))  
            
    plt.plot(np.arange(epochs),train_losses,label='Train loss')
    plt.plot(np.arange(epochs),test_losses,label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend()
    plt.show()
optimizer=optim.SGD(model_N1.parameters(),lr=0.01)
nn_train(10, optimizer, model_N1)
optimizer=optim.SGD(model_N1.parameters(),lr=0.05)
nn_train(10, optimizer, model_N1)
optimizer=optim.SGD(model_N1.parameters(),momentum=0.9,nesterov=True,lr=0.05)
nn_train(10, optimizer, model_N1)
optimizer=optim.SGD(model_N1.parameters(),momentum=0.8,nesterov=True,lr=0.1)
nn_train(10, optimizer, model_N1)
optimizer=optim.Adam(model_N1.parameters(),lr=0.01)
nn_train(10, optimizer, model_N1)
optimizer=optim.Adam(model_N1.parameters(),lr=0.005)
nn_train(10, optimizer, model_N1)
optimizer=optim.Adam(model_N1.parameters(),lr=0.001)
nn_train(10, optimizer, model_N1)
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_1 = nn.Linear(784, 128)
        self.hidden_2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Now with dropout
        x = self.dropout(F.relu(self.hidden_1(x)))
        x = self.dropout(F.relu(self.hidden_2(x)))

        # output so no dropout here
        x = F.log_softmax(self.output(x), dim=1)

        return x
    
model_N2=Network()
criterion=nn.NLLLoss()
optimizer=optim.SGD(model_N2.parameters(),lr=0.01)
nn_train(10, optimizer, model_N2)
optimizer=optim.SGD(model_N2.parameters(),lr=0.05)
nn_train(10, optimizer, model_N2)
optimizer=optim.SGD(model_N2.parameters(),momentum=0.9,nesterov=True,lr=0.05)
nn_train(10, optimizer, model_N2)
optimizer=optim.SGD(model_N2.parameters(),momentum=0.9,nesterov=True,lr=0.1)
nn_train(10, optimizer, model_N2)
optimizer=optim.Adam(model_N2.parameters(),lr=0.01)
nn_train(10, optimizer, model_N2)
optimizer=optim.Adam(model_N2.parameters(),lr=0.005)
nn_train(10, optimizer, model_N2)
optimizer=optim.Adam(model_N2.parameters(),lr=0.001)
nn_train(10, optimizer, model_N2)
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_1 = nn.Linear(784, 64)
        self.hidden_2 = nn.Linear(64, 36)
        self.output = nn.Linear(36, 10)

        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        # Now with dropout
        x = self.dropout(F.relu(self.hidden_1(x)))
        x = self.dropout(F.relu(self.hidden_2(x)))

        # output so no dropout here
        x = F.log_softmax(self.output(x), dim=1)

        return x
    
model_N3=Network()
criterion=nn.NLLLoss()
optimizer=optim.SGD(model_N3.parameters(),lr=0.01)
nn_train(10, optimizer, model_N3)
optimizer=optim.SGD(model_N3.parameters(),lr=0.05)
nn_train(10, optimizer, model_N3)
optimizer=optim.SGD(model_N3.parameters(),momentum=0.9,nesterov=True,lr=0.05)
nn_train(10, optimizer, model_N3)
optimizer=optim.SGD(model_N3.parameters(),momentum=0.9,nesterov=True,lr=0.1)
nn_train(10, optimizer, model_N3)
optimizer=optim.Adam(model_N3.parameters(),lr=0.01)
nn_train(10, optimizer, model_N3)
optimizer=optim.Adam(model_N3.parameters(),lr=0.005)
nn_train(10, optimizer, model_N3)
optimizer=optim.Adam(model_N3.parameters(),lr=0.001)
nn_train(10, optimizer, model_N3)
print("Our model: \n\n", model_N3, '\n')
print("The state dict keys: \n\n", model_N3.state_dict().keys())
torch.save(model_N3.state_dict(), 'optimal_NN_model_checkpoint.pth')
state_dict = torch.load('optimal_NN_model_checkpoint.pth')
print(state_dict.keys())

print(state_dict)
checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [64,36],
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
    model_N3.eval()
    for images in new_test_loader:
        output = model_N3(images)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim = 1)
        results += top_class.numpy().tolist()
predictions = np.array(results).flatten()
print(predictions[:5])
print(predictions.shape)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("final_submissions.csv", index=False, header=True)