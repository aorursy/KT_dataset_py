from torchvision import datasets,transforms

from torch.utils.data import TensorDataset, DataLoader,Dataset

from torch import nn,optim

import torch.nn.functional as F

import torchvision.transforms.functional as FF

import torch 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
class NeuralNetwork(nn.Module):

    def __init__(self):

        # Super function. It inherits from nn.Module and we can access everythink in nn.Module

        super().__init__()

        

        self.conv1 = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True), # inplace=True helps to save some memory

            

            nn.Conv2d(32, 32, kernel_size=3),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            

            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=14), # compute same padding by hand

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            

            nn.MaxPool2d(2, 2),

            nn.Dropout2d(0.25)

        )

        

        self.conv2 = nn.Sequential(

            nn.Conv2d(32, 64, kernel_size=3),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True), # inplace=True helps to save some memory

            

            nn.Conv2d(64, 64, kernel_size=3),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            

            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=6), # compute same padding by hand

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            

            nn.MaxPool2d(2, 2),

            nn.Dropout2d(0.25)

        )

        

        self.conv3 = nn.Sequential(

            nn.Conv2d(64, 128, kernel_size=4),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.Dropout2d(0.25)

        )

        

        self.fc = nn.Sequential(

            nn.Linear(128*1*1, 10)

        )

        

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = x.view(-1, 128*1*1)

        x = self.fc(x)

        

        return x
class MNIST_data(Dataset):

    """MNIST dtaa set"""

    

    def __init__(self, file_path, 

                 transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), 

                     transforms.Normalize(mean=(0.5,), std=(0.5,))])

                ):

        

        df = pd.read_csv(file_path)

        

        if len(df.columns) == 784:

            # test data

            self.X = df.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]

            self.y = None

        else:

            # training data

            self.X = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]

            self.y = torch.from_numpy(df.iloc[:,0].values)

            

        self.transform = transform

    

    def __len__(self):

        return len(self.X)



    def __getitem__(self, idx):

        if self.y is not None:

            return self.transform(self.X[idx]), self.y[idx]

        else:

            return self.transform(self.X[idx])
!ls "../input/"
batch_size = 64



train_dataset = MNIST_data('../input/digit-recognizer/train.csv', transform= transforms.Compose(

                            [transforms.ToPILImage(),transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))

test_dataset = MNIST_data('../input/digit-recognizer/test.csv')



train_size = int(0.85 * len(train_dataset))

valid_size=int(0.15*len(train_dataset))



train_dataset,valid_dataset = torch.utils.data.random_split(train_dataset, [train_size,valid_size])



train_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                           batch_size=batch_size, shuffle=True)



valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,

                                           batch_size=batch_size, shuffle=True)





test_loader = torch.utils.data.DataLoader(dataset=test_dataset,

                                           batch_size=batch_size, shuffle=False)
dftrain=pd.read_csv("../input/digit-recognizer/train.csv")

dftest=pd.read_csv("../input/digit-recognizer/test.csv")

submission=pd.read_csv("../input/digit-recognizer/sample_submission.csv")
dftrain.head()
train=np.array(dftrain.iloc[:,1:])

labels=dftrain['label']
fig = plt.figure()

for i in range(3):

    plt.subplot(2,3,i+1)

    plt.tight_layout()

    plt.imshow(train[i].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Ground Truth: {}".format(labels[i]))

    plt.xticks([])

    plt.yticks([])

fig
model=NeuralNetwork().cuda()

print(model)
optimizer=optim.Adam(model.parameters(),lr=0.0003)

crit=nn.CrossEntropyLoss()
#Train Data



epochs=30

step=0

train_losses,test_losses,accu=[],[],[]

for i in range(epochs):

    running_loss=0

    for images,labels in train_loader:

        images,labels=images.cuda(),labels.cuda()

        optimizer.zero_grad()

        logps=model(images)

        loss=crit(logps,labels)

        loss.backward()

        optimizer.step()

        running_loss+=loss.item()

    else:

        test_loss=0

        accuracy=0

        

        #Turn off gradient for validation ,saves momery and computation

        

        with torch.no_grad():

            model.eval()

            for images,labels in valid_loader:

                images,labels=images.cuda(),labels.cuda()

                logps=model(images)

                test_loss+=crit(logps,labels)

                ps=torch.exp(logps)

                topp,topc=ps.topk(1,dim=1)

                equal=topc==labels.view(*topc.shape)

                accuracy+=torch.mean(equal.type(torch.FloatTensor))

        model.train()

        

        train_losses.append(running_loss/len(train_loader))

        test_losses.append(test_loss/len(valid_loader))

        accu.append(accuracy)

        

        print("Epoch: {}/{}".format(i+1,epochs),

        "Training Loss {}".format(running_loss/len(train_loader)),

        "Testing Loss {}".format(test_loss/len(valid_loader)),

        "Accuracy {}".format(accuracy/len(valid_loader)))
%matplotlib inline

%config InlineBackend.figure_format='retina'

plt.plot(train_losses,label='Traning Loss')

plt.plot(test_losses,label='Validation Loss')

plt.legend(frameon=False)
pred=[]
with torch.no_grad():

    

    model.eval()

    for images in test_loader:

        logps=model(images.cuda())

        ps=torch.exp(logps)

        topp,topc=ps.topk(1,dim=1)

        for i in topc:

            pred.append(i[0].item())

        
submission['Label']=pred
submission.to_csv("submissionD1.csv",index=False)