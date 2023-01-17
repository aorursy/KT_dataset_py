import pandas as pd

import torch

import numpy as np

from matplotlib import pyplot as plt

import torchvision.transforms as transforms

from torch import nn,functional

from sklearn.model_selection import train_test_split

from torch.autograd import Variable

%matplotlib inline
train_batch_size=127

test_batch_size=127



in_data=pd.read_csv("../input/digit-recognizer/train.csv")
x=in_data.loc[:,in_data.columns!='label'].values/255

y=in_data.label.values



feature_train,feature_test, target_train, target_test=train_test_split(x,y,test_size=0.2, random_state=42)

x_train=torch.from_numpy(feature_train)

x_test=torch.from_numpy(feature_test)



y_train=torch.from_numpy(target_train).type(torch.LongTensor)

y_test=torch.from_numpy(target_test).type(torch.LongTensor)



train=torch.utils.data.TensorDataset(x_train,y_train)

test=torch.utils.data.TensorDataset(x_test,y_test)
train_loader=torch.utils.data.DataLoader(train,batch_size=train_batch_size,shuffle=False)

test_loader=torch.utils.data.DataLoader(test,batch_size=test_batch_size,shuffle=False)
#CNN



class Net(nn.Module):

    def __init__(self):

        super(Net,self).__init__()

        self.cnn_1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride=1, padding=0)

        self.cnn_2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride=1, padding=0)

        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(2,2)

        self.dropout = nn.Dropout(p=0.2)

        self.dropout2d = nn.Dropout2d(p=0.2)

        

        self.fc1 = nn.Linear(32 * 4 * 4, 128) 

        self.fc2 = nn.Linear(128, 64) 

        self.out = nn.Linear(64, 10) 

        

    def forward(self,x):

        

        out = self.cnn_1(x)

        out = self.relu(out)

        out = self.dropout2d(out)

        out = self.maxpool(out)

        

        out = self.cnn_2(out)

        out = self.relu(out)

        out = self.dropout2d(out)

        out = self.maxpool(out)

        

        out = out.view(out.size(0), -1)

        out = self.fc1(out)

        out = self.dropout(out)

        out = self.fc2(out)

        out = self.dropout(out)

        out = self.out(out)

        

        return out
model=Net()

model=model.double()

criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters(),lr=0.003)
epochs=15

train_losses, test_losses = [] ,[]

for epoch in range(epochs):

    running_loss = 0

    for images,labels in train_loader:

        train = Variable(images.view(-1,1,28,28))

        labels = Variable(labels)

        

        optimizer.zero_grad()

        

        output = model(train)

        loss = criterion(output,labels)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

    else:

        test_loss = 0

        accuracy = 0

        

        with torch.no_grad(): #Turning off gradients to speed up

            model.eval()

            for images,labels in test_loader:

                

                test = Variable(images.view(-1,1,28,28))

                labels = Variable(labels)

                

                log_ps = model(test)

                test_loss += criterion(log_ps,labels)

                

                ps = torch.exp(log_ps)

                top_p, top_class = ps.topk(1, dim = 1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

        model.train()        

        train_losses.append(running_loss/len(train_loader))

        test_losses.append(test_loss/len(test_loader))



        print("Epoch: {}/{}.. ".format(epoch+1, epochs),

              "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),

              "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),

              "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))
plt.plot(train_losses, label='training loss')

plt.plot(test_losses, label='test loss')

plt.legend(frameon=False)
test_images=pd.read_csv("../input/digit-recognizer/test.csv")

test_image=test_images.loc[:,test_images.columns!='label'].values/255



test_dataset=torch.from_numpy(test_image)
tester=torch.utils.data.DataLoader(test_dataset,batch_size=100,shuffle=False)



results = []

with torch.no_grad():

    model.eval()

    for images in tester:

        test = Variable(images.view(-1,1,28,28))

        output = model(test)

        ps = torch.exp(output)

        top_p, top_class = ps.topk(1, dim = 1)

        results += top_class.numpy().tolist()
predictions=np.array(results).flatten()

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("my_submissions.csv", index=False, header=True)
