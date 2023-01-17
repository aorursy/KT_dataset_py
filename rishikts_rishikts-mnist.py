import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import torch

import torch.nn as nn

from torch.utils.data import DataLoader

import torch.optim as optim

from torch.autograd import Variable
train_df = pd.read_csv(r"../input/digit-recognizer/train.csv",dtype = np.float32)

test_df = pd.read_csv(r"../input/digit-recognizer/test.csv",dtype = np.float32)
train_feats=train_df.loc[:,train_df.columns!='label'].values/255

train_label = train_df['label']



Xtrain,Xtest,Ytrain,Ytest=train_test_split(train_feats,train_label,test_size=0.2,random_state=42)
X_train=torch.from_numpy(np.asarray(Xtrain))

Y_train=torch.from_numpy(np.asarray(Ytrain.astype(np.float32))).type(torch.LongTensor)

X_test=torch.from_numpy(np.asarray(Xtest))

Y_test=torch.from_numpy(np.asarray(Ytest.astype(np.float32))).type(torch.LongTensor)



n_iters=10000

batch_size=100

n_epochs=n_iters/(len(X_train)/batch_size)

n_epochs=int(n_epochs)



train = torch.utils.data.TensorDataset(X_train,Y_train)

test = torch.utils.data.TensorDataset(X_test,Y_test)



train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)

test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)
class LogisReg(nn.Module):

    def __init__(self,input_dim,output_dim):

        super(LogisReg,self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self,X):

        Y_pred=self.linear(X)

        return Y_pred

model=LogisReg(784,10)



criterion=nn.CrossEntropyLoss()



optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
count=0

for epoch in range(n_epochs):

    for i,(images,labels) in enumerate(train_loader):

        train=Variable(images.view(-1,784))

        labels=Variable(labels)

        optimizer.zero_grad()

        output=model(train)

        loss = criterion(output,labels)

        loss.backward()

        optimizer.step()

        count+=1

        if count%1000==0:

            #calculate accuracy

            total=0

            correct=0

            for images,labels in test_loader:

                test=Variable(images.view(-1,784))

                outputs=model(test)

                predicted = torch.max(outputs.data, 1)[1]

                total+=len(labels)

                correct+= (predicted==labels).sum()

            accuracy=(torch.div(correct,float(total)))*100

            #Lets print the results

            print('Iteration : {}, Loss : {}, Accuracy : {}'.format(count,loss,accuracy))
class CNNModel(nn.Module):

    def __init__(self):

        super(CNNModel, self).__init__()

        #First Layer

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)

        self.relu1 = nn.ReLU()

        self.mp1 = nn.MaxPool2d(kernel_size=2)

        #Second Layer

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)

        self.relu2 = nn.ReLU()

        self.mp2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(32*4*4, 10) 

    

    def forward(self, x):

        out = self.conv1(x)

        out = self.relu1(out)

        out = self.mp1(out)

        out = self.conv2(out)

        out = self.relu2(out)

        out = self.mp2(out)

        out = out.view(out.size(0),-1)

        out = self.fc1(out)

        return out

#CNN Model

modelCNN = CNNModel()



criterion = nn.CrossEntropyLoss()



optimizerCNN = torch.optim.SGD(modelCNN.parameters(), lr=0.01)

n_itersCNN=10000

batch_size=100

n_epochsCNN = n_itersCNN / (len(X_train) / batch_size)

n_epochsCNN = int(n_epochsCNN)
count=0

for epoch in range(n_epochsCNN):

    for i, (images,labels) in enumerate(train_loader):

        train=Variable(images.view(100,1,28,28))

        labels=Variable(labels)

        optimizerCNN.zero_grad()

        outputCNN=modelCNN(train)

        loss = criterion(outputCNN,labels)

        loss.backward()

        optimizerCNN.step()

        count+=1

        if count%1000==0:

            #calculate accuracy

            total=0

            correct=0

            for images,labels in test_loader:

                test=Variable(images.view(100,1,28,28))

                outputsCNN=modelCNN(test)

                predicted = torch.max(outputsCNN.data, 1)[1]

                total+=len(labels)

                correct+= (predicted==labels).sum()

            accuracy=(torch.div(correct,float(total)))*100

            #Lets print results

            print('Iteration : {}, Loss : {}, Accuracy : {}'.format(count,loss,accuracy))
test_df_numpy=test_df.to_numpy()

test_df_numpy=test_df_numpy/255

test_df_numpy = test_df_numpy.reshape(-1,1,28,28)

preds = modelCNN(torch.from_numpy(test_df_numpy))

preds.relu()

preds = np.argmax(preds.detach(),axis = 1)

results = pd.DataFrame()

results['ImageId'] = np.arange(len(preds)) + 1

results['Label'] = pd.Series(preds.detach())

results.to_csv('final_submission.csv', index = False)