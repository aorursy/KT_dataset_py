#PyTorch Specific libraries

import torch

from torch import nn, optim

import torch.nn.functional as F

from torch.autograd import Variable



#Data manipulation and visualisation specific libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# For splitting the data into Train and Test set

from sklearn.model_selection import train_test_split



# This piece of code is required to make use of the GPU instead of CPU for faster processing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)



#If it prints "cuda:0" that means it has access to GPU. If it prints out "cpu", then it's still running on CPU.
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



#Let's check if they have been loaded properly

print('train.shape:\n', train.shape)

print('test.shape:\n', test.shape)
X = train.iloc[:,:-1]

y = train.iloc[:,-1:] #Could have done like this 

y = train.label.values # but needed to convert it to np.ndarray for torch tensor conversion
print('X.shape: ', X.shape, 'X.type: ', type(X) )

print('y.shape: ', y.shape, 'y.type: ', type(y) )
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.1, random_state = 1)



print('X_train.shape: ', X_train.shape)

print('y_train.shape: ', y_train.shape)

print('X_test.shape: ', X_test.shape)

print('y_test.shape: ', y_test.shape)
#Rescaling values

X_train = X_train.values/255

X_test = X_test.values/255
#Converting to Tensors

X_train = torch.from_numpy(X_train)

X_test = torch.from_numpy(X_test)



y_train = torch.from_numpy(y_train).type(torch.LongTensor)

y_test = torch.from_numpy(y_test).type(torch.LongTensor)



print('X_train.dtype:', X_train.dtype)

print('X_test.dtype:', X_test.dtype)

print('y_train.dtype:', y_train.dtype)

print('y_test.dtype:', y_test.dtype)
train = torch.utils.data.TensorDataset(X_train, y_train)

test = torch.utils.data.TensorDataset(X_test, y_test)



batch = 100



# Set our data loaders

train_loader = torch.utils.data.DataLoader(train, batch_size = batch, shuffle = True)

test_loader = torch.utils.data.DataLoader(test, batch_size = batch, shuffle = True)
class Net(nn.Module):

    

    def __init__(self):

        super().__init__()

        

        self.conv1 = nn.Conv2d(1, 128, 5)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.drop1 = nn.Dropout(p=0.3)

        

        self.conv2 = nn.Conv2d(128, 224, 5)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.drop2 = nn.Dropout(p=0.4)

        

        self.fc3 = nn.Linear(224*4*4, 64)

        self.drop3 = nn.Dropout(p=0.4)

        

        self.fc4 = nn.Linear(64, 32)

        self.drop4 = nn.Dropout(p=0.4)

        

        self.fc5 = nn.Linear(32, 10)

        self.softmax = nn.Softmax(dim=1)

   

    

    def forward(self, x):

        x = self.drop1(self.pool1(F.relu(self.conv1(x))))

        x = self.drop2(self.pool2(F.relu(self.conv2(x))))

        

        x = x.view(-1,224*4*4)

        

        x = self.drop3(F.relu(self.fc3(x)))

        x = self.drop4(F.relu(self.fc4(x)))

        

        x = self.softmax(self.fc5(x))

        

        return x



print(Net()) 
#Making an object of the Net class

model = Net().to(device)



#Loss function

criterion = nn.CrossEntropyLoss ()
# Optimizer

optimizer = optim.Adam(model.parameters(), lr = 0.0015)
# Initialising variables

epochs = 30

steps = 0

print_every = 100

trainLoss = [] 

testLoss = []
for e in range(epochs):

    running_loss = 0

    for images, labels in train_loader:

        steps += 1   # Forward pass

        

        images = (images.view(-1,1,28,28)).type(torch.DoubleTensor)

        optimizer.zero_grad()

        log_ps = model(images.type(torch.FloatTensor).to(device))

        labels = labels.to(device)

        loss = criterion(log_ps, labels)

        loss.backward()   # Backward pass

        optimizer.step()

        

        running_loss += loss.item()

        if steps % print_every == 0:

            test_loss = 0

            accuracy = 0



            with torch.no_grad():

                model.eval()

                for images, labels in test_loader:

                    images = (images.view(-1,1,28,28)).type(torch.DoubleTensor)

                    log_ps = model(images.type(torch.FloatTensor).to(device))

                    labels = labels.to(device)

                    test_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)

                    

                    top_p, top_class = ps.topk(1, dim = 1)

                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor))



            model.train()



            trainLoss.append(running_loss/len(train_loader))

            testLoss.append(test_loss/len(test_loader))



            print("Epoch: {}/{}.. ".format(e + 1, epochs),

                  "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))
%matplotlib inline



plt.plot(trainLoss, label = 'Training Loss')

plt.plot(testLoss, label = 'Validation Loss')

plt.legend(frameon = False)
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



finalTest = test.values/255



finalTest = torch.from_numpy(finalTest)



temp = np.zeros(finalTest.shape)

temp = torch.from_numpy(temp)



data = torch.utils.data.TensorDataset(finalTest, temp)



submissionLoader = torch.utils.data.DataLoader(data, batch_size = batch, shuffle = False)



submission = [['ImageId', 'Label']]



with torch.no_grad():

    model.eval()

    image_id = 1

    for images, _ in submissionLoader:

        images = (images.view(-1,1,28,28)).type(torch.DoubleTensor)

        log_ps = model(images.type(torch.FloatTensor).to(device))

        ps = torch.exp(log_ps)

        top_p, top_class = ps.topk(1, dim = 1)

        

        for prediction in top_class:

            submission.append([image_id, prediction.item()])

            image_id += 1

            





pytorchSubmission = pd.DataFrame(submission)

pytorchSubmission.columns = pytorchSubmission.iloc[0]

pytorchSubmission = pytorchSubmission.drop(0, axis = 0)



pytorchSubmission.to_csv("submission.csv", index = False)
