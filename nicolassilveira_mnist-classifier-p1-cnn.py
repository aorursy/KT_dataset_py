import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import torch

import torch.optim as optim

import torch.nn as nn

import torch.nn.functional as F



import torchvision

import torchvision.transforms as transforms



import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

%matplotlib inline
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 12

batch_Size = 100

learning_Rate = 0.02

weight_Decay = 0.001

model_Dropout = 0.35
def read_file(fileName):

    

    df = pd.read_csv('../input/digit-recognizer/' + fileName)

    

    if(fileName == 'train.csv'):

        

        #Train data

        label = np.array(df['label'])

        data = np.array(df[df.columns[1:]], dtype = np.float)

        new_data = np.reshape(a = data, newshape = (data.shape[0],28,28))

        

        return new_data, label

    

    elif(fileName == 'test.csv'):

        

        #Test data

        data = np.array(df,dtype = np.float)

        new_data = np.reshape(a = data, newshape = (data.shape[0],28,28))

        

        return new_data
train_data, train_label = read_file('train.csv')

test_data = read_file('test.csv')
#Normalizing

train_data = train_data / 255 #Range[0,255] to [0,1]

test_data = test_data / 255



#Standarizing

train_data = (train_data - 0.5) / 0.5

test_data = (test_data - 0.5) / 0.5



#Numpy to tensor

train_data = torch.from_numpy(train_data)

test_data = torch.from_numpy(test_data)



train_data, test_data = train_data.type(torch.FloatTensor), test_data.type(torch.FloatTensor)



#Add column dim

train_data = train_data.unsqueeze_(dim=1)

test_data = test_data.unsqueeze_(dim=1)



#Train dataset (data + labels)

train_dataset = torch.utils.data.TensorDataset(train_data, torch.from_numpy(train_label))



#Split the datasets

train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [35000, 7000]) #50/10/40



#DataLoaders

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_Size, shuffle = True)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_Size, shuffle = True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size = 28000, shuffle = False) #batch_size = batch_Size,



#classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']



print ('Train set: ' + str(len(train_dataset)))

print ('Validation set: ' + str(len(valid_dataset)))

print ('Test set: ' + str(len(test_data)))
from random import randrange



temp = train_loader.dataset[randrange(10)][0].numpy()

temp = np.reshape(a=temp, newshape=(temp.shape[1], temp.shape[2]))

plt.imshow(temp)
class ConvNet(nn.Module):

    

    def __init__(self):

        

        super().__init__()

        

        #Layer sizes

        hidden1 = 512

        hidden2 = 128

        hidden3 = 64

        

        #Convolutions & Pooling

        self.conv1 = nn.Conv2d(1, 32, 3)   #input channel size, output channel size, kernel size

        self.conv2 = nn.Conv2d(32, 64, 3)

        self.pool = nn.MaxPool2d(2, 2)     #size, stride

        

        #Fully connected layers

        self.fc1 = nn.Linear(1600, hidden1)

        self.fc2 = nn.Linear(hidden1, hidden2)

        self.fc3 = nn.Linear(hidden2, hidden3)

        self.fc4 = nn.Linear(hidden3, 10)

        

        self.dropout = nn.Dropout(model_Dropout)

    

    def forward(self, x):

        

        #Feature learning

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        

        #Classification

        x = x.view(x.shape[0], -1)

        

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = F.relu(self.fc2(x))

        x = self.dropout(x)

        x = F.relu(self.fc3(x))

        x = self.dropout(x)

        x = self.fc4(x)

        

        return F.log_softmax(x)
recog_model = ConvNet()#.to(device)



criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(recog_model.parameters(), lr=learning_Rate, weight_decay = weight_Decay)



steps = 0



accuracy_list = []

train_losses, valid_losses = [], []
for e in range (epochs):

    

    running_loss = 0

    

    for images, labels in train_loader:

        

        #GPU support

        #images = images.to(device)

        #labels = labels.to(device)

        

        #Forward

        output = recog_model(images)

        loss = criterion(output, labels)

        

        #Backward

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

    else:

        

        valid_loss = 0

        accuracy = 0

        

        #Evaluate the model

        with torch.no_grad():

            recog_model.eval()

            

            for images, labels in valid_loader:

                

                output = recog_model(images)

                valid_loss += criterion(output, labels)

                

                ps = torch.exp(output)

                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.reshape(*top_class.shape)

                

                accuracy = torch.mean(equals.type(torch.FloatTensor))

                

        print('Accuracy ' + str(e+1) + ': ' + str(accuracy.item() * 100) + '%')

                

        #Store values for plotting

        accuracy_list.append(accuracy.item() * 100)

        train_losses.append(running_loss / len(train_loader))

        valid_losses.append(valid_loss / len(valid_loader))



#Print the losses

print('Final training loss: ' + str(train_losses[epochs-1]))

print('Final validation loss: ' + str(valid_losses[epochs-1]))

        

### Plot ###

xAxis = []

for i in range(epochs):

    xAxis.append(i+1)

    

#Accuracy graph

plt.plot(xAxis, accuracy_list)

plt.ylabel('%')

plt.xlabel('Epoch')

plt.title('Accuracy')

plt.show()



#Loss graph

plt.plot(xAxis, train_losses, 'r--', xAxis, valid_losses, 'g--')

red_line = mpatches.Patch(color='red', label='train loss')

green_line = mpatches.Patch(color='green', label='valid loss')

plt.legend(handles=[red_line, green_line])

plt.xlabel('Epoch')

plt.title('Loss')

plt.show()
#Evaluate the model

with torch.no_grad():

    recog_model.eval()

            

    for images in test_loader:

              

        output = recog_model(images)

        ps = torch.exp(output)

        top_p, top_class = ps.topk(1, dim=1)



print('Done')        
#results = recog_model.predict(test_loader)

results = np.argmax(output, axis = 1) #results por output

results = pd.Series(results, name = 'Label')

results
submission = pd.concat([pd.Series(range(1, 28001), name = 'ImageId'), results], axis = 1)

submission.to_csv('submission.csv', index = False)