import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from time import time



import torch

from torch import optim

import torch.nn as nn

import torch.nn.functional as F



from sklearn.preprocessing import MinMaxScaler
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



train_set = np.array(train.drop(['label'], axis=1))

train_label = np.array(train['label'])
trainset = []

batch_size = 64

valid_size = 0.2



for i in range(len(train_label)):    

    trainset.append((torch.tensor(train_set[i]), torch.tensor(train_label[i])))



valid_num = int(len(trainset)*0.2)

trainloader = torch.utils.data.DataLoader(trainset[valid_num:], batch_size=64, shuffle=True)

validloader = torch.utils.data.DataLoader(trainset[:valid_num], batch_size=64, shuffle=True)
dataiter = iter(trainloader)

images, labels = dataiter.next()

print(images.shape)

print(labels.shape)
img = images[3].reshape(1,28,28)

plt.imshow(img.numpy().squeeze(), cmap='gray_r')
figure = plt.figure(figsize=(10,10))

num_of_images = 60

for i in range(0, num_of_images):

    plt.subplot(6,10,i+1)

    img = images[i].reshape(1,28,28)

    plt.imshow(img.numpy().squeeze(), cmap='gray_r')
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = nn.Linear(784, 512)

        self.fc2 = nn.Linear(512, 256)

        self.fc3 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(0.2)

        

    def forward(self, input):

        x = F.relu(self.fc1(input))

        x = self.dropout(x)

        x = F.relu(self.fc2(x))

        x = self.dropout(x)

        x = F.log_softmax(self.fc3(x), dim=1)

        return x
model = Net()



criterion = nn.NLLLoss()

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)



n_epochs = 30

time0 = time()

scaler = MinMaxScaler((0,1))





for epoch in range(n_epochs):

    train_loss = 0.0

    valid_loss = 0.0

    

    for images, labels in trainloader:

        optimizer.zero_grad() 

        images_scaling = scaler.fit_transform(images.view(-1,len(images)))

        output = model(torch.tensor(images_scaling).view(len(images),-1).float())

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()*images.size(0)

        

    for images, labels in validloader:

        images_scaling = scaler.fit_transform(images.view(-1, len(images)))

        output = model(torch.tensor(images_scaling).view(len(images), -1).float())

        loss = criterion(output, labels)

        valid_loss += loss.item()*images.size(0)

    

    train_loss = train_loss/len(trainset[valid_num:])

    valid_loss = valid_loss/len(trainset[:valid_num])

    print('Epoch: {} \tTraining loss: {:.6f} \tValidation loss: {:.6f}'.format(

        epoch+1,

        train_loss,

        valid_loss

    ))

    

#     if valid_loss <= valid_loss_min

print("\nTraining Time (in minutes) =",(time()-time0)/60)
images, labels = next(iter(validloader))



img = images[1]

images_scaling = scaler.fit_transform(img.reshape(-1,1)).reshape(1,-1)



with torch.no_grad():

    logps = model(torch.from_numpy(images_scaling).float())



ps = torch.exp(logps)

probab = list(ps.numpy()[0])



print("Predicted Digit =", probab.index(max(probab)))

print("Label Digit =", labels[1])



img = img.reshape(1,28,28)

plt.imshow(img.numpy().squeeze(), cmap='gray_r')
correct_count, all_count = 0, 0

for images,labels in validloader:

    for i in range(len(labels)):

        img = images[i]

        images_scaling = scaler.fit_transform(img.reshape(-1,1)).reshape(1,-1) 

        with torch.no_grad():

            logps = model(torch.from_numpy(images_scaling).float())



        ps = torch.exp(logps)

        probab = list(ps.numpy()[0])

        pred_label = probab.index(max(probab))

        true_label = labels.numpy()[i]

        if(true_label == pred_label):

          correct_count += 1

        all_count += 1



print("Number Of Images Tested =", all_count)

print("\nModel Accuracy =", (correct_count/all_count))
test_set = np.array(test)

testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

pred_test = []

pred_id = []

imageid = 1

for images in testloader:

    for i in range(len(images)):

        img = images[i]

        images_scaling = scaler.fit_transform(img.reshape(-1,1)).reshape(1,-1) 

        with torch.no_grad():

            logps = model(torch.from_numpy(images_scaling).float())



        ps = torch.exp(logps)

        probab = list(ps.numpy()[0])

        pred_label = probab.index(max(probab))

        pred_test.append(pred_label)

        pred_id.append(imageid)

        imageid += 1
my_prediction = pd.DataFrame({'ImageId' : pred_id, 'Label' : pred_test})

my_prediction.to_csv('my_prediction.csv', index=False)