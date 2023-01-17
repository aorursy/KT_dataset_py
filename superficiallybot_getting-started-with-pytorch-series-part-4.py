# do the necessary imports

import torch

import numpy as np

import torch.nn as nn

import torchvision.transforms as transforms

import torchvision.datasets as dsets

from torch.autograd import Variable

from torch.utils.data import Dataset

from torch.utils.data import DataLoader

import torch.nn.functional as F



import glob

import os

from PIL import Image

%matplotlib inline

import matplotlib.pyplot as plt

image_size = (100,100)

image_row_size = image_size[0] * image_size[1]
class CatDogDataset(Dataset):

    def __init__(self, path, transform = None):

        self.classes = os.listdir(path)

        self.path = [f'{path}/{classname}' for classname in self.classes]

        

        self.file_list = [glob.glob(f'{x}/*') for x in self.path]

        self.transform = transform

        

        files = []

        for i, classname in enumerate(self.classes):

            for filename in self.file_list[i]:

                files.append([i, classname, filename])

        self.file_list = files

        files = None

        

    

    def __len__(self):

        return len(self.file_list)

    

    def __getitem__(self, idx):

        filename = self.file_list[idx][2]

        classCategory = self.file_list[idx][0]

        

        im = Image.open(filename)

        if self.transform:

            im = self.transform(im)

        return im.view(-1), classCategory

        
mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]



transform = transforms.Compose([

    transforms.Resize(image_size),

    transforms.ToTensor(),

    transforms.Normalize(mean, std)

])
train = CatDogDataset('../input/dogs-cats-images/dataset/training_set/', transform)
test = CatDogDataset('../input/dogs-cats-images/dataset/test_set/', transform)
# Visualize the Image from the dataset



def imshow(source):

    plt.figure(figsize=(10,10))

    imt = (source.view(-1, image_size[0], image_size[0]))

    imt = imt.numpy().transpose([1,2,0])

    imt = (std * imt + mean).clip(0,1) #clip to remove noise # de-normalize by multiplying with std and mean

    plt.subplot(1,2,2)

    plt.imshow(imt)

shuffle = True

batch_size = 64

num_workers = 0

dataloader = DataLoader(dataset = train,

                       shuffle = shuffle,

                       batch_size = batch_size,

                       num_workers = num_workers)
shuffle_test = False

batch_size = 64

num_workers = 0

testloader = DataLoader(dataset = test,

                       shuffle = shuffle_test,

                       batch_size = batch_size,

                       num_workers = 0)
class MyModel(nn.Module):

    def __init__(self, in_feature):

        super(MyModel, self).__init__()

        self.fc1 = nn.Linear(in_features = in_feature, out_features= 1024)

        self.fc2 = nn.Linear(in_features = 1024, out_features = 512)

        self.fc3 = nn.Linear(in_features = 512, out_features = 256)

        self.fc4 = nn.Linear(in_features = 256, out_features = 128)

        self.fc5 = nn.Linear(in_features = 128, out_features = 64)

        self.fc6 = nn.Linear(in_features = 64, out_features = 32)

        self.fc7= nn.Linear(in_features = 32, out_features = 2)

        

        

        

    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        x = F.relu(self.fc4(x))

        x = F.relu(self.fc5(x))

        x = F.relu(self.fc6(x))

        x = self.fc7(x)

        x = nn.LogSoftmax(dim = 1)(x)

        return x
model = MyModel(image_row_size*3)

print(model)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
device
model.to(device)
# Hyper-Parameters



criterion = nn.NLLLoss()

optimizer = torch.optim.Adam(model.parameters(), lr =0.003)



epochs = 10

# train the model

traininglosses = []

testinglosses = []

testaccuracy = []

totalsteps = []

epochs = 10

steps = 0

running_loss = 0

print_every = 5





for epoch in range(epochs):  # epochs loop

    for ii, (inputs, labels) in enumerate(dataloader):  # batch loop within an epoch

        

        # increment the step count

        steps += 1

        

        #Move the input and label tensors to the GPU

        inputs, labels = inputs.to(device), labels.to(device)  # shifting the inputs & labels to GPU if available

        

        optimizer.zero_grad()    # always clear the optimizer from previous step

        

        outputs = model.forward(inputs)   # obtain outputs for the inputs

        loss = criterion(outputs, labels)   # calculate loss

        loss.backward()                       # calculate gradients

        optimizer.step()                       # take optimizer step to adjust the weights

        

        running_loss += loss.item()

        

        if steps % print_every == 0:           # validation/test score

            test_loss = 0

            accuracy = 0

            model.eval() # IMPORTANT -> SET YOUR MODEL TO EVAL MODE WHEN EVALUATING THE MODEL TO ENSURE THE GRADIENTS ARE NOT CHANGED

            with torch.no_grad():

                for inputs, labels in testloader:

                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs_val = model.forward(inputs)

                    

                    loss_val = criterion(outputs_val, labels)

                    test_loss += loss_val.item()

                    

                    # Calculate accuracy

                    ps = torch.exp(outputs_val)

                    top_p, top_class = ps.topk(1, dim = 1)

                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    

            traininglosses.append(running_loss/print_every)  # overall loss for the training averaged by the number of steps to be taken into consideration

            testinglosses.append(test_loss/len(testloader))

            testaccuracy.append(accuracy/ len(testloader))

            

            totalsteps.append(steps)

            

            print(f'Device {device}..'

                 f"Epoch {epoch+1}/{epochs}.. "

                  f"Step {steps}.. "

                  f"Train loss: {running_loss/print_every:.3f}.. "

                  f"Test loss: {test_loss/len(testloader):.3f}.. "

                  f"Test accuracy: {accuracy/len(testloader):.3f}")

            

            running_loss = 0

            model.train()  #switch back to train mode

                    

            
import matplotlib.pyplot as plt



plt.plot(totalsteps, traininglosses, label = 'Train Loss')

plt.plot(totalsteps, testinglosses, label = 'Test Loss')

plt.plot(totalsteps, testaccuracy, label = 'Test Accuracy')

plt.legend()

plt.grid()

plt.show()
checkpoint = {

    'parameters' : model.parameters, 

    'state_dict' : model.state_dict()

}
torch.save(checkpoint, './catvdog.pth')