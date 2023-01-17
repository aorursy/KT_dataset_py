import torch

import torch.nn.functional as F

from torch import nn, optim

from torchvision import transforms



import pandas as pd

import numpy as np
dataset = pd.read_csv('../input/train.csv')
# Percentage of data mounting for training that will be used for training tests.

PERCENTAGE_OF_TESTS = 0.1



length_tests = int(len(dataset) * PERCENTAGE_OF_TESTS)

testdata = dataset[:length_tests]

traindata = dataset[length_tests:]



# Verifies that all training data is being used.

testdata.shape[0] + traindata.shape[0] == len(dataset)
len(traindata)
# Dataset responsible for manipulating data for training as well as training tests.

class DatasetMNIST(torch.utils.data.Dataset):

    def __init__(self, data, transform=None):

        self.data = data

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        item = self.data.iloc[index]

                

        image = item[1:].values.astype(np.uint8).reshape((28, 28, 1))

        label = item[0]

        

        if self.transform is not None:

            image = self.transform(image)

            

        return image, label
BATCH_SIZE = 100



def new_trainloader(random_affine=True):

    train_transform = None



    if random_affine == False:

        train_transform = transforms.Compose([

            transforms.ToTensor(),

            transforms.Normalize(mean=(0.5,), std=(0.5,))

        ])

    else:

        train_transform = transforms.Compose([

            transforms.ToPILImage(),

            transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),

            transforms.ToTensor(),

            transforms.Normalize(mean=(0.5,), std=(0.5,))

        ])

    trainset = DatasetMNIST(traindata, transform=train_transform)

    

    return torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
test_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])



testset = DatasetMNIST(testdata, transform=test_transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)



trainloader = new_trainloader(random_affine=False)
class Network(nn.Module):

    def __init__(self):

        super().__init__()

        

        self.fc1 = nn.Linear(784, 512)

        self.fc2 = nn.Linear(512, 256)

        self.fc3 = nn.Linear(256, 64)

        self.fc4 = nn.Linear(64, 10)

        

        

        self.dropout = nn.Dropout(p=0.1680)

        

    def forward(self, x):

        # Make sure input tensor is flattened.

        x = x.view(x.shape[0], -1)

        

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        x = self.dropout(F.relu(self.fc3(x)))



        x = F.log_softmax(self.fc4(x), dim=1)

        

        return x
# Configuring and Creating the Network for Training.

LEARNING_RATE = 0.0001680



model = Network()

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)



epochs = 500
train_losses, test_losses = [], []



for e in range(epochs):

    running_loss = 0

    if (e == 3 or e % 5 == 0) and e != 5:

        trainloader = new_trainloader()



    for images, labels in trainloader:

        # Clear the gradients, do this because gradients are accumulated.

        optimizer.zero_grad()

        

        # Forward pass, get our log-probabilities.

        log_ps = model(images)



        # Calculate the loss with the logps and the labels.

        loss = criterion(log_ps, labels)

        

        # Turning loss back.

        loss.backward()

        

        # Take an update step and few the new weights.

        optimizer.step()

        

        running_loss += loss.item()

    else:

        test_loss = 0

        accuracy = 0

        

        # Turn off gradients for validation, saves memory and computations.

        with torch.no_grad():

            model.eval() # change the network to evaluation mode

            for images, labels in testloader:

                # Forward pass, get our log-probabilities.

                log_ps = model(images)

                

                # Calculating probabilities for each class.

                ps = torch.exp(log_ps)

                

                # Capturing the class more likely.

                top_p, top_class = ps.topk(1, dim=1)

                

                # Verifying the prediction with the labels provided.

                equals = top_class == labels.view(*top_class.shape)

                

                test_loss += criterion(log_ps, labels)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

                

        model.train() # change the network to training mode

        

        train_losses.append(running_loss/len(trainloader))

        test_losses.append(test_loss/len(testloader))



        if e == 0 or e == 10 or e % 50 == 0:

            print(f"Epoch: {e+1}/{epochs}.. ",

                  f"Training Loss: {running_loss/len(trainloader):.3f}.. ",

                  f"Test Loss: {test_loss/len(testloader):.3f}.. ",

                  f"Test Accuracy: {accuracy/len(testloader):.3f}")
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt



plt.plot(train_losses, label='Training loss')

plt.plot(test_losses, label='Validation loss')

plt.legend(frameon=False)
fig, axis = plt.subplots(3, 8, figsize=(20, 10))

images, labels = next(iter(testloader))



for i, ax in enumerate(axis.flat):

    with torch.no_grad():

        model.eval()

        image = images[i]



        log_ps = model(image)

        ps = torch.exp(log_ps)

        top_p, top_class = ps.topk(1, dim=1)

        predicted = top_class



        ax.imshow(image.view(28, 28), cmap='binary') # add image

        ax.set(title = "Predicted Digit: {}".format(predicted.item())) # add label
class DatasetSubmissionMNIST(torch.utils.data.Dataset):

    def __init__(self, file_path, transform=None):

        self.data = pd.read_csv(file_path)

        self.transform = transform

        

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        image = self.data.iloc[index].values.astype(np.uint8).reshape((1, 784))



        

        if self.transform is not None:

            image = self.transform(image)

            

        return image
submissionset = DatasetSubmissionMNIST('../input/test.csv', transform=test_transform)

submissionloader = torch.utils.data.DataLoader(submissionset, batch_size=BATCH_SIZE, shuffle=False)
submission = [['ImageId', 'Label']]



with torch.no_grad():

    model.eval()

    image_id = 1



    for images in submissionloader:

        log_ps = model(images)

        ps = torch.exp(log_ps)

        top_p, top_class = ps.topk(1, dim=1)

        

        for prediction in top_class:

            submission.append([image_id, prediction.item()])

            image_id += 1

            

print(len(submission) - 1)
import csv



with open('submission.csv', 'w') as submissionFile:

    writer = csv.writer(submissionFile)

    writer.writerows(submission)

    

print('Submission Complete!')