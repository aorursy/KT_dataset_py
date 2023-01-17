%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt



import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models
data_dir = '../input/Cat_Dog_data'



# TODO: Define transforms for the training data and testing data

train_transforms = transforms.Compose([transforms.RandomRotation(30),

                                       transforms.RandomResizedCrop(224),

                                       transforms.RandomHorizontalFlip(),

                                       transforms.ToTensor(),

                                       transforms.Normalize([0.485, 0.456, 0.406],

                                                            [0.229, 0.224, 0.225])])



test_transforms = transforms.Compose([transforms.Resize(255),

                                      transforms.CenterCrop(224),

                                      transforms.ToTensor(),

                                      transforms.Normalize([0.485, 0.456, 0.406],

                                                           [0.229, 0.224, 0.225])])



# Pass transforms in here, then run the next cell to see how the transforms look

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)

test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)



trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

images,labels = next(iter(trainloader))

print(images.shape)
model = models.densenet121(pretrained=True)

model
# Freeze parameters so we don't backprop through them

for param in model.parameters():

    param.requires_grad = False



from collections import OrderedDict

classifier = nn.Sequential(OrderedDict([

                          ('fc1', nn.Linear(1024, 500)),

                          ('relu', nn.ReLU()),

                          ('fc2', nn.Linear(500, 2)),

                          ('output', nn.LogSoftmax(dim=1))

                          ]))

    

model.classifier = classifier
import time
for device in ['cpu', 'cuda']:



    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)



    model.to(device)



    for ii, (inputs, labels) in enumerate(trainloader):



        # Move input and label tensors to the GPU

        inputs, labels = inputs.to(device), labels.to(device)



        start = time.time()



        outputs = model.forward(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()



        if ii==3:

            break

        

    print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")
# Use GPU if it's available

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

model = models.densenet121(pretrained=True)



# Freeze parameters so we don't backprop through them

for param in model.parameters():

    param.requires_grad = False

    

model.classifier = nn.Sequential(nn.Linear(1024, 256),

                                 nn.ReLU(),

                                 nn.Dropout(0.2),

                                 nn.Linear(256, 2),

                                 nn.LogSoftmax(dim=1))



criterion = nn.NLLLoss()



# Only train the classifier parameters, feature parameters are frozen

optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)



model.cuda();
import time

epochs = 1

steps = 0

running_loss = 0

print_every = 5

for epoch in range(epochs):

    for inputs, labels in trainloader:

        steps += 1

        model.cuda();

        # Move input and label tensors to the default device

        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        start = time.time()

        optimizer.zero_grad()

        

        logps = model.forward(inputs)

        loss = criterion(logps, labels)

        loss.backward()

        optimizer.step()



        running_loss += loss.item()

        print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")

        if steps % print_every == 0:

            test_loss = 0

            accuracy = 0

            model.eval()

            with torch.no_grad():

                for inputs, labels in testloader:

                    inputs, labels = inputs.to(device), labels.to(device)

                    logps = model.forward(inputs)

                    batch_loss = criterion(logps, labels)

                    

                    test_loss += batch_loss.item()

                    

                    # Calculate accuracy

                    ps = torch.exp(logps)

                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    

            print(f"Epoch {epoch+1}/{epochs}.. "

                  f"Train loss: {running_loss/print_every:.3f}.. "

                  f"Test loss: {test_loss/len(testloader):.3f}.. "

                  f"Test accuracy: {accuracy/len(testloader):.3f}")

            running_loss = 0

            model.train()
images,labels = next(iter(testloader))

image = images[0]