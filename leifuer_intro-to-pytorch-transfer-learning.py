%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt



import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models
# define helper.py 

import matplotlib.pyplot as plt

import numpy as np

from torch import nn, optim

from torch.autograd import Variable





def test_network(net, trainloader):



    criterion = nn.MSELoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)



    dataiter = iter(trainloader)

    images, labels = dataiter.next()



    # Create Variables for the inputs and targets

    inputs = Variable(images)

    targets = Variable(images)



    # Clear the gradients from all Variables

    optimizer.zero_grad()



    # Forward pass, then backward pass, then update weights

    output = net.forward(inputs)

    loss = criterion(output, targets)

    loss.backward()

    optimizer.step()



    return True





def imshow(image, ax=None, title=None, normalize=True):

    """Imshow for Tensor."""

    if ax is None:

        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))



    if normalize:

        mean = np.array([0.485, 0.456, 0.406])

        std = np.array([0.229, 0.224, 0.225])

        image = std * image + mean

        image = np.clip(image, 0, 1)



    ax.imshow(image)

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.tick_params(axis='both', length=0)

    ax.set_xticklabels('')

    ax.set_yticklabels('')



    return ax





def view_recon(img, recon):

    ''' Function for displaying an image (as a PyTorch Tensor) and its

        reconstruction also a PyTorch Tensor

    '''



    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)

    axes[0].imshow(img.numpy().squeeze())

    axes[1].imshow(recon.data.numpy().squeeze())

    for ax in axes:

        ax.axis('off')

        ax.set_adjustable('box-forced')



def view_classify(img, ps, version="MNIST"):

    ''' Function for viewing an image and it's predicted classes.

    '''

    ps = ps.data.numpy().squeeze()



    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())

    ax1.axis('off')

    ax2.barh(np.arange(10), ps)

    ax2.set_aspect(0.1)

    ax2.set_yticks(np.arange(10))

    if version == "MNIST":

        ax2.set_yticklabels(np.arange(10))

    elif version == "Fashion":

        ax2.set_yticklabels(['T-shirt/top',

                            'Trouser',

                            'Pullover',

                            'Dress',

                            'Coat',

                            'Sandal',

                            'Shirt',

                            'Sneaker',

                            'Bag',

                            'Ankle Boot'], size='small');

    ax2.set_title('Class Probability')

    ax2.set_xlim(0, 1.1)



    plt.tight_layout()
# Define default PATH

PATH = '../input/dogs-vs-cats-for-pytorch/cat_dog_data/Cat_Dog_data'
data_dir = PATH



# TODO: Define transforms for the training data and testing data

train_transforms = transforms.Compose([transforms.Resize(255),

                                      transforms.CenterCrop(224),

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
## TODO: Use a pretrained model to classify the cat and dog images



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = model.to(device)



criterion = nn.NLLLoss()



optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

                       

epoches = 1

steps = 0



train_losses, test_losses = [], []



for e in range(epoches):

    running_loss = 0

    

    for images, labels in trainloader:

        steps += 1

        images, labels = images.to(device), labels.to(device)

        

        optimizer.zero_grad()

        

        logits = model(images)

        

        loss = criterion(logits, labels)

        

        loss.backward()

        

        optimizer.step()

        

        running_loss += loss.item()

        

        train_losses.append(running_loss)

        

        if steps % 5 == 0:

            test_loss, accuracy = 0, 0

        

            with torch.no_grad():

                model.eval()



                for images, labels, in testloader:

                    images, labels = images.to(device), labels.to(device)



                    logits = model(images)



                    test_loss += criterion(logits, labels)



                    ps = torch.exp(logits)



                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)



                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()





            print(f"Epoch: {e+1}/{epoches};"

                  f"Train_loss: {running_loss};"

                  f"Test_loss: {test_loss/len(testloader)};"

                  f"Accuracy: {accuracy/len(testloader)}")

            model.train()

            running_loss = 0


# # Use GPU if it's available

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# model = models.densenet121(pretrained=True)



# # Freeze parameters so we don't backprop through them

# for param in model.parameters():

#     param.requires_grad = False

    

# model.classifier = nn.Sequential(nn.Linear(1024, 256),

#                                  nn.ReLU(),

#                                  nn.Dropout(0.2),

#                                  nn.Linear(256, 2),

#                                  nn.LogSoftmax(dim=1))



# criterion = nn.NLLLoss()



# # Only train the classifier parameters, feature parameters are frozen

# optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)



# model.to(device);
# epochs = 1

# steps = 0

# running_loss = 0

# print_every = 5

# for epoch in range(epochs):

#     for inputs, labels in trainloader:

#         steps += 1

#         # Move input and label tensors to the default device

#         inputs, labels = inputs.to(device), labels.to(device)

        

#         optimizer.zero_grad()

        

#         logps = model.forward(inputs)

#         loss = criterion(logps, labels)

#         loss.backward()

#         optimizer.step()



#         running_loss += loss.item()

        

#         if steps % print_every == 0:

#             test_loss = 0

#             accuracy = 0

#             model.eval()

#             with torch.no_grad():

#                 for inputs, labels in testloader:

#                     inputs, labels = inputs.to(device), labels.to(device)

#                     logps = model.forward(inputs)

#                     batch_loss = criterion(logps, labels)

                    

#                     test_loss += batch_loss.item()

                    

#                     # Calculate accuracy

#                     ps = torch.exp(logps)

#                     top_p, top_class = ps.topk(1, dim=1)

#                     equals = top_class == labels.view(*top_class.shape)

#                     accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    

#             print(f"Epoch {epoch+1}/{epochs}.. "

#                   f"Train loss: {running_loss/print_every:.3f}.. "

#                   f"Test loss: {test_loss/len(testloader):.3f}.. "

#                   f"Test accuracy: {accuracy/len(testloader):.3f}")

#             running_loss = 0

#             model.train()