import torch

from torchvision import datasets, transforms

from torch.autograd import Variable



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]



train_transform = transforms.Compose([

                                transforms.Resize(256),

                                transforms.RandomResizedCrop(224),

                                transforms.RandomHorizontalFlip(),

                                transforms.ToTensor(),

                                transforms.Normalize(mean, std)])



test_transform = transforms.Compose([

                                transforms.Resize(256),

                                transforms.CenterCrop(224),

                                transforms.ToTensor(),

                                transforms.Normalize(mean, std)])
data_dir = "../input/flowers_/flowers_/"

img_datasets ={}
# That's how easily you can for images folders in Pytorch for further operations

img_datasets['train']= datasets.ImageFolder(data_dir + '/train', train_transform)

img_datasets['test']= datasets.ImageFolder(data_dir + '/test', test_transform)
# these gets extracted from the folder name

class_names = img_datasets['train'].classes

class_names
# these gets extracted from the folder name - class label mapping

class_idx = img_datasets['train'].class_to_idx

class_idx
train_loader = torch.utils.data.DataLoader(img_datasets['train'],

                                                   batch_size=10,

                                                   shuffle=True,

                                                   num_workers=4)



test_loader = torch.utils.data.DataLoader(img_datasets['test'],

                                                   batch_size=10,

                                                   shuffle=True,

                                                   num_workers=4)
images , labels = next(iter(train_loader))

images.shape
# lets look at the labels

labels
import torchvision.models as models



model = models.vgg16(pretrained=True)
for param in model.parameters():

    param.required_grad = False
# Now let's check the model archietecture

model
num_of_inputs = model.classifier[0].in_features

num_of_inputs
# restructaring the classifier

import torch.nn as nn

model.classifier = nn.Sequential(

                      nn.Linear(num_of_inputs, 5),

                        nn.LogSoftmax(dim=1))
# Now let's check the model archietecture again to see the changes 

model
# check if CUDA is available

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')

# move tensors to GPU if CUDA is available

if train_on_gpu:

    model.cuda()
# loss function and optimizer

criterion = nn.NLLLoss()

optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
# number of epochs to train the model

n_epochs = 10





for epoch in range(n_epochs):

    # monitor training loss

    train_loss = 0.0

    train_accuracy = 0

    

    ###################

    # train the model #

    ###################

    model.train() # prep model for training

    for data, target in train_loader:

        if train_on_gpu:

            data, target = Variable(data.cuda()), Variable(target.cuda())

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the loss

        loss = criterion(output, target)

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update running training loss

        train_loss += loss.item()*data.size(0)

        #calculate accuracy

        ps = torch.exp(output)

        top_p, top_class = ps.topk(1, dim=1)

        equals = top_class == target.view(*top_class.shape)

        train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    

# calculate average loss over an epoch

    train_loss = train_loss/len(train_loader.dataset)



    print('Epoch: {} \tTraining Loss: {:.6f}'.format(

            epoch+1, 

            train_loss

            ))

    print(f"Train accuracy: {train_accuracy/len(train_loader):.3f}")

# Checking Test Performence

test_accuracy = 0

model.eval() # prep model for evaluation

for data, target in test_loader:

    if train_on_gpu:

        data, target = Variable(data.cuda()), Variable(target.cuda())

    # forward pass: compute predicted outputs by passing inputs to the model

    output = model(data)

    # calculate the loss

    loss = criterion(output, target)

    #calculate accuracy

    ps = torch.exp(output)

    top_p, top_class = ps.topk(1, dim=1)

    equals = top_class == target.view(*top_class.shape)

    test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()



print(f"Test accuracy: {test_accuracy/len(test_loader):.3f}")