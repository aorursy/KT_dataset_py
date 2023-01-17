%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt



import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models

import numpy as np
data_dir = '../input/flower_data/flower_data/'



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



train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)

test_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)



train_loader = torch.utils.data.DataLoader(train_data, batch_size=40, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=40)
model = models.densenet161(pretrained=True)
for param in model.parameters():

    param.requires_grad = False



from collections import OrderedDict

classifier = nn.Sequential(OrderedDict([

    ('fc1', nn.Linear(2208, 512)),

    ('relu', nn.ReLU()),

    ('dropout', nn.Dropout(0.4)),

    ('fc2', nn.Linear(512, 102)),

    ('output', nn.LogSoftmax(dim=1))

]))

    

model.classifier = classifier

model.classifier
import json



with open('../input/cat_to_name.json', 'r') as f:

    cat_to_name = json.load(f)
class_names = train_data.classes
for i in range(0,len(class_names)):

    class_names[i] = cat_to_name.get(class_names[i])

class_names[20]
class_names[20]
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

epochs = 25



valid_loss_min = np.Inf



epoch_list = []

train_loss_list = []

val_loss_list = []

train_acc_list = []

val_acc_list = []



for epoch in range(epochs):

    train_loss = 0.0

    val_loss = 0.0

    model.train()       

    correct = 0

    total = 0



    for data, target in train_loader:

        

        if train_on_gpu:

            data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        

        if type(output) == tuple:

            output, _ = output

        predicted = torch.max(output.data, 1)[1]        

        total += len(target)

        correct += (predicted == target).sum()

        

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()*data.size(0)

        train_loss = train_loss/len(train_loader.dataset)

    

    accuracy = 100 * correct / float(total)

    train_acc_list.append(accuracy)

    train_loss_list.append(train_loss)

    

    model.eval() 

    correct = 0

    total = 0

    with torch.no_grad():

        for data, target in test_loader:



            if train_on_gpu:

                data, target = data.cuda(), target.cuda()



            output = model(data)

            if type(output) == tuple:

                output, _ = output



            # Calculate Loss

            loss = criterion(output, target)

            val_loss += loss.item()*data.size(0)

            # Get predictions from the maximum value

            predicted = torch.max(output.data, 1)[1]



            # Total number of labels

            total += len(target)



            # Total correct predictions

            correct += (predicted == target).sum()

    

    # calculate average training loss and accuracy over an epoch

    val_loss = val_loss/len(test_loader.dataset)

    accuracy = 100 * correct/ float(total)

    

    # Put them in their list

    val_acc_list.append(accuracy)

    val_loss_list.append(val_loss)

    

    # Print the Epoch and Training Loss Details with Validation Accuracy   

    print('Epoch: {} \tTraining Loss: {:.4f}\t Val. acc: {:.2f}%'.format(

        epoch+1, 

        train_loss,

        accuracy

        ))

    # save model if validation loss has decreased

    if val_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,

        val_loss))

        # Save Model State on Checkpoint

        torch.save(model.state_dict(), 'blossom.pt')

        valid_loss_min = val_loss

#         scheduler.step(val_loss)

    # Move to next epoch

    epoch_list.append(epoch + 1)
model.load_state_dict(torch.load('blossom.pt'))
# Training / Validation Loss

plt.plot(epoch_list,train_loss_list)

plt.plot(val_loss_list)

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.title("Training/Validation Loss vs Number of Epochs")

plt.legend(['Train', 'Valid'], loc='upper right')

plt.show()
# Train/Valid Accuracy

plt.plot(epoch_list,train_acc_list)

plt.plot(val_acc_list)

plt.xlabel("Epochs")

plt.ylabel("Training/Validation Accuracy")

plt.title("Accuracy vs Number of Epochs")

plt.legend(['Train', 'Valid'], loc='best')

plt.show()
val_acc = sum(val_acc_list[:]).item()/len(val_acc_list)

print("Validation Accuracy of model = {} %".format(val_acc))
for param in model.parameters():

    param.requires_grad = True
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

epochs = 10



valid_loss_min = np.Inf

epoch_list = []

train_loss_list = []

val_loss_list = []

train_acc_list = []

val_acc_list = []

# Start epochs

for epoch in range(epochs):

    train_loss = 0.0

    val_loss = 0.0

    model.train()    

    correct = 0

    total = 0

    for data, target in train_loader:

        if train_on_gpu:

            data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        if type(output) == tuple:

            output, _ = output

        predicted = torch.max(output.data, 1)[1]        

        # Total number of labels

        total += len(target)

        # Total correct predictions

        correct += (predicted == target).sum()

        loss = criterion(output, target)

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update running training loss

        train_loss += loss.item()*data.size(0)

    

    # calculate average training loss over an epoch

    train_loss = train_loss/len(train_loader.dataset)

    

    # Avg Accuracy

    accuracy = 100 * correct / float(total)

    

    # Put them in their list

    train_acc_list.append(accuracy)

    train_loss_list.append(train_loss)

    model.eval() 

    correct = 0

    total = 0

    with torch.no_grad():

        for data, target in test_loader:

            if train_on_gpu:

                data, target = data.cuda(), target.cuda()

            output = model(data)

            if type(output) == tuple:

                output, _ = output

            loss = criterion(output, target)

            val_loss += loss.item()*data.size(0)

            # Get predictions from the maximum value

            predicted = torch.max(output.data, 1)[1]



            # Total number of labels

            total += len(target)



            # Total correct predictions

            correct += (predicted == target).sum()

    

    val_loss = val_loss/len(test_loader.dataset)

    accuracy = 100 * correct/ float(total)

    val_acc_list.append(accuracy)

    val_loss_list.append(val_loss)  

    print('Epoch: {} \tTraining Loss: {:.4f}\t Val. acc: {:.2f}%'.format(

        epoch+1, 

        train_loss,

        accuracy

        ))

    # save model if validation loss has decreased

    if val_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,

        val_loss))

        # Save Model State on Checkpoint

        torch.save(model.state_dict(), 'blossom.pt')

        valid_loss_min = val_loss

#         scheduler.step(val_loss)

    # Move to next epoch

    epoch_list.append(epoch + 1)
def comp_accuracy(model, dataloader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)

    model.to(device)

    model.eval()

    

    with torch.no_grad():

        running_acc = 0.0

        for ii, (images, labels) in enumerate(dataloader, start=1):

            if ii % 5 == 0:

                print('Batch {}/{}'.format(ii, len(dataloader)))

            images, labels = images.to(device), labels.to(device)

            logps = model(images)

            ps = torch.exp(logps)  # in my case the outputs are logits so I take the exp()

            equals = ps.topk(1)[1].view(labels.shape) == labels          

            running_acc += equals.sum().item()

        acc = running_acc/len(dataloader.dataset) 

        print(f'\nAccuracy: {acc:.5f}') 

        

    return acc



test_dir = '../input/test set/'
test_folder = datasets.ImageFolder(test_dir, transform=test_transforms)

tests = torch.utils.data.DataLoader(test_data, batch_size=64)
comp_accuracy(model, tests)