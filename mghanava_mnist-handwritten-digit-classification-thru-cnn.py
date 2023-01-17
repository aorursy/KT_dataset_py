import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import torch

import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

# from torch.utils.data import TensorDataset 

# from sklearn.model_selection import train_test_split

from torch import nn, optim

import torch.nn.functional as F

import time

import torchvision

from torch.utils.data.sampler import SubsetRandomSampler
! ls ../input/digit-recognizer/
# Make an instance of Dataset and customize to get input train dataframe thru pandas

class TrainDataset(Dataset): # take important note that Dataset class with capital letter should be passed 

    def __init__(self, file_path, transform=None):

        self.data = pd.read_csv(file_path)

        self.transform = transform

        

    def __len__(self): # so that len(dataset) returns the size of the dataset that is required for batching and sampling

        return len(self.data)

    

    def __getitem__(self, index): # to support the indexing such that dataset[i] can be used to get ith sample

        # convert dtype to np.uint8 [Unsigned integer (0 to 255)] where 0: black and 255: white

        # define image numy array as (Height * Width * Channel) as later the shape can be transformed 

        # to tensor in PyTorch as (H, W, C) --> (C, H, W)

        images = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))

        labels = self.data.iloc[index, 0]

        

        if self.transform is not None:

            images = self.transform(images)

            

        return images, labels



transform = transforms.ToTensor()



train_data = TrainDataset('../input/digit-recognizer/train.csv', transform=transform)



# split train set into training and validation data

## percentage of training set to use as validation

valid_size = 0.15



# obtain training indices that will be used for validation

num_train = len(train_data)

indices = list(range(num_train))

torch.manual_seed(0)

np.random.shuffle(indices)

split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]



# define samplers for obtaining training and validation batches

train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)



batch_size = 20



# prepare the relevant training DataLoader

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)



# prepare the relevant validation DataLoader

valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)



# Make an instance of Dataset and customize to get input test dataframe thru pandas

class TestDataset(Dataset): 

    def __init__(self, file_path, transform=None):

        self.data = pd.read_csv(file_path)

        self.transform = transform

        

    def __len__(self): 

        return len(self.data)

    

    def __getitem__(self, index): 

        images = self.data.iloc[index, :].values.astype(np.uint8).reshape((28, 28, 1))

        

        if self.transform is not None:

            images = self.transform(images)

            

        return images

    



test_data = TestDataset('../input/digit-recognizer/test.csv', transform=transform)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
# batch_size = 20



# # augment data



# transform = transforms.ToTensor()



# augment = transforms.Compose([

#     transforms.ToPILImage(),

#     transforms.RandomAffine(degrees=(-45, 45)),

#     transforms.ToTensor()

# ])





# train_data = TrainDataset('../input/digit-recognizer/train.csv', transform=transform)

# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)



# train_data_aug = TrainDataset('../input/digit-recognizer/train.csv', transform=augment)

# train_loader_aug = torch.utils.data.DataLoader(train_data_aug, batch_size=batch_size, shuffle=False)



# def imshow(img, title=''):

#     """Plot the image batch.

#     """

#     plt.figure(figsize=(10, 15))

#     plt.title(title)

#     plt.imshow(np.transpose(img.numpy(), (1, 2, 0)), cmap='gray')

#     plt.show()

    

# for i, data in enumerate(train_loader):

#     x, y = data  

#     imshow(torchvision.utils.make_grid(x, 10), title='Normal')

#     break # just view one batch

    

# # view augmented data

# for i, data in enumerate(train_loader_aug):

#     x, y = data  

#     imshow(torchvision.utils.make_grid(x, 10), title='Augmented Data')

#     break # just view one batch
# view tensor shapes of images and labels in all three DataLoaders

# print("shape of train images: ", next(iter(train_loader))[0].shape,

#       "\nshape of label images: ", next(iter(train_loader))[1].shape,

#       "\nshape of valid images: ", next(iter(valid_loader))[0].shape,

#       "\nshape of valid labels: ", next(iter(valid_loader))[1].shape, 

#       "\nshape of test images : ", next(iter(test_loader)).shape)
# obtain one batch of training images and their relevant labels

dataiter = iter(train_loader)

images, labels = dataiter.next()



images = images.numpy() # convert tensor image to array so that we can squeeze it 

img = np.squeeze(images[0]) # remove dimesion 1 from (1*28*28) shape to display 28*28 pixels    



fig = plt.figure(figsize = (12,12)) 

ax = fig.add_subplot(111)

ax.imshow(img, cmap='gray')

width, height = img.shape

thresh = img.max()/2.5

for x in range(width):

    for y in range(height):

        val = round(img[x][y],2) if img[x][y] !=0 else 0

        ax.annotate(str(val), xy=(y,x),

                    horizontalalignment='center',

                    verticalalignment='center',

                    color='white' if img[x][y]<thresh else 'black')
# define convolutional neural network architecture 

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        # convolutional layer (sees 1x28x28 image tensor)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        # convolutional layer (sees 16x14x14 image tensor)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # max pooling layer

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1st linear layer (64 * 7 * 7 -> hidden_1)

        hidden_1 = 1024

        hidden_2 = 512

        self.fc1 = nn.Linear(128*7*7, hidden_1)

        # 2nd linear layer (hidden_1 -> hidden_2)

        self.fc2 = nn.Linear(hidden_1, hidden_2)

      # 3rd linear layer (hidden_2 -> 10)

        self.fc3 = nn.Linear(hidden_2, 10)

        # add dropout layer

        self.dropout = nn.Dropout(p=0.5)

    

    def forward(self, x):

        # add sequence of convolutional and max pooling layers

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        # flatten convolutional layers image output as the fully connected layers input

        x = x.view(-1, 128*7*7)

        # add dropout layer

        x = self.dropout(x)

        # add 1st hidden layer, with relu activation function

        x = F.relu(self.fc1(x))

        # add dropout layer

        x = self.dropout(x)

        # add 2nd hidden layer, with relu activation function

        x = F.relu(self.fc2(x))

        # add dropout layer

        x = self.dropout(x)

        # add 2nd hidden layer, with relu activation function

        x = self.fc3(x)

        return x
# move the model to GPU, if available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# create a complete CNN

model = Net()



# specify loss function (categorical cross-entropy)

criterion = nn.CrossEntropyLoss()



# specify optimizer

lr = 0.001

optimizer = optim.Adam(model.parameters(), lr = lr)



model.to(device)
# initialize time to measure the elapsed time

t0 = time.time()

# number of epochs to train the model

n_epochs = 50



# initialize tracker for minimum validation loss

valid_loss_min = np.Inf # set initial "min" to infinity



for epoch in range(n_epochs):

    # monitor training loss

    train_loss = 0.0    

    ###################

    # train the model #

    ###################

    model.train() # prep model for training by turning on the drop out layers to avoid overfitting

    for data, target in train_loader:

        # Move data and target tensors to the default device

        data, target = data.to(device),target.to(device)

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        # forward pass: compute predicted outputs (10 digit class probabilities) by passing inputs 

        # (vectors of the batch size (data.size(0))) to the model

        output = model(data)

        # calculate the loss

        loss = criterion(output, target)

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update running training loss

        train_loss += loss.item()*data.size(0) # multiply mean loss of each batch by its size

        

    ######################    

    # validate the model #

    ######################

    # monitor validation loss

    valid_loss = 0.0

    model.eval() # prep model for evaluation

    with torch.no_grad(): # turn off gradient calculation to save memory and computation time

        for data, target in valid_loader:

            # Move data and target tensors to the default device

            data, target = data.to(device),target.to(device)

            # forward pass: compute predicted outputs by passing inputs to the model

            output = model(data)

            # calculate the loss

            loss = criterion(output, target)

            # update running validation loss 

            valid_loss += loss.item()*data.size(0) # multiply mean loss of each batch by its size

        

    # print training/validation statistics 

    # calculate average loss over an epoch

    train_loss = train_loss/len(train_loader.dataset)

    valid_loss = valid_loss/len(valid_loader.dataset)

    

    print("Epoch: {}/{}.. ".format(epoch+1, n_epochs),

          "Training Loss: {:.6f}.. ".format(train_loss),

          "Validation Loss: {:.6f}.. ".format(valid_loss))

    

    # save model if validation loss has decreased

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,

        valid_loss))

        torch.save(model.state_dict(), 'model.pt')

        valid_loss_min = valid_loss



print('Training and validation executed in {:.1f} minutes.'.format((time.time() - t0)/60))
model.load_state_dict(torch.load('model.pt'))
# initialize lists to monitor validation accuracy

class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))



model.eval() 

with torch.no_grad(): # turn off gradient calculation to save memory and time  

    for data, target in valid_loader:

        # Move data and target tensors to the default device

        data, target = data.to(device),target.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        _, pred = torch.max(output, dim=1) # _ represents probabilities of each predicted digit that is unimportant

        # compare predictions to true label

        correct = pred == target.view_as(pred)

        for i in range(len(target)):

            label = target.data[i]

            class_correct[label] += correct[i].item()

            class_total[label] += 1

# calculate and print avg validation loss

for i in range(10):

    if class_total[i] > 0:

        print('Validation Accuracy of %5s: %.1f%% (%2d/%2d)' % (

            str(i), 100 * class_correct[i] / class_total[i],

            np.sum(class_correct[i]), np.sum(class_total[i])))

    else:

        print('Validation Accuracy of %5s: N/A (no training examples)' % (classes[i]))



print('\nValidation Accuracy (Overall): %.1f%% (%2d/%2d)' % (

    100. * np.sum(class_correct) / np.sum(class_total),

    np.sum(class_correct), np.sum(class_total)))
test_preds = torch.LongTensor()

model.eval() 

with torch.no_grad(): 

    for data in test_loader:

        # Move data and target tensors to the default device

        data, target = data.to(device),target.to(device)

        output = model(data)

        _, pred = torch.max(output, dim=1) 

        # concatenate pred tensors along row  

        test_preds = torch.cat((test_preds.cpu(), pred.cpu()), dim=0)



submission = pd.DataFrame({"ImageId":list(range(1, len(test_preds)+1)),

                           "Label":test_preds.numpy()})

# view the dataframe

submission
# write dataframe to csv file

submission.to_csv("my_submission.csv", index=False, header=True)
submission = pd.read_csv("my_submission.csv")

submission