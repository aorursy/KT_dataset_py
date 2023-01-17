# load in required packages 



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import torch

import torchvision.transforms as transforms

from torch.utils.data import DataLoader, TensorDataset

from torch.utils.data.sampler import SubsetRandomSampler

from torch import nn, optim

import torch.nn.functional as F

import time

from sklearn.model_selection import train_test_split
# print(torch.cuda.is_available())

# print(torch.backends.cudnn.enabled)



# if torch.cuda.is_available():

#     device = torch.device('cuda')

# print(device)
! ls ../input/digit-recognizer/
input_folder_path = "../input/digit-recognizer/"

train = pd.read_csv(input_folder_path+"train.csv")

test = pd.read_csv(input_folder_path+"test.csv")
train.head()
train.dtypes
train.max().sort_values()
train_labels = train['label'].values

train_images = (train.iloc[:,1:].values).astype('float32')

test_images = (test.iloc[:,:].values).astype('float32')



#Training and Validation Split

train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, train_labels,

                                                                     stratify=train_labels, random_state=42,

                                                                     test_size=0.20)

# reshape images 

train_images = train_images.reshape(train_images.shape[0], 28, 28)

valid_images = valid_images.reshape(valid_images.shape[0], 28, 28)

test_images = test_images.reshape(test_images.shape[0], 28, 28)



# convert images to tesnsors and normalize them 

# train

train_images_tensor = torch.tensor(train_images)/255.0

train_labels_tensor = torch.tensor(train_labels)

train_tensor = TensorDataset(train_images_tensor, train_labels_tensor)



# valid

valid_images_tensor = torch.tensor(valid_images)/255.0

valid_labels_tensor = torch.tensor(valid_labels)

valid_tensor = TensorDataset(valid_images_tensor, valid_labels_tensor)



# test

test_images_tensor = torch.tensor(test_images)/255.0



# Load images into the data generator

batch_size = 20

train_loader = DataLoader(train_tensor, batch_size=batch_size, num_workers=2, shuffle=True)

valid_loader = DataLoader(valid_tensor, batch_size=batch_size, num_workers=2, shuffle=True)  

test_loader = DataLoader(test_images_tensor, batch_size=batch_size, num_workers=2, shuffle=False) # make sure shuffle=False to keep data order the same
print("number of image batches:",

      "train dataset: {:2d}.. ".format(len(train_loader) - len(valid_loader)),

      "validation dataset: {:2d}.. ".format(len(valid_loader)),

      "test dataset: {:2d}".format(len(test_loader)))

print("number of image samples:",

      "train dataset: {:2d}.. ".format(len(train_loader.sampler) - len(valid_loader.sampler)),

      "validation dataset: {:2d}.. ".format(len(valid_loader.sampler)),

      "test dataset: {:2d}".format(len(test_loader.sampler)))
# from torch.utils.data import TensorDataset

# class TrainDataset(Dataset): # take important note that Dataset class with capital letter should be passed 

#     def __init__(self, file_path, transform=None):

#         self.data = pd.read_csv(file_path)

#         self.transform = transform

        

#     def __len__(self): # so that len(dataset) returns the size of the dataset that is required for batching and sampling

#         return len(self.data)

    

#     def __getitem__(self, index): # to support the indexing such that dataset[i] can be used to get ith sample

#         # convert dtype to np.uint8 [Unsigned integer (0 to 255)] where 0: black and 255: white

#         # define image numy array as (Height * Width * Channel) as later the shape can be transformed 

#         # to tensor in PyTorch as (H, W, C) --> (C, H, W)

#         images = self.data.iloc[index, 1:].values.astype(np.uint8).reshape((28, 28, 1))

#         labels = self.data.iloc[index, 0]

        

#         if self.transform is not None:

#             images = self.transform(images)

            

#         return images, labels
# batch_size = 20

# train_data = TrainDataset('../input/digit-recognizer/train.csv', transform=None)

# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# # get one batch of images and labels at each iteration

# dataiter = iter(train_loader)

# images, labels = dataiter.next()

# images.shape
# # convert data to torch.FloatTensor

# transform = transforms.ToTensor()



# train_data = TrainDataset('../input/digit-recognizer/train.csv', transform=transform)

# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

# dataiter = iter(train_loader)

# image, label = dataiter.next()

# image.shape
# # percentage of training set to use as validation

# valid_size = 0.2

# # obtain training indices that will be used for validation

# num_train = len(train_data)

# indices = list(range(num_train))

# np.random.shuffle(indices)

# split = int(np.floor(valid_size * num_train))

# train_idx, valid_idx = indices[split:], indices[:split]



# # define samplers for obtaining training and validation batches

# train_sampler = SubsetRandomSampler(train_idx)

# valid_sampler = SubsetRandomSampler(valid_idx)



# # prepare the relevant validation DataLoader

# valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 

#     sampler=valid_sampler)
# obtain one batch of training images and their relevant labels

dataiter = iter(train_loader)

images, labels = dataiter.next()



# plot the images in the batch, along with the corresponding labels

nrows = 2

ncol = batch_size/nrows



fig = plt.figure(figsize=(25, 4))

for idx in np.arange(batch_size):

    ax = fig.add_subplot(nrows, ncol, idx+1, xticks=[], yticks=[])

    ax.imshow(np.squeeze(images[idx]), cmap='gray')

    # print out the correct label for each image

    # .item() gets the value contained in a Tensor

    ax.set_title(str(labels[idx].item()))
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
# class TestDataset(Dataset): # take important note that Dataset class with capital letter should be passed 

#     def __init__(self, file_path, transform=None):

#         self.data = pd.read_csv(file_path)

#         self.transform = transform

        

#     def __len__(self): # so that len(dataset) returns the size of the dataset that is required for batching and sampling

#         return len(self.data)

    

#     def __getitem__(self, index): # to support the indexing such that dataset[i] can be used to get ith sample

#         # convert dtype to np.uint8 [Unsigned integer (0 to 255)] where 0: black and 255: white

#         # define image numy array as (Height * Width * Channel) as later the shape can be transformed 

#         # to tensor in PyTorch as (H, W, C) --> (C, H, W)

#         images = self.data.iloc[index, :].values.astype(np.uint8).reshape((28, 28, 1))

        

#         if self.transform is not None:

#             images = self.transform(images)

            

#         return images
# # convert data to torch.FloatTensor

# transform = transforms.ToTensor()



# test_data = TestDataset('../input/digit-recognizer/test.csv', transform=transform)

# test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

# dataiter = iter(test_loader)

# image = dataiter.next()

# image.shape
# define the NN architecture

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        # number of hidden nodes in each layer (512)

        hidden_1 = 512

        hidden_2 = 512

        # linear layer (784 -> hidden_1)

        self.fc1 = nn.Linear(28 * 28, hidden_1)

        # linear layer (n_hidden -> hidden_2)

        self.fc2 = nn.Linear(hidden_1, hidden_2)

        # linear layer (n_hidden -> 10)

        self.fc3 = nn.Linear(hidden_2, 10)

        # dropout layer (p=0.2)

        # dropout prevents overfitting of data

        self.dropout = nn.Dropout(0.2)



    def forward(self, x):

        # flatten image input

        x = x.view(-1, 28 * 28)

        # add hidden layer, with relu activation function

        x = F.relu(self.fc1(x))

        # add dropout layer

        x = self.dropout(x)

        # add hidden layer, with relu activation function

        x = F.relu(self.fc2(x))

        # add dropout layer

        x = self.dropout(x)

        # add output layer

        x = self.fc3(x)

        return x

    

# initialize the NN

model = Net()

print(model)
# specify loss function

criterion = nn.CrossEntropyLoss(reduction='mean') # the sum of the output will be divided 

                                                  # by the number of elements in the output



# specify optimizer

lr = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
trained_weights_fc1 = (abs(model.fc1.weight - 0) > 0.001) * 1

trained_weights_fc2 = (abs(model.fc2.weight - 0) > 0.001) * 1

trained_weights_fc3 = (abs(model.fc3.weight - 0) > 0.001) * 1



print("fraction of non-zero weights for connected layer:",

      "fc1: {:.3f}.. ".format(torch.mean(trained_weights_fc1.type(torch.FloatTensor))),

      "fc2: {:.3f}.. ".format(torch.mean(trained_weights_fc2.type(torch.FloatTensor))),

      "fc3: {:.3f}".format(torch.mean(trained_weights_fc3.type(torch.FloatTensor))))
# initialize lists to monitor validation accuracy

class_correct = list(0. for i in range(10))

class_total = list(0. for i in range(10))



model.eval() 

with torch.no_grad(): # turn off gradient calculation to save memory and time  

    for data, target in valid_loader:

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
valid_loader = DataLoader(valid_tensor, batch_size=batch_size, num_workers=2, shuffle=False) # make sure shuffle=False to keep data order the same 

valid_preds = torch.LongTensor()

model.eval() 

with torch.no_grad(): 

    for data, target in valid_loader:

        output = model(data)

        _, pred = torch.max(output, dim=1) 

        # concatenate pred tensors along row  

        valid_preds = torch.cat((valid_preds, pred), dim=0)



valid_part_df = pd.DataFrame({"true label":valid_labels,

                              "predicted label":valid_preds.numpy(),

                              "correct": (abs(valid_labels-valid_preds.numpy()==0)*1)})



accuracy = 100. * valid_part_df["correct"].mean()

accuracy
valid_part_df
test_preds = torch.LongTensor()

model.eval() 

with torch.no_grad(): 

    for data in test_loader:

        output = model(data)

        _, pred = torch.max(output, dim=1) 

        # concatenate pred tensors along row  

        test_preds = torch.cat((test_preds, pred), dim=0)



submission = pd.DataFrame({"ImageId":list(range(1, len(test_preds)+1)),

                           "Label":test_preds.numpy()})

# view the dataframe

submission
# write dataframe to csv file

submission.to_csv("my_submission.csv", index=False, header=True)
submission = pd.read_csv("my_submission.csv")

submission