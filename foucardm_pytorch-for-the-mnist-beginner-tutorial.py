import numpy as np

import pandas as pd



import os

print(os.listdir('../input/digit-recognizer'))



import matplotlib.pyplot as plt

plt.style.use("ggplot")



from sklearn.model_selection import train_test_split



# OpenCV Image Library

import cv2 # Image provessing library



# Import PyTorch

import torchvision.transforms as transforms # Useful to make transformation on images easily

from torch.utils.data.sampler import SubsetRandomSampler

import torch

import torch.nn as nn # To build neural network on PyTorch

import torch.nn.functional as F # Useful to get specific layers for the nn

from torch.utils.data import TensorDataset, DataLoader, Dataset # To create torch datasets

import torchvision # Very large package that contains datasets, models etc...

import torch.optim as optim # To get the optimizer of our model
# Data paths

train_path = '../input/digit-recognizer/train.csv'

test_path = '../input/digit-recognizer/test.csv'

sample_sub_path = '../input/digit-recognizer/sample_submission.csv'
df_train = pd.read_csv(train_path, dtype=np.float32)

df_test = pd.read_csv(test_path, dtype=np.float32)

df_sample_sub = pd.read_csv(sample_sub_path)
print(df_train.shape)

print(df_test.shape)

print(df_sample_sub.shape)
y_train_full = df_train['label'].to_numpy() # We need to convert pd.DataFrame to numpy array to then, convert it to PyTorch tensor

X_train_full = df_train.loc[:,df_train.columns != 'label']
X_train_full = X_train_full.values.reshape(-1,1,28,28) # (nbr of samples, height, width, channel) Since these are not colored images, there's only one channel

df_test = df_test.values.reshape(-1,1,28,28)



X_train_full /= 255.0

df_test /= 255.0



# Split into training and test set

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable

X_train_pyt = torch.from_numpy(X_train)

y_train_pyt = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long



# create feature and targets tensor for validation set.

X_val_pyt = torch.from_numpy(X_val)

y_val_pyt = torch.from_numpy(y_val).type(torch.LongTensor) # data type is long



# create feature and targets tensor for the final set that will be used to train the final model.

X_train_full_pyt = torch.from_numpy(X_train_full)

y_train_full_pyt = torch.from_numpy(y_train_full).type(torch.LongTensor) # data type is long



# Data for submission

X_test_pyt = torch.from_numpy(df_test)

#y_test_pyt = torch.from_numpy(y_test).type(torch.LongTensor)
print(X_train_pyt.shape)

print(X_val_pyt.shape)

print(X_train_full_pyt.shape)

print(X_test_pyt.shape)
# Set batch size

batch_size = 64



# Pytorch train and test sets

df_train_pyt = torch.utils.data.TensorDataset(X_train_pyt,y_train_pyt)

df_val_pyt = torch.utils.data.TensorDataset(X_val_pyt,y_val_pyt)

df_train_full_pyt = torch.utils.data.TensorDataset(X_train_full_pyt,y_train_full_pyt)

df_test_pyt = torch.utils.data.TensorDataset(X_test_pyt)



# data loader

train_loader = torch.utils.data.DataLoader(df_train_pyt, batch_size = batch_size, shuffle = True)

valid_loader = torch.utils.data.DataLoader(df_val_pyt, batch_size = batch_size, shuffle = True)

submission_loader = torch.utils.data.DataLoader(df_train_full_pyt, batch_size = batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(df_test_pyt, batch_size = batch_size, shuffle = False)
one_batch_images, one_batch_labels = next(iter(train_loader))



one_image = one_batch_images[0,0,:,:]

one_label = one_batch_labels[0]



plt.imshow(one_image)

plt.title(one_label)

plt.axis('off')

plt.show()
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size=5)

        self.conv2 = nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size=5)

        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(in_features = 320,  out_features = 50)

        self.fc2 = nn.Linear(in_features = 50, out_features = 10)



    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(-1, 320)

        x = F.relu(self.fc1(x))

        x = F.dropout(x, training=self.training)

        x = self.fc2(x)

        return F.log_softmax(x)
# check if CUDA is available

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
# create a complete CNN

model = Net()

print(model)



# Move model to GPU if available

if train_on_gpu: model.cuda()
criterion = nn.NLLLoss()



# Define the optimier

optimizer = optim.Adam(model.parameters(), lr=0.0015)
# number of epochs to train the model

n_epochs = 30



valid_loss_min = np.Inf # track change in validation loss



# keeping track of losses as it happen

train_losses = []

valid_losses = []



for epoch in range(1, n_epochs+1):



    # keep track of training and validation loss

    train_loss = 0.0

    valid_loss = 0.0

    

    ###################

    # train the model #

    ###################

    model.train()

    for data, target in train_loader:

        # move tensors to GPU if CUDA is available

        if train_on_gpu:

            data, target = data.cuda(), target.cuda()

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the batch loss

        loss = criterion(output, target)

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update training loss

        train_loss += loss.item()*data.size(0)

        

    ######################    

    # validate the model #

    ######################

    model.eval()

    for data, target in valid_loader:

        # move tensors to GPU if CUDA is available

        if train_on_gpu:

            data, target = data.cuda(), target.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the batch loss

        loss = criterion(output, target)

        # update average validation loss 

        valid_loss += loss.item()*data.size(0)

    

    # calculate average losses

    train_loss = train_loss/len(train_loader.sampler)

    valid_loss = valid_loss/len(valid_loader.sampler)

    train_losses.append(train_loss)

    valid_losses.append(valid_loss)

        

    # print training/validation statistics 

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

        epoch, train_loss, valid_loss))

    

    # save model if validation loss has decreased

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,

        valid_loss))

        torch.save(model.state_dict(), 'best_model.pt')

        valid_loss_min = valid_loss
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



plt.plot(train_losses, label='Training loss')

plt.plot(valid_losses, label='Validation loss')

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend(frameon=False)
# Load Best parameters learned from training into our model to make predictions later

#best_model = model.load_state_dict(torch.load('best_model.pt'))

#model.load_state_dict(torch.load('best_model.pt'))



#model.load_state_dict(torch.load('best_model.pt'))

checkpoint = torch.load('best_model.pt')

model.load_state_dict(torch.load('best_model.pt'))
for keys in checkpoint:

    print(keys)
#final_test_np = final_test.values/255

test_tn = torch.from_numpy(df_test)
# Creating fake labels for convenience of passing into DataLoader

## CAUTION: There are other ways of doing this, I just did it this way

fake_labels = np.zeros(df_test.shape)

fake_labels = torch.from_numpy(fake_labels)
submission_tn_data = torch.utils.data.TensorDataset(test_tn, fake_labels)



submission_loader = torch.utils.data.DataLoader(submission_tn_data, batch_size = batch_size, shuffle = False)
# Making it submission ready

submission = [['ImageId', 'Label']]



        



# Turn off gradients for validation

with torch.no_grad():

    #best_model.eval()

    model.eval()

    image_id = 1

    for images, _ in submission_loader:

        if train_on_gpu:

            images = images.cuda()

        log_ps = model(images)

        ps = torch.exp(log_ps)

        top_p, top_class = ps.topk(1, dim=1)

        

        for prediction in top_class:

            submission.append([image_id, prediction.item()])

            image_id += 1
submission_df = pd.DataFrame(submission)

submission_df.columns = submission_df.iloc[0]

submission_df = submission_df.drop(0, axis=0)
submission_df.to_csv("submission.csv", index=False)